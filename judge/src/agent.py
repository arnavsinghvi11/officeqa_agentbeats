import asyncio
import csv
import io
import json
import logging
import re
from dataclasses import dataclass, field
from uuid import uuid4

import httpx
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    Artifact,
    DataPart,
    Message,
    Part,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from pydantic import BaseModel, Field

from messenger import Messenger

logger = logging.getLogger(__name__)


DATASET_URL = "https://raw.githubusercontent.com/databricks/officeqa/main/officeqa.csv"


class EvalRequest(BaseModel):
    participants: dict[str, str] = Field(description="Role to URL mapping")
    config: dict = Field(default_factory=dict)


class QuestionResult(BaseModel):
    uid: str
    question: str
    ground_truth: str
    predicted: str
    is_correct: bool
    rationale: str
    difficulty: str


class EvaluationResults(BaseModel):
    total_questions: int
    correct_answers: int
    accuracy: float
    easy_accuracy: float | None = None
    hard_accuracy: float | None = None
    results: list[QuestionResult]


def normalize_text(text: str) -> str:
    if not text:
        raise ValueError("Cannot normalize empty or None text")
    normalized = text.replace('\u2212', '-')
    normalized = normalized.replace('−', '-')
    return normalized


def extract_numbers_with_context(text: str) -> list[tuple[float, str, bool, bool]]:
    if not text:
        raise ValueError("Cannot extract numbers from empty text")
    text = normalize_text(text)
    text_no_commas = text.replace(',', '')
    numbers_with_context = []
    pattern = r'-?\d+\.?\d*%?'
    for match in re.finditer(pattern, text_no_commas):
        matched_text = match.group()
        if not matched_text or matched_text == '-':
            continue
        has_percent = matched_text.endswith('%')
        num_text = matched_text.rstrip('%')
        is_negative = num_text.startswith('-')
        try:
            num = float(num_text)
        except ValueError as e:
            raise ValueError(f"Failed to parse number from '{matched_text}': {e}") from e
        start = max(0, match.start() - 20)
        end = min(len(text_no_commas), match.end() + 20)
        context = text_no_commas[start:end].lower()
        numbers_with_context.append((num, context, has_percent, is_negative))
    return numbers_with_context


def detect_unit_in_context(context: str) -> tuple[str | None, float]:
    context_lower = context.lower()
    if re.search(r'\btrillions?\b', context_lower):
        return ('trillion', 1e12)
    if re.search(r'\bbillions?\b', context_lower) or re.search(r'\bb\b', context_lower):
        return ('billion', 1e9)
    if re.search(r'\bmillions?\b', context_lower) or re.search(r'\bm\b', context_lower):
        return ('million', 1e6)
    if re.search(r'\bthousands?\b', context_lower) or re.search(r'\bk\b', context_lower):
        return ('thousand', 1e3)
    return (None, 1.0)


def normalize_number_with_units(number: float, context: str) -> tuple[float, str | None]:
    try:
        unit_name, _ = detect_unit_in_context(context)
        return (number, unit_name)
    except Exception as e:
        raise ValueError(f"Failed to normalize number {number} with context '{context}': {e}") from e


def is_likely_year(num: float) -> bool:
    return 1900 <= num <= 2100 and num == int(num)


def has_significant_text(text: str) -> tuple[bool, str]:
    if not text:
        return False, ""
    cleaned = normalize_text(text).lower()
    cleaned = re.sub(r'-?\d+\.?\d*%?', '', cleaned)
    cleaned = re.sub(r'[,]', '', cleaned)
    unit_words = [
        'trillion', 'trillions', 'billion', 'billions', 'million', 'millions',
        'thousand', 'thousands', 'hundred', 'hundreds',
        'percent', 'percentage', '%'
    ]
    for unit in unit_words:
        cleaned = re.sub(r'\b' + unit + r'\b', '', cleaned)
    cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    has_text = len(cleaned) >= 2
    return has_text, cleaned


def check_text_overlap(gt_text: str, pred_text: str) -> tuple[bool, str]:
    if not gt_text or not pred_text:
        return False, "Empty text in comparison"
    gt_has_text, gt_cleaned = has_significant_text(gt_text)
    pred_has_text, pred_cleaned = has_significant_text(pred_text)
    if not gt_has_text:
        return True, "GT is purely numeric, text check not required"
    if not pred_has_text:
        return False, f"GT has text '{gt_cleaned}' but prediction is purely numeric"
    if gt_cleaned in pred_cleaned:
        return True, f"Text overlap: '{gt_cleaned}' found in prediction"
    if pred_cleaned in gt_cleaned:
        return True, f"Text overlap: prediction text '{pred_cleaned}' matches GT"
    return False, f"Text mismatch: GT='{gt_cleaned}', Pred='{pred_cleaned}'"


def extract_final_answer(text: str) -> str:
    if not text:
        raise ValueError("Cannot extract from empty text")
    match = re.search(r'<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>', text, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        if not content:
            raise ValueError("FINAL_ANSWER tags are empty")
        return content
    return text


def fuzzy_match_answer(ground_truth: str, predicted: str, tolerance: float = 0.05) -> tuple[bool, str]:
    if not ground_truth:
        raise ValueError("Ground truth cannot be empty")
    if not predicted:
        raise ValueError("Predicted answer cannot be empty")
    if not 0 <= tolerance <= 1:
        raise ValueError(f"Tolerance must be between 0 and 1, got {tolerance}")

    try:
        gt_numbers_with_context = extract_numbers_with_context(ground_truth)
    except Exception as e:
        raise ValueError(f"Failed to extract numbers: {e}") from e

    try:
        pred_numbers_with_context = extract_numbers_with_context(predicted)
    except Exception as e:
        raise ValueError(f"Failed to extract numbers: {e}") from e

    gt_numbers = [(num, ctx) for num, ctx, _, _ in gt_numbers_with_context]
    pred_numbers = [(num, ctx) for num, ctx, _, _ in pred_numbers_with_context]

    if gt_numbers and pred_numbers:
        if len(gt_numbers) > 1:
            pred_non_years = [(n, c) for n, c in pred_numbers
                             if not is_likely_year(n) or any(is_likely_year(g) for g, _ in gt_numbers)]
            matched_gt = []
            unmatched_gt = []
            for gt_val, gt_context in gt_numbers:
                try:
                    gt_base, gt_unit = normalize_number_with_units(gt_val, gt_context)
                except Exception as e:
                    raise ValueError(f"Failed to normalize GT number {gt_val}: {e}") from e
                found_match = False
                for pred_val, pred_context in pred_non_years:
                    try:
                        pred_base, pred_unit = normalize_number_with_units(pred_val, pred_context)
                    except Exception as e:
                        raise ValueError(f"Failed to normalize prediction number {pred_val}: {e}") from e
                    if gt_base == 0:
                        if pred_base == 0:
                            text_matches, _ = check_text_overlap(ground_truth, predicted)
                            if text_matches:
                                found_match = True
                                break
                    else:
                        diff_pct = abs(gt_base - pred_base) / abs(gt_base)
                        if diff_pct <= tolerance:
                            text_matches, _ = check_text_overlap(ground_truth, predicted)
                            if text_matches:
                                found_match = True
                                break
                if found_match:
                    matched_gt.append(gt_val)
                else:
                    unmatched_gt.append(gt_val)
            if len(matched_gt) == len(gt_numbers):
                return True, f"List match: All {len(gt_numbers)} numbers found in prediction"
            else:
                return False, f"List mismatch: Found {len(matched_gt)}/{len(gt_numbers)} numbers. Missing: {unmatched_gt}"
        else:
            gt_val, gt_context = gt_numbers[0]
            try:
                gt_base, gt_unit = normalize_number_with_units(gt_val, gt_context)
            except Exception as e:
                raise ValueError(f"Failed to normalize GT number: {e}") from e
            gt_has_text, _ = has_significant_text(ground_truth)
            should_filter_years = not (is_likely_year(gt_val) or gt_has_text)
            best_match = None
            best_diff = float('inf')
            best_pred_info = None
            for pred_val, pred_context in pred_numbers:
                if should_filter_years and is_likely_year(pred_val):
                    continue
                try:
                    pred_base, pred_unit = normalize_number_with_units(pred_val, pred_context)
                except Exception as e:
                    raise ValueError(f"Failed to normalize prediction number: {e}") from e
                if gt_base == 0:
                    if pred_base == 0:
                        text_matches, text_rationale = check_text_overlap(ground_truth, predicted)
                        if text_matches:
                            return True, f"Exact match: Found 0 in response. {text_rationale}"
                    continue
                diff_pct = abs(gt_base - pred_base) / abs(gt_base)
                if diff_pct < best_diff:
                    best_diff = diff_pct
                    best_match = pred_base
                    best_pred_info = (pred_base, pred_unit)
                if diff_pct <= tolerance:
                    text_matches, text_rationale = check_text_overlap(ground_truth, predicted)
                    if not text_matches:
                        continue
                    return True, f"Numerical match: GT={gt_base} ({gt_unit or 'no unit'}), Pred={pred_base} ({pred_unit or 'no unit'}), Diff={diff_pct*100:.2f}%. {text_rationale}"
            if best_match is not None and best_pred_info is not None:
                return False, f"No match: GT={gt_base} ({gt_unit or 'no unit'}), Closest={best_pred_info[0]} ({best_pred_info[1] or 'no unit'}), Diff={best_diff*100:.2f}%"
            else:
                return False, f"No valid numbers found in prediction (filtered out years: {[n for n, _ in pred_numbers[:5]]})"

    gt_clean = ground_truth.strip().lower().strip('"').strip("'")
    pred_clean = predicted.strip().lower().strip('"').strip("'")
    gt_clean = re.sub(r'\([^)]*\)', '', gt_clean).strip()
    pred_clean = re.sub(r'\([^)]*\)', '', pred_clean).strip()

    if gt_clean in pred_clean:
        return True, f"Text match: '{ground_truth}' found in prediction"
    if gt_clean == pred_clean:
        return True, "Exact text match"

    return False, f"No match found. GT: '{ground_truth[:100]}', Pred: '{predicted[:100]}'"


def score_answer(ground_truth: str, predicted: str, tolerance: float = 0.00) -> tuple[bool, str]:
    try:
        predicted_answer = extract_final_answer(predicted)
    except ValueError:
        return False, "Failed to extract final answer"
    try:
        return fuzzy_match_answer(ground_truth, predicted_answer, tolerance)
    except ValueError as e:
        return False, str(e)


@dataclass
class OfficeQAAgent:
    messenger: Messenger = field(default_factory=Messenger)
    questions: list[dict] = field(default_factory=list)

    def parse_request(self, message: Message) -> EvalRequest:
        for part in message.parts:
            root = part.root if hasattr(part, 'root') else part
            if isinstance(root, TextPart):
                try:
                    data = json.loads(root.text)
                    return EvalRequest(**data)
                except (json.JSONDecodeError, ValueError):
                    continue
            elif isinstance(root, DataPart):
                return EvalRequest(**root.data)
        raise ValueError("No valid evaluation request found in message")

    def validate_request(self, request: EvalRequest) -> None:
        if "officeqa_agent" not in request.participants:
            raise ValueError("Missing required participant: officeqa_agent")

    async def load_dataset(self, config: dict) -> list[dict]:
        url = config.get("dataset_url", DATASET_URL)
        num_questions = config.get("num_questions", 10)
        difficulty = config.get("difficulty", "all")

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.get(url)
            response.raise_for_status()
            content = response.text

        reader = csv.DictReader(io.StringIO(content))
        questions = []
        for row in reader:
            q = {
                "uid": row.get("uid", ""),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "source_docs": row.get("source_docs", ""),
                "source_files": row.get("source_files", ""),
                "difficulty": row.get("difficulty", "unknown"),
            }
            if difficulty != "all" and q["difficulty"] != difficulty:
                continue
            questions.append(q)
            if len(questions) >= num_questions:
                break

        return questions

    async def _emit_status(
        self,
        event_queue: EventQueue,
        task_id: str,
        context_id: str,
        state: TaskState,
        message_text: str,
        final: bool = False,
    ) -> None:
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                taskId=task_id,
                contextId=context_id,
                status=TaskStatus(
                    state=state,
                    message=Message(
                        messageId=uuid4().hex,
                        role="agent",
                        parts=[Part(root=TextPart(kind="text", text=message_text))],
                    ),
                ),
                final=final,
            )
        )

    async def _evaluate_single_question(
        self,
        q: dict,
        agent_url: str,
        tolerance: float,
    ) -> QuestionResult:
        prompt = self._build_prompt(q)
        try:
            response = await self.messenger.talk_to_agent(
                message=prompt,
                url=agent_url,
                new_conversation=True,
                timeout=120,
            )
        except Exception as e:
            logger.error(f"Failed to get response for {q['uid']}: {e}")
            response = f"Error: {e}"

        is_correct, rationale = score_answer(q["answer"], response, tolerance)

        return QuestionResult(
            uid=q["uid"],
            question=q["question"],
            ground_truth=q["answer"],
            predicted=response if response else "",
            is_correct=is_correct,
            rationale=rationale,
            difficulty=q["difficulty"],
        )

    async def evaluate_agent(
        self,
        agent_url: str,
        questions: list[dict],
        tolerance: float,
        event_queue: EventQueue,
        task_id: str,
        context_id: str,
        max_concurrent: int = 10,
    ) -> EvaluationResults:
        await self._emit_status(
            event_queue, task_id, context_id, TaskState.working,
            f"Evaluating {len(questions)} questions in parallel (max {max_concurrent} concurrent)..."
        )

        semaphore = asyncio.Semaphore(max_concurrent)
        completed = 0
        total = len(questions)
        results_list: list[QuestionResult] = []
        lock = asyncio.Lock()

        async def bounded_evaluate(q: dict) -> QuestionResult:
            nonlocal completed
            async with semaphore:
                result = await self._evaluate_single_question(q, agent_url, tolerance)
                async with lock:
                    completed += 1
                    results_list.append(result)
                    if completed % 10 == 0 or completed == total:
                        await self._emit_status(
                            event_queue, task_id, context_id, TaskState.working,
                            f"Progress: {completed}/{total} questions completed ({completed*100//total}%)"
                        )
                return result

        tasks = [bounded_evaluate(q) for q in questions]
        await asyncio.gather(*tasks)

        correct = sum(1 for r in results_list if r.is_correct)
        easy_results = [r for r in results_list if r.difficulty == "easy"]
        hard_results = [r for r in results_list if r.difficulty == "hard"]
        easy_correct = sum(1 for r in easy_results if r.is_correct)
        hard_correct = sum(1 for r in hard_results if r.is_correct)

        return EvaluationResults(
            total_questions=len(questions),
            correct_answers=correct,
            accuracy=correct / len(questions) if questions else 0,
            easy_accuracy=easy_correct / len(easy_results) if easy_results else None,
            hard_accuracy=hard_correct / len(hard_results) if hard_results else None,
            results=results_list,
        )

    def _build_prompt(self, question: dict) -> str:
        return f"""You are a helpful agent. Answer the following question about the U.S. Treasury Bulletin documents.

{question['question']}

Provide your final answer in the following format:
<FINAL_ANSWER>
[value]
</FINAL_ANSWER>"""

    async def run(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        task_id = context.task_id or "unknown"
        context_id = context.context_id or "unknown"

        await self._emit_status(
            event_queue, task_id, context_id, TaskState.working,
            "Starting OfficeQA evaluation..."
        )

        message = context.message
        if not message:
            raise ValueError("No message in context")

        request = self.parse_request(message)
        self.validate_request(request)

        config = request.config
        tolerance = config.get("tolerance", 0.0)

        await self._emit_status(
            event_queue, task_id, context_id, TaskState.working,
            "Loading OfficeQA dataset..."
        )

        questions = await self.load_dataset(config)
        logger.info(f"Loaded {len(questions)} questions")

        agent_url = request.participants["officeqa_agent"]

        await self._emit_status(
            event_queue, task_id, context_id, TaskState.working,
            f"Evaluating agent at {agent_url} on {len(questions)} questions..."
        )

        results = await self.evaluate_agent(
            agent_url=agent_url,
            questions=questions,
            tolerance=tolerance,
            event_queue=event_queue,
            task_id=task_id,
            context_id=context_id,
        )

        summary = f"""OfficeQA Evaluation Complete

Total Questions: {results.total_questions}
Correct Answers: {results.correct_answers}
Overall Accuracy: {results.accuracy:.2%}
"""
        if results.easy_accuracy is not None:
            summary += f"Easy Accuracy: {results.easy_accuracy:.2%}\n"
        if results.hard_accuracy is not None:
            summary += f"Hard Accuracy: {results.hard_accuracy:.2%}\n"

        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                taskId=task_id,
                contextId=context_id,
                artifact=Artifact(
                    artifactId=uuid4().hex,
                    name="evaluation_results",
                    parts=[Part(root=DataPart(kind="data", data=results.model_dump()))],
                ),
            )
        )

        await self._emit_status(
            event_queue, task_id, context_id, TaskState.completed,
            summary, final=True
        )
