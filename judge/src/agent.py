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
        return ""
    return text.replace('\u2212', '-').replace('-', '-')


def extract_numbers_with_context(text: str) -> list[tuple[float, str]]:
    if not text:
        return []
    text = normalize_text(text).replace(',', '')
    numbers = []
    for match in re.finditer(r'-?\d+\.?\d*%?', text):
        matched = match.group().rstrip('%')
        if matched and matched != '-':
            try:
                num = float(matched)
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end].lower()
                numbers.append((num, context))
            except ValueError:
                continue
    return numbers


def detect_unit(context: str) -> tuple[str | None, float]:
    ctx = context.lower()
    if re.search(r'\btrillions?\b', ctx):
        return ('trillion', 1e12)
    if re.search(r'\bbillions?\b', ctx):
        return ('billion', 1e9)
    if re.search(r'\bmillions?\b', ctx):
        return ('million', 1e6)
    if re.search(r'\bthousands?\b', ctx):
        return ('thousand', 1e3)
    return (None, 1.0)


def has_significant_text(text: str) -> tuple[bool, str]:
    if not text:
        return False, ''
    cleaned = normalize_text(text).lower()
    cleaned = re.sub(r'-?\d+\.?\d*%?', '', cleaned)
    cleaned = re.sub(r'[,]', '', cleaned)
    for unit in ['trillion', 'trillions', 'billion', 'billions', 'million', 'millions',
                 'thousand', 'thousands', 'hundred', 'hundreds', 'percent', '%']:
        cleaned = re.sub(r'\b' + unit + r'\b', '', cleaned)
    cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return len(cleaned) >= 2, cleaned


def check_text_overlap(gt: str, pred: str) -> bool:
    gt_has, gt_clean = has_significant_text(gt)
    pred_has, pred_clean = has_significant_text(pred)
    if not gt_has:
        return True
    if not pred_has:
        return False
    return gt_clean in pred_clean or pred_clean in gt_clean


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    match = re.search(r'<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def fuzzy_match_answer(ground_truth: str, predicted: str, tolerance: float = 0.0) -> tuple[bool, str]:
    if not ground_truth:
        return False, "Empty ground truth"
    if not predicted:
        return False, "Empty prediction"

    if 'unable to determine' in predicted.lower():
        return False, "Answer contains 'Unable to determine'"

    predicted = extract_final_answer(predicted)
    gt_nums = extract_numbers_with_context(ground_truth)
    pred_nums = extract_numbers_with_context(predicted)

    if gt_nums and pred_nums:
        if len(gt_nums) > 1:
            matched = 0
            for gt_val, _ in gt_nums:
                for pred_val, _ in pred_nums:
                    if gt_val == 0 and pred_val == 0:
                        if check_text_overlap(ground_truth, predicted):
                            matched += 1
                            break
                    elif gt_val != 0:
                        diff = abs(gt_val - pred_val) / abs(gt_val)
                        if diff <= tolerance and check_text_overlap(ground_truth, predicted):
                            matched += 1
                            break
            if matched == len(gt_nums):
                return True, f"All {len(gt_nums)} numbers matched"
            return False, f"Matched {matched}/{len(gt_nums)} numbers"

        gt_val, _ = gt_nums[0]

        for pred_val, _ in pred_nums:
            if gt_val == 0 and pred_val == 0:
                if check_text_overlap(ground_truth, predicted):
                    return True, "Exact zero match"
            elif gt_val != 0:
                diff = abs(gt_val - pred_val) / abs(gt_val)
                if diff <= tolerance:
                    if check_text_overlap(ground_truth, predicted):
                        return True, f"Match within {diff*100:.2f}% tolerance"

        return False, f"No matching number found for {gt_val}"

    gt_clean = ground_truth.strip().lower().strip('"\'')
    pred_clean = predicted.strip().lower().strip('"\'')
    gt_clean = re.sub(r'\([^)]*\)', '', gt_clean).strip()
    pred_clean = re.sub(r'\([^)]*\)', '', pred_clean).strip()

    if gt_clean in pred_clean or gt_clean == pred_clean:
        return True, "Text match"

    return False, f"No match: GT='{ground_truth[:50]}', Pred='{predicted[:50]}'"


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

    async def evaluate_agent(
        self,
        agent_url: str,
        questions: list[dict],
        tolerance: float,
        event_queue: EventQueue,
        task_id: str,
        context_id: str,
    ) -> EvaluationResults:
        results = []
        correct = 0
        easy_correct = 0
        easy_total = 0
        hard_correct = 0
        hard_total = 0

        for i, q in enumerate(questions):
            await self._emit_status(
                event_queue, task_id, context_id, TaskState.working,
                f"Evaluating question {i+1}/{len(questions)}: {q['uid']}"
            )

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

            is_correct, rationale = fuzzy_match_answer(
                q["answer"], response, tolerance
            )

            if is_correct:
                correct += 1

            if q["difficulty"] == "easy":
                easy_total += 1
                if is_correct:
                    easy_correct += 1
            elif q["difficulty"] == "hard":
                hard_total += 1
                if is_correct:
                    hard_correct += 1

            results.append(QuestionResult(
                uid=q["uid"],
                question=q["question"],
                ground_truth=q["answer"],
                predicted=response[:500] if response else "",
                is_correct=is_correct,
                rationale=rationale,
                difficulty=q["difficulty"],
            ))

        return EvaluationResults(
            total_questions=len(questions),
            correct_answers=correct,
            accuracy=correct / len(questions) if questions else 0,
            easy_accuracy=easy_correct / easy_total if easy_total > 0 else None,
            hard_accuracy=hard_correct / hard_total if hard_total > 0 else None,
            results=results,
        )

    def _build_prompt(self, question: dict) -> str:
        return f"""You are being evaluated on the OfficeQA benchmark.

TASK: Answer the following question about U.S. Treasury Bulletin documents.

SOURCE DOCUMENTS: {question['source_files']}

QUESTION: {question['question']}

INSTRUCTIONS:
1. Analyze the question carefully
2. Use the source documents to find the answer
3. Provide your final answer in <FINAL_ANSWER></FINAL_ANSWER> tags

Your response:"""

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
