import asyncio
import logging
import os
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    Message,
    Part,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
    UnsupportedOperationError,
)

logger = logging.getLogger(__name__)

TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


SYSTEM_PROMPT = """You are a helpful agent that answers questions about the U.S. Treasury Bulletin. Ensure numerical accuracy and full precision in calculations while answering the question.

Provide your final answer in the following required format:
<REASONING>
[steps and calculations]
</REASONING>
<FINAL_ANSWER>
[value]
</FINAL_ANSWER>

If you do not produce a <FINAL_ANSWER> tag with the canonical final answer enclosed, your response will be considered incorrect.

"""



def get_llm_response(prompt: str) -> str:
    provider = os.environ.get("LLM_PROVIDER", "").lower()

    use_openai = (
        OPENAI_AVAILABLE and
        os.environ.get("OPENAI_API_KEY") and
        (provider == "openai" or (provider == "" and not os.environ.get("ANTHROPIC_API_KEY")))
    )
    use_anthropic = (
        ANTHROPIC_AVAILABLE and
        os.environ.get("ANTHROPIC_API_KEY") and
        (provider == "anthropic" or (provider == "" and not use_openai))
    )

    if use_openai:
        client = OpenAI()
        model = os.environ["OPENAI_MODEL"]

        if model.startswith("gpt-5"):
            reasoning_effort = os.environ.get("REASONING_EFFORT", "")
            enable_web_search = os.environ.get("ENABLE_WEB_SEARCH", "false").lower() == "true"
            tools = [{"type": "web_search"}] if enable_web_search else None
            kwargs = {
                "model": model,
                "instructions": SYSTEM_PROMPT,
                "input": [{"role": "user", "content": prompt}],
                "tools": tools,
            }
            if reasoning_effort:
                kwargs["reasoning"] = {"effort": reasoning_effort}
            else:
                kwargs["temperature"] = 0
            response = client.responses.create(**kwargs)
            return response.output_text or ""
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            return response.choices[0].message.content or ""

    if use_anthropic:
        client = anthropic.Anthropic()
        model = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-5-20251101")
        max_tokens = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "16000"))
        enable_web_search = os.environ.get("ENABLE_WEB_SEARCH", "false").lower() == "true"
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        if enable_web_search:
            kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 10}]
        response = client.messages.create(**kwargs)
        text_parts = [block.text for block in response.content if hasattr(block, 'text')]
        return "\n".join(text_parts) if text_parts else ""

    return "<FINAL_ANSWER>Unable to determine - no LLM configured</FINAL_ANSWER>"


class Executor(AgentExecutor):
    def __init__(self):
        self._contexts: dict[str, list[dict]] = {}

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        message = context.message
        if not message or not message.parts:
            logger.warning("Received empty message")
            return

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            logger.info(f"Task {task.id} already in terminal state")
            return

        task_id = context.task_id or "unknown"
        context_id = context.context_id or "unknown"

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                taskId=task_id,
                contextId=context_id,
                status=TaskStatus(
                    state=TaskState.working,
                    message=Message(
                        messageId=uuid4().hex,
                        role="agent",
                        parts=[Part(root=TextPart(kind="text", text="Processing question..."))],
                    ),
                ),
                final=False,
            )
        )

        question_text = ""
        for part in message.parts:
            root = part.root if hasattr(part, 'root') else part
            if isinstance(root, TextPart):
                question_text = root.text
                break

        try:
            response = await asyncio.to_thread(get_llm_response, question_text)
        except Exception as e:
            logger.exception(f"LLM call failed: {e}")
            response = f"<FINAL_ANSWER>Error: {e}</FINAL_ANSWER>"

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                taskId=task_id,
                contextId=context_id,
                status=TaskStatus(
                    state=TaskState.completed,
                    message=Message(
                        messageId=uuid4().hex,
                        role="agent",
                        parts=[Part(root=TextPart(kind="text", text=response))],
                    ),
                ),
                final=True,
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError(message="Cancellation not supported")
