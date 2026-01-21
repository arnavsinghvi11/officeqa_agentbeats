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
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


SYSTEM_PROMPT = """You are a helpful agent that answers questions about the U.S. Treasury Bulletin documents. Ensure numerical accuracy and full precision in calculations while answering the question.

Provide your final answer in the following format:
<REASONING>
[steps, calculations and references used]
</REASONING>
<FINAL_ANSWER>
[value]
</FINAL_ANSWER>"""


def get_llm_response(prompt: str) -> str:
    if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
        client = OpenAI()
        response = client.chat.completions.create(
            model=os.environ["OPENAI_MODEL"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content or ""

    if GOOGLE_AVAILABLE and os.environ.get("GOOGLE_API_KEY"):
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel(os.environ["GOOGLE_MODEL"])
        response = model.generate_content(prompt)
        return response.text or ""

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
            response = get_llm_response(question_text)
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
