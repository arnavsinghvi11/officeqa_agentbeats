import logging
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

from agent import OfficeQAAgent

logger = logging.getLogger(__name__)


TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}


class Executor(AgentExecutor):
    def __init__(self):
        self._agents: dict[str, OfficeQAAgent] = {}

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
            logger.info(f"Task {task.id} already in terminal state: {task.status.state}")
            return

        context_id = context.context_id or "default"
        if context_id not in self._agents:
            self._agents[context_id] = OfficeQAAgent()

        agent = self._agents[context_id]

        try:
            await agent.run(context, event_queue)
        except Exception as e:
            logger.exception(f"Agent execution failed: {e}")
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    taskId=context.task_id,
                    contextId=context.context_id,
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=Message(
                            messageId=uuid4().hex,
                            role="agent",
                            parts=[Part(root=TextPart(kind="text", text=f"Evaluation failed: {e}"))],
                        ),
                    ),
                    final=True,
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError(message="Cancellation not supported")
