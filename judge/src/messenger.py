import json
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import DataPart, Message, Part, Role, TextPart


DEFAULT_TIMEOUT = 300


def create_message(*, role: Role = Role.user, text: str, context_id: str | None = None) -> Message:
    return Message(
        kind="message",
        role=role,
        parts=[Part(root=TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
        context_id=context_id,
    )


def merge_parts(parts: list[Part]) -> str:
    chunks: list[str] = []
    for part in parts:
        root = part.root if hasattr(part, 'root') else part
        if isinstance(root, TextPart):
            chunks.append(root.text)
        elif isinstance(root, DataPart):
            chunks.append(json.dumps(root.data, indent=2))
    return "\n".join(chunks)


async def send_message(
    message: str,
    base_url: str,
    context_id: str | None = None,
    streaming: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
):
    async with httpx.AsyncClient(timeout=timeout) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=streaming)
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        outbound_msg = create_message(text=message, context_id=context_id)
        outputs: dict[str, object] = {"response": "", "context_id": None}
        last_event = None

        async for event in client.send_message(outbound_msg):
            last_event = event

        match last_event:
            case Message() as msg:
                outputs["context_id"] = msg.context_id
                outputs["response"] = str(outputs["response"]) + merge_parts(msg.parts)
            case (task, _update):
                outputs["context_id"] = task.context_id
                outputs["status"] = task.status.state.value
                status_msg = task.status.message
                if status_msg:
                    outputs["response"] = str(outputs["response"]) + merge_parts(status_msg.parts)
                if task.artifacts:
                    for artifact in task.artifacts:
                        outputs["response"] = str(outputs["response"]) + merge_parts(artifact.parts)
            case _:
                pass

        return outputs


class Messenger:
    def __init__(self):
        self._context_ids: dict[str, str | None] = {}

    async def talk_to_agent(
        self,
        message: str,
        url: str,
        new_conversation: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> str:
        outputs = await send_message(
            message=message,
            base_url=url,
            context_id=None if new_conversation else self._context_ids.get(url),
            timeout=timeout,
        )
        if outputs.get("status", "completed") != "completed":
            raise RuntimeError(f"{url} responded with: {outputs}")
        self._context_ids[url] = outputs.get("context_id")
        return str(outputs["response"])

    def reset(self) -> None:
        self._context_ids = {}
