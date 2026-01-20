import argparse
import logging

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill

from executor import Executor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="OfficeQA Judge (Green Agent)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind to")
    parser.add_argument("--card-url", default=None, help="URL for agent card")
    args = parser.parse_args()

    skill = AgentSkill(
        id="officeqa_evaluation",
        name="OfficeQA Benchmark Evaluation",
        description=(
            "Evaluates document understanding and reasoning capabilities using "
            "the OfficeQA benchmark. Tests agents on complex questions requiring "
            "extraction and computation from U.S. Treasury Bulletin documents."
        ),
        tags=["benchmark", "document-qa", "reasoning", "financial-data"],
        examples=[
            "Evaluate agent on OfficeQA benchmark with 10 questions",
            "Run full OfficeQA evaluation on document understanding agent",
        ],
    )

    agent_card = AgentCard(
        name="OfficeQA Judge",
        description=(
            "Green agent that evaluates purple agents on the OfficeQA benchmark. "
            "Tests document understanding and reasoning over U.S. Treasury Bulletins."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}",
        version="1.0.0",
        skills=[skill],
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities={"streaming": True},
    )

    task_store = InMemoryTaskStore()
    executor = Executor()
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )

    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    logger.info(f"Starting OfficeQA Judge on {args.host}:{args.port}")
    uvicorn.run(app.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
