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
    parser = argparse.ArgumentParser(description="OfficeQA Participant (Purple Agent)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind to")
    parser.add_argument("--card-url", default=None, help="URL for agent card")
    args = parser.parse_args()

    skill = AgentSkill(
        id="document_qa",
        name="Document Question Answering",
        description=(
            "Answers questions about financial documents, particularly "
            "U.S. Treasury Bulletins. Capable of extraction, calculation, "
            "and reasoning over tabular and textual data."
        ),
        tags=["document-qa", "financial", "reasoning", "extraction"],
        examples=[
            "What were total expenditures in fiscal year 1940?",
            "Calculate the percent change in receipts between 1939 and 1940",
        ],
    )

    agent_card = AgentCard(
        name="OfficeQA Baseline Agent",
        description=(
            "Baseline purple agent for the OfficeQA benchmark. "
            "Demonstrates document understanding and reasoning capabilities."
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

    logger.info(f"Starting OfficeQA Baseline Agent on {args.host}:{args.port}")
    uvicorn.run(app.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
