# OfficeQA AgentBeats Benchmark

A green agent implementation for the [AgentBeats Competition](https://rdi.berkeley.edu/agentx-agentbeats.html) (Phase 1 - Green) that evaluates document understanding and reasoning capabilities using the OfficeQA benchmark.

## Overview

**OfficeQA** is a grounded reasoning benchmark that tests AI systems on complex questions requiring extraction and computation from real-world financial documents (U.S. Treasury Bulletins from 1939-2025).

This submission ports the OfficeQA benchmark to the AgentBeats platform, providing:
- A **Green Agent (Judge)** that orchestrates evaluations
- A **Baseline Purple Agent** for demonstration
- Automated scoring using fuzzy matching with configurable tolerance

## Benchmark Details

| Metric | Value |
|--------|-------|
| Total Questions | 246 |
| Document Source | U.S. Treasury Bulletins |
| Time Span | January 1939 - September 2025 |
| Difficulty Levels | Easy, Hard |
| Question Types | Extraction, Calculation, Statistical Analysis |

### Question Categories
- Simple data extraction
- Multi-year calculations with inflation adjustments
- Statistical analysis (regression, correlation, standard deviation)
- Time series forecasting
- Complex financial metrics (VaR, weighted averages)

## Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker (for containerized deployment)

### Local Development

1. Clone and setup:
```bash
git clone https://github.com/arnavsinghvi11/officeqa_agentbeats.git
cd officeqa_agentbeats
cp sample.env .env
```

2. Install dependencies:
```bash
uv sync --extra judge --extra participant
```

3. Run the evaluation:
```bash
uv run agentbeats-run scenario.toml
```

Or manually start each agent:
```bash
# Terminal 1: Start the judge (green agent)
uv run python judge/src/server.py --host 127.0.0.1 --port 9009

# Terminal 2: Start the participant (purple agent)
uv run python participant/src/server.py --host 127.0.0.1 --port 9019
```

### Docker Deployment

Build images:
```bash
docker build --platform linux/amd64 -t officeqa-judge -f Dockerfile.officeqa-judge .
docker build --platform linux/amd64 -t officeqa-agent -f Dockerfile.officeqa-agent .
```

Run:
```bash
# Judge
docker run -p 9009:9009 officeqa-judge

# Participant (with API key)
docker run -p 9019:9019 -e OPENAI_API_KEY=$OPENAI_API_KEY officeqa-agent
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Green Agent (Judge)                   │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │   server.py │  │  executor.py │  │   agent.py    │  │
│  │  A2A Server │──│ Task Handler │──│ Orchestration │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
│                                            │            │
│                                            ▼            │
│                                   ┌───────────────┐    │
│                                   │ messenger.py  │    │
│                                   │   A2A Client  │    │
│                                   └───────────────┘    │
└───────────────────────────────────────────┬─────────────┘
                                            │
                                            ▼ A2A Protocol
┌─────────────────────────────────────────────────────────┐
│                 Purple Agent (Participant)               │
│  ┌─────────────┐  ┌──────────────┐                     │
│  │   server.py │  │  executor.py │                     │
│  │  A2A Server │──│  LLM Calls   │                     │
│  └─────────────┘  └──────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

## Evaluation Protocol

1. Judge loads questions from OfficeQA dataset
2. For each question:
   - Send question with source document references to purple agent
   - Receive answer (expecting `<FINAL_ANSWER>` tags)
   - Score using fuzzy matching against ground truth
3. Report aggregate results as artifacts

### Scoring Criteria
- **Numerical answers**: Fuzzy matching with unit awareness (million, billion, etc.)
- **Text answers**: Case-insensitive exact match
- **Hybrid answers**: Both text and number components must match

## Configuration

Edit `scenario.toml` to customize:

```toml
[config]
num_questions = 10      # Number of questions to evaluate
difficulty = "all"      # "easy", "hard", or "all"
tolerance = 0.0         # Numerical tolerance (0.0 = exact, 0.05 = 5%)
```

## API Keys

The baseline purple agent supports:
- **OpenAI**: Set `OPENAI_API_KEY` and optionally `OPENAI_MODEL`
- **Anthropic Claude**: Set `ANTHROPIC_API_KEY` and optionally `ANTHROPIC_MODEL`

## Dataset Access

The [OfficeQA Dataset](https://github.com/databricks/officeqa) is publicly available:
- **Questions**: https://github.com/databricks/officeqa/blob/main/officeqa.csv
- **Source Documents**: https://github.com/databricks/officeqa/tree/main/treasury_bulletins_parsed
- **Original PDFs**: https://github.com/databricks/officeqa/tree/main/treasury_bulletin_pdfs

## License

- **Code**: Apache 2.0
- **Dataset**: CC-BY-SA 4.0
