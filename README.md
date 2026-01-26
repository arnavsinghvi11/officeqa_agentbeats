# OfficeQA Benchmark AgentBeats Implementation


**OfficeQA** is a grounded reasoning benchmark that tests AI systems on complex questions requiring extraction and computation from real-world financial documents (U.S. Treasury Bulletins from 1939-2025).

This submission implements the OfficeQA benchmark on the AgentBeats platform, providing:
- A **Green Agent (Evaluator)** in `judge/` that orchestrates evaluations
- A **Baseline Purple Agent** in `participant/` for demonstration
- Automated scoring using fuzzy matching with configurable tolerance

## Benchmark Details

| Metric | Value |
|--------|-------|
| Total Questions | 246 |
| Corpus | U.S. Treasury Bulletins |
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

3. Start each agent in separate terminals:
```bash
# Terminal 1: Start the judge (green agent)
uv run python judge/src/server.py --host 127.0.0.1 --port 9009

# Terminal 2: Start the participant (purple agent)
uv run python participant/src/server.py --host 127.0.0.1 --port 9019
```

### Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/arnavsinghvi11/officeqa_agentbeats.git
cd officeqa_agentbeats

# Configure environment
cp sample.env .env
# Edit .env with your API key (OPENAI_API_KEY or ANTHROPIC_API_KEY)

# Start both agents
docker compose up

# View logs
docker compose logs -f

# Stop agents
docker compose down
```

### Building Images Locally

```bash
docker build -f Dockerfile.officeqa-judge -t ghcr.io/arnavsinghvi11/officeqa-judge:latest .
docker build -f Dockerfile.officeqa-agent -t ghcr.io/arnavsinghvi11/officeqa-agent:latest .
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
   - Send question to purple agent, which is instructed to answer the question while 
   returning its response with `<REASONING>` and `<FINAL_ANSWER>` tags for observability on its solution. 
   - Receive answer (expecting `<FINAL_ANSWER>` tags)
   - Score using fuzzy matching against ground truth
3. Report aggregate results as artifacts

### Scoring Criteria
- **Numerical answers**: Fuzzy matching with unit awareness (million, billion, etc.)
- **Text answers**: Case-insensitive exact match
- **Hybrid answers**: Both text and number components must match

## Configuration

Edit `scenario.toml` to customize benchmark and agent settings:

```toml
[config]
num_questions = 246      # Number of questions to evaluate
difficulty = "all"       # "easy", "hard", or "all"
tolerance = 0.0          # Numerical tolerance (0.0 = exact, 0.05 = 5%)

[[participants]]
name = "officeqa_agent"
image = "ghcr.io/arnavsinghvi11/officeqa-agent:latest"
env = { OPENAI_API_KEY = "${OPENAI_API_KEY}", OPENAI_MODEL = "gpt-5.2" }
```

### Supported Agent Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | - |
| `OPENAI_MODEL` | Model name | `gpt-5.2` |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `ANTHROPIC_MODEL` | Model name | `claude-opus-4-5-20251101` |
#TODO add web search mention here

## Submit Your Agent to Leaderboard

To submit your agent for evaluation on the official leaderboard:

1. Fork the [OfficeQA Leaderboard](https://github.com/arnavsinghvi11/officeqa-leaderboard)
2. Edit `scenario.toml`:
   - Set your agent's `agentbeats_id` under `[[participants]]`
   - Add API keys to your fork's GitHub Secrets
3. Push changes to trigger the assessment
4. Submit a PR with your results

### Agent Requirements

Your agent must:
- Implement the A2A protocol
- Accept questions about U.S. Treasury Bulletin documents
- Return answers wrapped in `<FINAL_ANSWER></FINAL_ANSWER>` tags

## Dataset Access

The [OfficeQA Dataset](https://github.com/databricks/officeqa) is publicly available:
- **Questions**: https://github.com/databricks/officeqa/blob/main/officeqa.csv
- **Source Documents**: https://github.com/databricks/officeqa/tree/main/treasury_bulletins_parsed
- **Original PDFs**: https://github.com/databricks/officeqa/tree/main/treasury_bulletin_pdfs

## License

- **Code**: Apache 2.0
- **Dataset**: CC-BY-SA 4.0
