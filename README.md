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

Run the full evaluation in 4 steps:

### Step 1: Clone the repository
```bash
git clone https://github.com/arnavsinghvi11/officeqa_agentbeats.git
cd officeqa_agentbeats
```

### Step 2: Create your `.env` file
```bash
echo "LLM_PROVIDER=openai" > .env
echo "OPENAI_API_KEY=<your-openai-api-key>" >> .env
echo "OPENAI_MODEL=gpt-5.2" >> .env
echo "ENABLE_WEB_SEARCH=true" >> .env
```

### Step 3: Prepare the output directory
```bash
mkdir -p output && chmod 777 output
```

### Step 4: Run the evaluation
```bash
docker compose up --abort-on-container-exit --exit-code-from agentbeats-client
```

Results are saved to `output/results.json`:
```bash
cat output/results.json
```

Clean up when done:
```bash
docker compose down
```

### Quick Test (1 Question)

To verify everything works before running the full 246-question evaluation:

```bash
sed -i 's/num_questions = 246/num_questions = 1/' a2a-scenario.toml
docker compose up --abort-on-container-exit --exit-code-from agentbeats-client
cat output/results.json
docker compose down
sed -i 's/num_questions = 1/num_questions = 246/' a2a-scenario.toml
```

### Full Evaluation (246 Questions)

Once you've verified the quick test works, run the full evaluation:

```bash
sed -i 's/num_questions = 1/num_questions = 246/' a2a-scenario.toml
docker compose up --abort-on-container-exit --exit-code-from agentbeats-client
```
(Note that it is expected for this baseline purple agent, which is just an LLM configured with no tools, to perform poorly on this benchmark. We also test with additional configurations below, and will add true agentic purple agent systems on the leaderboard that will demonstrate accurate parsing, retrieval and reasoning capabilities.)

## Configuration

### `.env` - Agent Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider to use | `openai` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `OPENAI_MODEL` | OpenAI model name | `gpt-5.2` |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `ANTHROPIC_MODEL` | Anthropic model name | `claude-opus-4-5-20251101` |
| `ENABLE_WEB_SEARCH` | Enable web search for document retrieval | `false` |

### Baseline Configurations

We provide 4 tested baseline configurations:

#### GPT-5.2 with Web Search
```bash
cat > .env << 'EOF'
LLM_PROVIDER=openai
OPENAI_API_KEY=<your-openai-api-key>
OPENAI_MODEL=gpt-5.2
ENABLE_WEB_SEARCH=true
EOF
```

#### GPT-5.2 without Tools
```bash
cat > .env << 'EOF'
LLM_PROVIDER=openai
OPENAI_API_KEY=<your-openai-api-key>
OPENAI_MODEL=gpt-5.2
ENABLE_WEB_SEARCH=false
EOF
```

#### Claude Opus 4.5 with Web Search
```bash
cat > .env << 'EOF'
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=<your-anthropic-api-key>
ANTHROPIC_MODEL=claude-opus-4-5-20251101
ENABLE_WEB_SEARCH=true
EOF
```

#### Claude Opus 4.5 without Tools
```bash
cat > .env << 'EOF'
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=<your-anthropic-api-key>
ANTHROPIC_MODEL=claude-opus-4-5-20251101
ENABLE_WEB_SEARCH=false
EOF
```

### `a2a-scenario.toml` - Evaluation Parameters

| Parameter | Description | Values |
|-----------|-------------|--------|
| `num_questions` | Number of questions to evaluate | 1-246 |
| `difficulty` | Question difficulty filter | `"easy"`, `"hard"`, `"all"` |
| `tolerance` | Numerical matching tolerance | 0.0 (exact) to 1.0 (loose) |

### `scenario.toml` - Leaderboard / GitHub Actions

Used by the GitHub Actions workflow and `generate_compose.py` for leaderboard submissions:

```toml
[green_agent]
agentbeats_id = ""
image = "ghcr.io/arnavsinghvi11/officeqa-judge:latest"

[[participants]]
agentbeats_id = ""
name = "officeqa_agent"
image = "ghcr.io/arnavsinghvi11/officeqa-agent:latest"
env = { OPENAI_API_KEY = "${OPENAI_API_KEY}" }

[config]
num_questions = 246
difficulty = "all"
tolerance = 0.0
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


## Submit Your Agent to Leaderboard

To submit your agent for evaluation on the official leaderboard:

1. Fork the [OfficeQA Leaderboard](https://github.com/arnavsinghvi11/officeqa-leaderboard)
2. Edit `scenario.toml`:
   - Set your agent's `agentbeats_id` under `[[participants]]`
   - Configure your agent's Docker image and environment variables
   - Add API keys to your fork's GitHub Secrets
3. Push changes to trigger the GitHub Actions workflow
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

## Local Development

For development and debugging without Docker:

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

1. Clone and configure:
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
uv run python judge/src/server.py --host 127.0.0.1 --port 9009

uv run python participant/src/server.py --host 127.0.0.1 --port 9019
```

### Building Images Locally

```bash
docker build -f Dockerfile.officeqa-judge -t ghcr.io/arnavsinghvi11/officeqa-judge:latest .
docker build -f Dockerfile.officeqa-agent -t ghcr.io/arnavsinghvi11/officeqa-agent:latest .
```

## Resource Requirements
- RAM: ~2GB minimum
- CPU: 1+ cores
- Network: Required for LLM API calls

## License

- **Code**: Apache 2.0
- **Dataset**: CC-BY-SA 4.0
