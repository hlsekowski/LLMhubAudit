# LLMhubAudit
Audits configured LLM models against what is actually available and fits, for hosted free endpoints and local Ollama

## Ollama module (`llmhubaudit.ollama`)

Pulls, tests, and prunes Ollama models that support tool calling. Measures
cold/warm latency, resource usage (RSS, CPU, swap), and stores results in a
CSV. Implements resumption, interruption handling, and a composite score.

Install the Ollama extra:

```bash
pip install -e ".[ollama]"
```

Run the CLI:

```bash
python -m llmhubaudit.ollama.cli <model> [options]
```

See [`docs/ollama.md`](docs/ollama.md) for the full CLI reference, status
codes, and module layout.

