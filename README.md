# LLM Router

An intelligent gateway that analyses every incoming prompt and automatically routes it to the right model — a fast local LLM for simple tasks, a cloud API for complex ones.

Built for developers who run powerful hardware locally (tested on a MacBook with 48 GB RAM) and want to reduce cloud API usage without sacrificing quality on tasks that actually need it.

```
Your prompt
    │
    ▼
┌─────────────────────┐
│  Complexity Scorer  │  heuristic pipeline, <1 ms
│       0 – 100       │
└─────────────────────┘
    │               │
 simple          complex
    │               │
    ▼               ▼
 Local LLM      Cloud API
(LM Studio)  (Claude / Gemini)
```

---

## Features

- **Auto-routing** — scores every prompt 0–100 on a heuristic complexity scale and sends it to the right backend
- **OpenAI-compatible proxy** — run as a server on `:8080/v1` and point any tool that talks to OpenAI directly at it, no code changes needed
- **Interactive CLI chat** — `python chat.py` for instant streaming conversations
- **Supports Claude and Gemini** as cloud backends
- **Automatic fallback** — if LM Studio is offline, requests silently fall back to the cloud
- **Force routing** — override per-request via `X-Force-Route: local` / `cloud` header or CLI flags
- **Transparent logging** — every request prints its score, target, and reason

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | 3.12+ recommended |
| [LM Studio](https://lmstudio.ai) | Free, runs Qwen / Gemma / Llama locally |
| Anthropic or Gemini API key | Only needed for the cloud backend you choose |

### Recommended local models (LM Studio)

| Model | RAM needed | Good for |
|---|---|---|
| Qwen2.5-7B-Instruct (Q5) | ~6 GB | Lighter machines |
| Qwen2.5-32B-Instruct (Q4) | ~20 GB | Best quality/speed balance |
| Gemma-3-27B-Instruct (Q4) | ~18 GB | Strong at reasoning and code |

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd llm-router

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Open `.env` and fill in your values:

```env
# Local model — match this to whatever is loaded in LM Studio
LOCAL_MODEL=qwen2.5-32b-instruct

# Choose your cloud backend: "anthropic" or "gemini"
CLOUD_PROVIDER=gemini
GEMINI_API_KEY=AIza...
# or
CLOUD_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Start LM Studio

Open LM Studio → load a model → go to **Local Server** tab → click **Start Server**.  
The server starts on `http://localhost:1234` by default.

---

## Usage

### Interactive chat (simplest)

```bash
python chat.py              # auto-route based on complexity
python chat.py --local      # force everything to the local model
python chat.py --cloud      # force everything to the cloud
```

In-chat commands:

| Command | Description |
|---|---|
| `/info` | Show current model status and config |
| `/route` | Show where the last message was routed |
| `/clear` | Reset conversation history |
| `/exit` | Quit |

### Proxy server

Run the router as a local OpenAI-compatible API server:

```bash
python server.py
```

Then point any OpenAI SDK client at `http://localhost:8080/v1`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="router")

response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Summarize this email: ..."}]
)
print(response.choices[0].message.content)
```

#### Force routing per-request

```python
# Via header
response = client.chat.completions.create(
    model="auto",
    messages=[...],
    extra_headers={"X-Force-Route": "local"},  # or "cloud"
)
```

#### Available endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Status page |
| `GET` | `/health` | Liveness + model availability |
| `GET` | `/v1/models` | Lists available models |
| `POST` | `/v1/chat/completions` | Main routing endpoint |
| `POST` | `/classify` | Inspect routing decision without calling a model |
| `GET` | `/docs` | Interactive API docs (FastAPI) |

---

## How the classifier works

Every prompt is scored 0–100 using a fast heuristic pipeline (no model call needed):

| Signal | Points | Examples |
|---|---|---|
| Token count | 0–25 | Long prompts are more likely complex |
| Complex keywords | 0–50 | `implement`, `design`, `debug`, `prove`, `analyze` |
| Simple keywords | −0–15 | `summarize`, `translate`, `fix grammar`, `what is` |
| Structural signals | 0–25 | Code blocks, tables, JSON, multiple `?` |
| Reasoning cues | 0–20 | `first … then`, `however`, `on the other hand` |

Scores at or above `COMPLEXITY_THRESHOLD` (default: **50**) go to the cloud.

### Tuning

```env
# Send more to local (trust your local model more)
COMPLEXITY_THRESHOLD=65

# Send more to cloud (higher bar for local)
COMPLEXITY_THRESHOLD=35

# Hard cap: prompts longer than this always go to cloud
MAX_LOCAL_TOKENS=3000
```

---

## Project structure

```
llm-router/
├── chat.py              # Interactive CLI chat
├── server.py            # FastAPI proxy server
├── config.py            # Centralised configuration
├── router/
│   ├── classifier.py    # Heuristic complexity scorer
│   ├── router.py        # Routing logic + availability cache
│   ├── local_client.py  # LM Studio adapter (OpenAI SDK)
│   └── cloud_client.py  # Claude + Gemini adapters
├── test_router.py       # Classifier test suite
├── requirements.txt
├── .env.example         # Copy to .env and fill in keys
└── .gitignore
```

---

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `LOCAL_BASE_URL` | `http://localhost:1234/v1` | LM Studio API endpoint |
| `LOCAL_MODEL` | `qwen2.5-32b-instruct` | Must match the model name in LM Studio |
| `CLOUD_PROVIDER` | `anthropic` | `anthropic` or `gemini` |
| `ANTHROPIC_API_KEY` | — | Your Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Claude model ID |
| `GEMINI_API_KEY` | — | Your Google AI API key |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model ID |
| `COMPLEXITY_THRESHOLD` | `50` | Score cutoff for cloud routing (0–100) |
| `MAX_LOCAL_TOKENS` | `3000` | Prompts longer than this always go to cloud |
| `SERVER_PORT` | `8080` | Proxy server port |
| `LOG_DECISIONS` | `true` | Print routing decision per request |

---

## Running the classifier tests

```bash
python test_router.py           # classifier accuracy only
python test_router.py --live    # classifier + real model calls
```
