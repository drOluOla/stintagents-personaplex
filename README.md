# PersonaPlex on RunPod

A simple demonstation of how personaplex can be hosted to serve voices/audio for future StintAgents implementation.
This setup runs the PersonaPlex speech-to-speech model on a RunPod GPU pod and connects it to LiveKit as a native realtime model, with an optional HTTP endpoint for one-shot text requests.

- Downloads and loads `nvidia/personaplex-7b-v1` weights from Hugging Face on RunPod
- Hosts full-duplex model inference on a RunPod GPU
- Implements `PersonaPlexRealtimeModel` / `PersonaPlexRealtimeSession` — a drop-in LiveKit `RealtimeModel` that streams audio bidirectionally over WebSocket
- Optionally exposes `POST /v1/respond` for text-in / text-out requests

## Files in this workspace

- `presonaplex.py`: PersonaPlex WebSocket bridge, `PersonaPlexRealtimeModel`, `PersonaPlexRealtimeSession`, HTTP bridge API, and LiveKit agent entrypoint
- `Dockerfile.runpod`: Container image for RunPod
- `runpod_start.sh`: Starts the PersonaPlex model server (`moshi.server`) and the HTTP bridge
- `runpod_serverless.py`: Optional RunPod serverless entrypoint

## 1) Prepare Hugging Face access

1. Accept the model license for `nvidia/personaplex-7b-v1` on Hugging Face.
2. Create an HF token.
3. In RunPod, set the env var:

```bash
HF_TOKEN=<your_hf_token>
```

## 2) Deploy on RunPod (GPU pod)

Build and push the image from this folder:

```bash
docker build -f Dockerfile.runpod -t stintagents/personaplex:latest .
docker push stintagents/personaplex:latest
```

Create a RunPod GPU pod from that image and expose ports `8000` (HTTP bridge) and `8998` (model WebSocket).

At startup, `runpod_start.sh`:

- Launches `python -m moshi.server` on `:8998` (pulls weights from HF on first boot via `HF_TOKEN`)
- Starts the HTTP bridge on `:8000`

For RTX 5090 / Blackwell pods, the image installs PyTorch nightly CUDA wheels and `runpod_start.sh` verifies `sm_120` support before starting the model server.

If VRAM is low, set:

```bash
PERSONAPLEX_EXTRA_ARGS=--cpu-offload
```

## 3) LiveKit realtime mode (primary)

`PersonaPlexRealtimeModel` connects directly to the PersonaPlex WebSocket and streams audio bidirectionally — no separate STT, LLM, or TTS stages needed.

Set env vars and run:

```bash
export PERSONAPLEX_WS_URL="wss://<your-runpod-endpoint>/api/chat"
export PERSONAPLEX_TEXT_PROMPT="You enjoy having a good conversation."
export PERSONAPLEX_VOICE_PROMPT="NATF0.pt"   # optional, default: NATF0.pt
export PERSONAPLEX_SEED="-1"                  # optional, -1 = random
export LIVEKIT_URL="wss://..."
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
python presonaplex.py livekit dev
```

Below is what I currently use:
```bash
kill -9 $(pgrep -f "presonaplex.py") 2>/dev/null; sleep 1
LIVEKIT_URL=ws://localhost:7880 \
LIVEKIT_API_KEY=devkey \
LIVEKIT_API_SECRET=secret \
PERSONAPLEX_WS_URL=wss://0ykihj6mq4at8f-8998.proxy.runpod.net/api/chat \
PERSONAPLEX_TEXT_PROMPT="This is an onboarding meeting and you are the HR manager. You must welcome the new employee by saying 'Hello! Hi!  Welcome to the team. My name is Sarah and I will be working you through the onboarding process.'" \
PERSONAPLEX_VOICE_PROMPT=NATF0.pt \
/home/oluseyi/Olu-Projects/StintAgents/Experimental/personaplex/bin/python presonaplex.py livekit start

The agent uses `PersonaPlexRealtimeModel` with full-duplex audio — user audio flows to the server continuously and model audio is forwarded to the LiveKit room in real time.
```

## 4) HTTP bridge mode (optional)

The HTTP bridge provides a text-in / text-out interface. It sends silent audio to PersonaPlex over WebSocket and collects the model's text token output.

Start the bridge locally:

```bash
export PERSONAPLEX_WS_URL="wss://<your-runpod-endpoint>/api/chat"
python presonaplex.py serve-http --host 0.0.0.0 --port 8000
```

Call the endpoint:

```bash
curl -X POST https://<your-host>:8000/v1/respond \
  -H 'content-type: application/json' \
  -d '{
    "user_text": "Hello there",
    "text_prompt": "You enjoy having a good conversation.",
    "voice_prompt": "NATF0.pt",
    "text_temperature": 0.7,
    "text_topk": 25,
    "audio_temperature": 0.8,
    "audio_topk": 250,
    "pad_mult": 0,
    "repetition_penalty_context": 64,
    "repetition_penalty": 1.0,
    "seed": -1
  }'
```

Response:

```json
{
  "reply_text": "...",
  "latency_ms": 1234
}
```

A `GET /health` endpoint is also available.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | *(required)* | Hugging Face token for downloading model weights |
| `PERSONAPLEX_WS_URL` | `ws://127.0.0.1:8998/api/chat` | WebSocket URL of the running PersonaPlex server |
| `PERSONAPLEX_TEXT_PROMPT` | `"You enjoy having a good conversation."` | Persona / system prompt |
| `PERSONAPLEX_VOICE_PROMPT` | `NATF0.pt` | Voice embedding file on the server |
| `PERSONAPLEX_SEED` | `-1` | RNG seed (-1 = random) |
| `PERSONAPLEX_MAX_WAIT_SECONDS` | `8` | Timeout for HTTP bridge text roundtrip |
| `PERSONAPLEX_EXTRA_ARGS` | *(unset)* | Extra args passed to `moshi.server` (e.g. `--cpu-offload`) |

## Optional: RunPod serverless

If you prefer RunPod serverless workers, use `runpod_serverless.py` as the entrypoint.
It calls `runpod_handler(job)` from `presonaplex.py`, which runs a text roundtrip via the HTTP bridge logic and returns `{ "reply_text": "...", "latency_ms": ... }`.
