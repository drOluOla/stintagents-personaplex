# PersonaPlex × LiveKit — Agent Handoff

## What the codebase does

`presonaplex.py` is a single-file Python service with three independent modes:

| Mode | Command | Purpose |
|---|---|---|
| HTTP bridge | `python presonaplex.py serve-http` | REST API wrapping the PersonaPlex WebSocket — for text-in / text-out use cases |
| RunPod serverless | via `runpod_serverless.py` | Serverless job handler calling the same bridge logic |
| LiveKit realtime | `python presonaplex.py livekit` | Full-duplex voice agent using PersonaPlex as the end-to-end model |

---

## Architecture of the LiveKit mode

PersonaPlex is a full-duplex speech-to-speech model (NVIDIA, Moshi-based). The implementation bypasses the typical STT → LLM → TTS pipeline entirely. Instead:

```
LiveKit room mic
      │  PCM (int16, 24kHz)
      ▼
PersonaPlexRealtimeSession.push_audio()
      │  Opus encode (PyAV libopus)
      │  binary frame:  0x01 + <opus bytes>
      ▼
WebSocket  ws://<PERSONAPLEX_WS_URL>/api/chat?text_prompt=...&voice_prompt=...
      │
      │  binary frame:  0x01 + <opus bytes>  ← model audio out
      │  binary frame:  0x02 + <utf8>        ← model text tokens
      ▼
PersonaPlexRealtimeSession._recv_loop()
      │  Opus decode → rtc.AudioFrame
      ▼
LiveKit room speaker
```

**Protocol (PersonaPlex/Moshi binary WebSocket):**

| Direction | Byte | Payload | Meaning |
|---|---|---|---|
| Server → Client | `0x00` | — | Handshake: model warm and ready |
| Client → Server | `0x01` | Opus bytes | User audio input |
| Server → Client | `0x01` | Opus bytes | Model audio output |
| Server → Client | `0x02` | UTF-8 text | Model text token |

---

## Key classes (inside `build_livekit_server()`)

### `PersonaPlexRealtimeModel` — `livekit.agents.llm.RealtimeModel`

- Holds all generation parameters (`text_prompt`, `voice_prompt`, temperatures, topk, seed)
- `session()` creates a `PersonaPlexRealtimeSession`
- Declared capabilities:
  - `audio_output = True`
  - `turn_detection = False` — PersonaPlex handles its own full-duplex turn-taking
  - `user_transcription = False`
  - `manual_function_calls = False` — no tool calling

### `PersonaPlexRealtimeSession` — `livekit.agents.llm.RealtimeSession`

| Method / coroutine | Role |
|---|---|
| `_run()` | Connects to PersonaPlex WS, waits for `0x00` handshake, fires `generation_created` to start the LiveKit pipeline |
| `_send_loop()` | Drains `_out_queue`, sends `0x01 + opus` to PersonaPlex |
| `_recv_loop()` | Receives `0x01`/`0x02` frames; pushes decoded `rtc.AudioFrame` and text tokens into `utils.aio.Chan` channels |
| `push_audio()` | Receives `rtc.AudioFrame` from LiveKit, encodes to Opus via PyAV, queues for sending |
| `_encode_frame()` | PCM int16 → Opus via PyAV `libopus`, 24kHz mono |
| `_decode_opus()` | Opus bytes → `rtc.AudioFrame` via PyAV `libopus` |

---

## Deployment topology

```
┌─────────────────────────────────┐        ┌───────────────────────────────────┐
│  LiveKit Agent (CPU instance)   │        │  RunPod GPU pod                   │
│                                 │        │                                   │
│  python presonaplex.py livekit  │◄──WS──►│  python -m moshi.server           │
│                                 │        │  (PersonaPlex 7B weights)         │
│  PersonaPlexRealtimeModel       │        │  port 8998 /api/chat              │
└─────────────────────────────────┘        └───────────────────────────────────┘
         ▲                                           ▲
         │ WebRTC audio                              │ HF_TOKEN (model download)
         ▼                                           │
  LiveKit Room / SIP                        HuggingFace Hub
  (browser / phone)                         nvidia/personaplex-7b-v1
```

RunPod hosts the full PersonaPlex 7B weights and runs the Moshi-compatible WS server. The LiveKit agent (running on a separate, small CPU instance) connects to the RunPod WS endpoint via `PERSONAPLEX_WS_URL`.

---

## Environment variables

### LiveKit agent (CPU side)

| Variable | Default | Description |
|---|---|---|
| `PERSONAPLEX_WS_URL` | `ws://127.0.0.1:8998/api/chat` | RunPod WebSocket endpoint URL |
| `PERSONAPLEX_TEXT_PROMPT` | `You enjoy having a good conversation.` | Persona / system prompt |
| `PERSONAPLEX_VOICE_PROMPT` | `NATF0.pt` | Voice embedding filename |
| `PERSONAPLEX_SEED` | `-1` (random) | RNG seed for reproducibility |
| `LIVEKIT_URL` | — | LiveKit server URL |
| `LIVEKIT_API_KEY` | — | LiveKit API key |
| `LIVEKIT_API_SECRET` | — | LiveKit API secret |

### RunPod pod (GPU side)

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | ✅ | Hugging Face token to download `nvidia/personaplex-7b-v1` |
| `PERSONAPLEX_HOST` | `0.0.0.0` | Bind host for moshi.server |
| `PERSONAPLEX_PORT` | `8998` | Bind port for moshi.server |
| `PERSONAPLEX_EXTRA_ARGS` | — | Extra flags passed to `python -m moshi.server` (e.g. `--cpu-offload`) |

---

## Voice embeddings

PersonaPlex ships pre-built embeddings (pass filename as `PERSONAPLEX_VOICE_PROMPT`):

```
Natural (female): NATF0.pt  NATF1.pt  NATF2.pt  NATF3.pt
Natural (male):   NATM0.pt  NATM1.pt  NATM2.pt  NATM3.pt
Variety (female): VARF0.pt  VARF1.pt  VARF2.pt  VARF3.pt  VARF4.pt
Variety (male):   VARM0.pt  VARM1.pt  VARM2.pt  VARM3.pt  VARM4.pt
```

---

## File inventory

| File | Purpose |
|---|---|
| `presonaplex.py` | Main service: HTTP bridge + RunPod handler + LiveKit realtime agent |
| `runpod_serverless.py` | RunPod serverless entrypoint (`runpod.serverless.start`) |
| `runpod_start.sh` | Pod startup: Blackwell PyTorch check → `moshi.server` → HTTP bridge |
| `Dockerfile.runpod` | CUDA 12.4 image; installs PersonaPlex from GitHub, PyTorch nightly (cu128) |
| `requirements.txt` | Python dependencies for the agent/bridge side |

---

## Python dependencies (agent side)

```
fastapi>=0.115.0
uvicorn>=0.30.0
websockets>=12.0
httpx>=0.27.0
pydantic>=2.7.0
python-dotenv>=1.0.0
numpy>=1.26.0
av>=11.0.0        # PyAV — Opus encode/decode via bundled libopus
livekit-agents[openai,silero,deepgram,cartesia,turn-detector]~=1.4
runpod>=1.7.0
```

---

## References

- PersonaPlex GitHub: <https://github.com/NVIDIA/personaplex>
- Model weights: <https://huggingface.co/nvidia/personaplex-7b-v1>
- Paper: <https://arxiv.org/abs/2602.06053>
- Moshi architecture (base): <https://arxiv.org/abs/2410.00037>
- LiveKit Agents: <https://github.com/livekit/agents>
