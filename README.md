# PersonaPlex model on RunPod + callable endpoint

This setup does exactly what you asked:

- Downloads/loads PersonaPlex weights from Hugging Face on RunPod
- Hosts model inference on RunPod GPU
- Exposes one HTTP endpoint: `POST /v1/respond`
- Lets your LiveKit agent call that endpoint

## Files in this workspace

- `presonaplex.py`: PersonaPlex bridge API + LiveKit tool integration
- `Dockerfile.runpod`: Container image for RunPod
- `runpod_start.sh`: Starts PersonaPlex model server + bridge endpoint
- `runpod_serverless.py`: Optional RunPod serverless entrypoint

## 1) Prepare Hugging Face access

1. Accept model license for `nvidia/personaplex-7b-v1` on Hugging Face.
2. Create an HF token.
3. In RunPod, set env var:

```bash
HF_TOKEN=<your_hf_token>
```

## 2) Deploy on RunPod (GPU pod endpoint)

Build/push your image from this folder:

```bash
docker build -f Dockerfile.runpod -t <your-registry>/personaplex:latest .
docker push <your-registry>/personaplex:latest
```

Create a RunPod GPU pod from that image and expose port `8000`.

At startup:

- `runpod_start.sh` launches `python -m moshi.server` on `:8998`
- `moshi.server` pulls weights from HF on first boot via `HF_TOKEN`
- Bridge API starts on `:8000`

For RTX 5090 / Blackwell pods, this image installs PyTorch nightly CUDA wheels and `runpod_start.sh` verifies `sm_120` support before starting the model server.

If VRAM is low, set:

```bash
PERSONAPLEX_EXTRA_ARGS=--cpu-offload
```

## 3) Call your RunPod endpoint

Use your RunPod pod URL (or proxy URL) pointing to port `8000`:

```bash
curl -X POST https://<your-runpod-endpoint>/v1/respond \
  -H 'content-type: application/json' \
  -d '{
    "user_text": "Hello there",
    "text_prompt": "You enjoy having a good conversation.",
    "voice_prompt": "NATF0.pt"
  }'
```

## 4) Connect from LiveKit agent

In your LiveKit runtime:

```bash
export PERSONAPLEX_API_URL="https://<your-runpod-endpoint>/v1/respond"
export LIVEKIT_URL="..."
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
python presonaplex.py livekit dev
```

Your `ask_personaplex` tool in `presonaplex.py` will call that endpoint directly.

## Optional: RunPod serverless

If you prefer RunPod serverless workers, use `runpod_serverless.py` as entrypoint.
It uses `runpod_handler(job)` from `presonaplex.py`.
