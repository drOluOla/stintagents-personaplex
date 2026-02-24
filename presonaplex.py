import argparse
import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

logger = logging.getLogger("presonaplex")

import uuid

import av as _av
import sphn as _sphn
import httpx
import numpy as _np
import uvicorn
import websockets
from fastapi import FastAPI, HTTPException
from livekit import rtc as _rtc
from livekit.agents import (
	Agent as _Agent,
	AgentServer as _AgentServer,
	AgentSession as _AgentSession,
	JobContext as _JobContext,
	cli as _lk_cli,
	llm as _llm,
	utils as _lk_utils,
)
from livekit.agents.types import NOT_GIVEN as _NOT_GIVEN
from pydantic import BaseModel, Field


# The PersonaPlex/Moshi server recv_loop only processes kind=0x01 (user audio).
# All other client→server message kinds are logged as "unknown" and discarded.
# Control messages are kept here only for the legacy text_roundtrip path.
CONTROL_MESSAGES_MAP = {
	"start": 0,
	"endTurn": 1,
	"pause": 2,
	"restart": 3,
}


@dataclass
class BridgeSettings:
	personaplex_ws_url: str
	max_wait_seconds: float = 30.0


class PersonaPlexBridge:
	def __init__(self, settings: BridgeSettings):
		self.settings = settings

	@staticmethod
	def _encode_handshake() -> bytes:
		return bytes([0x00, 0x00, 0x00])

	@staticmethod
	def _encode_text(text: str) -> bytes:
		return bytes([0x02]) + text.encode("utf-8")

	@staticmethod
	def _encode_control(action: str) -> bytes:
		return bytes([0x03, CONTROL_MESSAGES_MAP[action]])

	def _build_chat_url(
		self,
		text_prompt: str,
		voice_prompt: str,
		text_temperature: float,
		text_topk: int,
		audio_temperature: float,
		audio_topk: int,
		pad_mult: int,
		repetition_penalty_context: int,
		repetition_penalty: float,
		seed: int,
	) -> str:
		parsed = urlparse(self.settings.personaplex_ws_url)
		params: dict = {
			"text_prompt": text_prompt,
			"voice_prompt": voice_prompt,
			"text_temperature": str(text_temperature),
			"text_topk": str(text_topk),
			"audio_temperature": str(audio_temperature),
			"audio_topk": str(audio_topk),
			"pad_mult": str(pad_mult),
			"repetition_penalty_context": str(repetition_penalty_context),
			"repetition_penalty": str(repetition_penalty),
		}
		# moshi.server has a bug: it checks `seed in request.query` but reads
		# via `request["seed"]` (state dict), causing KeyError. Omit seed
		# when -1 (random) so the server uses its own default.
		if seed != -1:
			params["seed"] = str(seed)
		query = urlencode(params)
		return urlunparse(parsed._replace(query=query))

	async def text_roundtrip(
		self,
		user_text: str,
		text_prompt: str,
		voice_prompt: str,
		text_temperature: float,
		text_topk: int,
		audio_temperature: float,
		audio_topk: int,
		pad_mult: int,
		repetition_penalty_context: int,
		repetition_penalty: float,
		seed: int,
	) -> str:
		import opuslib  # installed in container alongside libopus-dev

		ws_url = _upgrade_ws_scheme(self._build_chat_url(
			text_prompt=text_prompt,
			voice_prompt=voice_prompt,
			text_temperature=text_temperature,
			text_topk=text_topk,
			audio_temperature=audio_temperature,
			audio_topk=audio_topk,
			pad_mult=pad_mult,
			repetition_penalty_context=repetition_penalty_context,
			repetition_penalty=repetition_penalty,
			seed=seed,
		))

		# PersonaPlex is audio-in / audio-out. We send silent PCM so the model
		# can "hear" silence and speak freely based on its text_prompt persona.
		# 24 kHz mono, 20 ms frames = 480 samples of int16 zeros per frame.
		_SAMPLE_RATE = 24_000
		_FRAME_SAMPLES = 480  # 20 ms
		_SILENCE_PCM = b"\x00" * (_FRAME_SAMPLES * 2)  # int16 = 2 bytes/sample

		encoder = opuslib.Encoder(_SAMPLE_RATE, 1, opuslib.APPLICATION_VOIP)

		text_parts: list[str] = []

		try:
			async with websockets.connect(ws_url, max_size=None) as ws:
				# Wait for server's ready handshake (0x00)
				while True:
					msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
					if isinstance(msg, bytes) and len(msg) >= 1 and msg[0] == 0x00:
						break

				# Acknowledge the server handshake so it doesn't close the session
				await ws.send(self._encode_handshake())

				# Inject the user's text as a 0x02 frame and signal end-of-turn so
				# the model knows to generate a response
				await ws.send(self._encode_text(user_text))
				await ws.send(self._encode_control("endTurn"))

				# Start deadline after handshake
				deadline = time.monotonic() + self.settings.max_wait_seconds

				async def _send_silence():
					"""Stream silent Opus frames at real-time rate until deadline."""
					while time.monotonic() < deadline:
						opus_bytes = encoder.encode(_SILENCE_PCM, _FRAME_SAMPLES)
						await ws.send(b"\x01" + opus_bytes)
						await asyncio.sleep(0.02)  # 20 ms per frame

				send_task = asyncio.create_task(_send_silence())

				try:
					while time.monotonic() < deadline:
						remaining = max(0.01, deadline - time.monotonic())
						try:
							message = await asyncio.wait_for(ws.recv(), timeout=min(0.75, remaining))
						except asyncio.TimeoutError:
							if text_parts:
								break
							continue

						if not isinstance(message, bytes) or len(message) < 2:
							continue
						kind, payload = message[0], message[1:]
						if kind == 0x02:
							text_parts.append(payload.decode("utf-8", errors="ignore"))
						elif kind == 0x05:
							raise RuntimeError(payload.decode("utf-8", errors="ignore"))
				finally:
					send_task.cancel()
					try:
						await send_task
					except asyncio.CancelledError:
						pass

		except Exception as exc:
			raise RuntimeError(f"PersonaPlex websocket call failed: {exc}") from exc

		response_text = "".join(text_parts).strip()
		if not response_text:
			raise RuntimeError("PersonaPlex returned no text tokens for this turn")
		return response_text


class PersonaPlexRequest(BaseModel):
	user_text: str = Field(..., min_length=1)
	text_prompt: str = Field(
		default="You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."
	)
	voice_prompt: str = Field(default="NATF0.pt")
	text_temperature: float = Field(default=0.7)
	text_topk: int = Field(default=25)
	audio_temperature: float = Field(default=0.8)
	audio_topk: int = Field(default=250)
	pad_mult: int = Field(default=0)
	repetition_penalty_context: int = Field(default=64)
	repetition_penalty: float = Field(default=1.0)
	seed: int = Field(default=-1)


class PersonaPlexResponse(BaseModel):
	reply_text: str
	latency_ms: int


def create_app() -> FastAPI:
	ws_url = os.getenv("PERSONAPLEX_WS_URL", "ws://127.0.0.1:8998/api/chat")
	timeout_seconds = float(os.getenv("PERSONAPLEX_MAX_WAIT_SECONDS", "8"))
	bridge = PersonaPlexBridge(
		BridgeSettings(personaplex_ws_url=ws_url, max_wait_seconds=timeout_seconds)
	)

	app = FastAPI(title="PersonaPlex Bridge API", version="0.1.0")

	@app.get("/health")
	async def health() -> dict[str, str]:
		return {"status": "ok"}

	@app.post("/v1/respond", response_model=PersonaPlexResponse)
	async def respond(request: PersonaPlexRequest) -> PersonaPlexResponse:
		started = time.perf_counter()
		try:
			reply_text = await bridge.text_roundtrip(
				user_text=request.user_text,
				text_prompt=request.text_prompt,
				voice_prompt=request.voice_prompt,
				text_temperature=request.text_temperature,
				text_topk=request.text_topk,
				audio_temperature=request.audio_temperature,
				audio_topk=request.audio_topk,
				pad_mult=request.pad_mult,
				repetition_penalty_context=request.repetition_penalty_context,
				repetition_penalty=request.repetition_penalty,
				seed=request.seed,
			)
		except RuntimeError as exc:
			raise HTTPException(status_code=502, detail=str(exc)) from exc

		elapsed_ms = int((time.perf_counter() - started) * 1000)
		return PersonaPlexResponse(reply_text=reply_text, latency_ms=elapsed_ms)

	return app


def runpod_handler(job: dict[str, Any]) -> dict[str, Any]:
	payload = job.get("input", {})
	request = PersonaPlexRequest.model_validate(payload)
	ws_url = os.getenv("PERSONAPLEX_WS_URL", "ws://127.0.0.1:8998/api/chat")
	timeout_seconds = float(os.getenv("PERSONAPLEX_MAX_WAIT_SECONDS", "8"))
	bridge = PersonaPlexBridge(
		BridgeSettings(personaplex_ws_url=ws_url, max_wait_seconds=timeout_seconds)
	)

	async def _run() -> dict[str, Any]:
		start = time.perf_counter()
		text = await bridge.text_roundtrip(
			user_text=request.user_text,
			text_prompt=request.text_prompt,
			voice_prompt=request.voice_prompt,
			text_temperature=request.text_temperature,
			text_topk=request.text_topk,
			audio_temperature=request.audio_temperature,
			audio_topk=request.audio_topk,
			pad_mult=request.pad_mult,
			repetition_penalty_context=request.repetition_penalty_context,
			repetition_penalty=request.repetition_penalty,
			seed=request.seed,
		)
		return {
			"reply_text": text,
			"latency_ms": int((time.perf_counter() - start) * 1000),
		}

	return asyncio.run(_run())


def _upgrade_ws_scheme(url: str) -> str:
	"""Upgrade ws:// → wss:// for non-localhost hosts (e.g. RunPod proxies).
	Also normalises http/https → ws/wss in case the caller used the wrong scheme.
	"""
	parsed = urlparse(url)
	if parsed.scheme in ("http", "ws") and parsed.hostname not in ("localhost", "127.0.0.1", "::1", "", None):
		return urlunparse(parsed._replace(scheme="wss"))
	if parsed.scheme == "http":
		return urlunparse(parsed._replace(scheme="ws"))
	if parsed.scheme == "https":
		return urlunparse(parsed._replace(scheme="wss"))
	return url


# ── LiveKit module-level constants ────────────────────────────────────────────
_LK_WS_URL: str = os.getenv("PERSONAPLEX_WS_URL", "ws://127.0.0.1:8998/api/chat")
_LK_DEFAULT_TEXT_PROMPT: str = os.getenv(
	"PERSONAPLEX_TEXT_PROMPT", "You enjoy having a good conversation."
)
_LK_DEFAULT_VOICE_PROMPT: str = os.getenv("PERSONAPLEX_VOICE_PROMPT", "NATF0.pt")
_LK_SAMPLE_RATE: int = 24000
_LK_NUM_CHANNELS: int = 1


class PersonaPlexRealtimeSession(_llm.RealtimeSession):
	"""Full-duplex LiveKit session bridging to a PersonaPlex WebSocket endpoint.

	PersonaPlex binary WebSocket protocol (same as Moshi/kyutai):
	  Server → Client:  0x00           handshake (server ready)
	                    0x01 + <opus>  model audio output
	                    0x02 + <utf8>  model text token
	  Client → Server:  0x01 + <opus>  user audio input
	"""

	def __init__(self, model: "PersonaPlexRealtimeModel") -> None:
		super().__init__(model)
		self._model = model
		self._chat_ctx = _llm.ChatContext()
		self._tool_ctx = _llm.ToolContext([])
		self._instructions: str = ""

		# Lazy sphn Opus stream objects — created on first use
		# The server uses sphn.OpusStreamWriter/Reader which produce an Ogg/Opus
		# container stream, NOT bare Opus RTP packets. We must use the same library
		# so the wire format matches exactly.
		self._writer: _sphn.OpusStreamWriter | None = None
		self._reader: _sphn.OpusStreamReader | None = None
		# sphn requires fixed Opus frame sizes; buffer incoming PCM and flush in
		# 480-sample (20 ms @ 24 kHz) chunks — the same size the server uses.
		self._pcm_buf: _np.ndarray = _np.empty(0, dtype=_np.float32)
		self._OPUS_FRAME = 480
		# Resampler to convert LiveKit participant audio (typically 48 kHz) → 24 kHz
		self._resampler: _av.AudioResampler | None = None
		self._resampler_in_rate: int = 0

		# Outgoing Opus packets queued by push_audio; None = sentinel to stop
		self._out_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
		# Event set whenever a packet is enqueued so _send_loop wakes immediately
		self._out_event: asyncio.Event = asyncio.Event()

		# LiveKit pipeline consumes these channels for audio/text output
		self._audio_ch: _lk_utils.aio.Chan | None = None
		self._text_ch: _lk_utils.aio.Chan | None = None

		self._closed = False
		self._task = asyncio.ensure_future(self._run())

	# ── RealtimeSession ABC ──────────────────────────────────────────────────

	@property
	def chat_ctx(self) -> _llm.ChatContext:
		return self._chat_ctx

	@property
	def tools(self) -> _llm.ToolContext:
		return self._tool_ctx

	async def update_instructions(self, instructions: str) -> None:
		self._instructions = instructions

	async def update_chat_ctx(self, chat_ctx: _llm.ChatContext) -> None:
		self._chat_ctx = chat_ctx

	async def update_tools(self, tools: list) -> None:
		pass  # PersonaPlex has no tool calling

	def update_options(self, *, tool_choice=_NOT_GIVEN) -> None:
		pass

	def push_audio(self, frame: _rtc.AudioFrame) -> None:
		"""Encode incoming LiveKit PCM frame to Opus and queue for sending."""
		if self._closed:
			return
		try:
			enqueued = False
			for opus_bytes in self._encode_frame(frame):
				if opus_bytes:
					self._out_queue.put_nowait(opus_bytes)
					enqueued = True
			if enqueued:
				self._out_event.set()  # wake _send_loop immediately
		except Exception:
			logger.exception("push_audio: error encoding frame sr=%d ch=%d samples=%d",
							frame.sample_rate, frame.num_channels, frame.samples_per_channel)

	def push_video(self, frame: _rtc.VideoFrame) -> None:
		pass  # audio-only model

	def generate_reply(self, *, instructions=_NOT_GIVEN):
		"""Returns a future resolving to a GenerationCreatedEvent.

		PersonaPlex is full-duplex: the model speaks autonomously at all times.
		When the framework requests a new reply (e.g., after detecting user speech
		end) we create a fresh generation so the pipeline re-subscribes to the
		ongoing audio stream rather than blocking on an orphaned channel.
		"""
		loop = asyncio.get_event_loop()
		fut: asyncio.Future = loop.create_future()
		# Always create a new generation (old channels are closed inside
		# _make_generation_event so existing consumers unblock cleanly).
		evt = self._make_generation_event(user_initiated=True)
		logger.debug("generate_reply: created new generation (ws_connected=%s)",
					self._audio_ch is not None)
		fut.set_result(evt)
		return fut

	def commit_audio(self) -> None:
		pass

	def clear_audio(self) -> None:
		while not self._out_queue.empty():
			try:
				self._out_queue.get_nowait()
			except asyncio.QueueEmpty:
				break
		self._out_event.clear()

	def interrupt(self) -> None:
		"""Swap audio channels and re-announce the generation.

		PersonaPlex keeps streaming audio from the server regardless — it hears
		the user and naturally lowers its output.  What we must do on the client
		side is give the LiveKit pipeline a fresh GenerationCreatedEvent so it
		immediately re-subscribes to model audio.  Without this the pipeline enters
		an idle state and stops forwarding frames to the speaker even though
		_recv_loop is still writing them, causing the "no interruption / delayed
		response" symptom.
		"""
		logger.debug("interrupt() — re-subscribing pipeline to model audio stream")
		evt = self._make_generation_event(user_initiated=False)
		self.emit("generation_created", evt)

	def truncate(
		self, *, message_id, modalities, audio_end_ms, audio_transcript=_NOT_GIVEN
	) -> None:
		pass

	async def aclose(self) -> None:
		self._closed = True
		await self._out_queue.put(None)  # sentinel to stop send loop
		self._out_event.set()           # wake _send_loop so it sees the sentinel
		if self._audio_ch and not self._audio_ch.closed:
			self._audio_ch.close()
		if self._text_ch and not self._text_ch.closed:
			self._text_ch.close()
		if not self._task.done():
			self._task.cancel()
			try:
				await self._task
			except asyncio.CancelledError:
				pass

	# ── Opus codec helpers ───────────────────────────────────────────────────


	@staticmethod
	def _resample_numpy(pcm: _np.ndarray, in_rate: int, out_rate: int) -> _np.ndarray:
		"""Resample a 1-D float32 PCM array from in_rate to out_rate.

		Uses linear interpolation via numpy.interp — no extra dependencies.
		For the common 48 kHz → 24 kHz case this is an exact 2:1 decimation
		so quality is effectively identical to a proper polyphase filter.
		"""
		if in_rate == out_rate or len(pcm) == 0:
			return pcm
		n_out = max(1, int(round(len(pcm) * out_rate / in_rate)))
		x_in = _np.linspace(0.0, 1.0, len(pcm), endpoint=False)
		x_out = _np.linspace(0.0, 1.0, n_out, endpoint=False)
		return _np.interp(x_out, x_in, pcm).astype(_np.float32)

	def _init_encoder(self) -> None:
		# OpusStreamWriter produces the same Ogg/Opus container the server reads
		self._writer = _sphn.OpusStreamWriter(_LK_SAMPLE_RATE)

	def _init_decoder(self) -> None:
		# OpusStreamReader consumes the same Ogg/Opus container the server sends
		self._reader = _sphn.OpusStreamReader(_LK_SAMPLE_RATE)

	def _encode_frame(self, frame: _rtc.AudioFrame) -> list[bytes]:
		"""Convert a LiveKit AudioFrame (int16 PCM) to Ogg/Opus stream chunk(s)."""
		if self._writer is None:
			self._init_encoder()
		raw = _np.frombuffer(bytes(frame.data), dtype=_np.int16)
		nch = max(frame.num_channels, 1)
		# Downmix to mono by averaging channels (handles stereo WebRTC input)
		if nch > 1:
			raw = raw.reshape(-1, nch).mean(axis=1).astype(_np.int16)
		# Resample to 24 kHz if the incoming frame is at a different rate
		if frame.sample_rate != _LK_SAMPLE_RATE:
			if self._resampler is None or self._resampler_in_rate != frame.sample_rate:
				self._resampler = _av.AudioResampler(
					format="s16",
					layout="mono",
					rate=_LK_SAMPLE_RATE,
				)
				self._resampler_in_rate = frame.sample_rate
			av_frame = _av.AudioFrame.from_ndarray(raw.reshape(1, -1), format="s16", layout="mono")
			av_frame.sample_rate = frame.sample_rate
			resampled_frames = list(self._resampler.resample(av_frame))
			pcm = _np.concatenate(
				[f.to_ndarray().flatten().astype(_np.float32) / 32768.0
				 for f in resampled_frames]
			) if resampled_frames else _np.empty(0, dtype=_np.float32)
		else:
			pcm = raw.astype(_np.float32) / 32768.0
		if len(pcm) == 0:
			return []
		self._pcm_buf = _np.concatenate((self._pcm_buf, pcm))
		# Flush in 480-sample Opus frames
		result: list[bytes] = []
		while len(self._pcm_buf) >= self._OPUS_FRAME:
			self._writer.append_pcm(self._pcm_buf[: self._OPUS_FRAME])
			self._pcm_buf = self._pcm_buf[self._OPUS_FRAME :]
			chunk = self._writer.read_bytes()
			if chunk:
				result.append(chunk)
		return result

	def _decode_opus(self, opus_bytes: bytes) -> list[_rtc.AudioFrame]:
		"""Decode an Ogg/Opus stream chunk from PersonaPlex to LiveKit AudioFrame(s)."""
		if self._reader is None:
			self._init_decoder()
		self._reader.append_bytes(opus_bytes)
		pcm = self._reader.read_pcm()  # 1D float32 array
		if pcm is None or pcm.shape[0] == 0:
			return []
		# Convert float32 [-1, 1] → int16 for LiveKit
		pcm_int16 = (pcm * 32768.0).clip(-32768, 32767).astype(_np.int16)
		return [
			_rtc.AudioFrame(
				data=pcm_int16.tobytes(),
				sample_rate=_LK_SAMPLE_RATE,
				num_channels=_LK_NUM_CHANNELS,
				samples_per_channel=pcm_int16.shape[0],
			)
		]

	# ── Generation event factory ─────────────────────────────────────────────

	def _make_generation_event(
		self, *, user_initiated: bool
	) -> _llm.GenerationCreatedEvent:
		"""Create a GenerationCreatedEvent backed by the live audio/text channels.

		PersonaPlex is fully full-duplex: the model generates audio continuously
		once connected. When a new generation is requested we close the old channels
		first so that any existing consumer unblocks cleanly, then open fresh ones.
		"""
		# Close old channels so previous consumers (if any) can drain and exit.
		if self._audio_ch and not self._audio_ch.closed:
			self._audio_ch.close()
		if self._text_ch and not self._text_ch.closed:
			self._text_ch.close()
		self._audio_ch = _lk_utils.aio.Chan()
		self._text_ch = _lk_utils.aio.Chan()
		audio_ch = self._audio_ch
		text_ch = self._text_ch

		async def _audio_gen():
			async for frame in audio_ch:
				yield frame

		async def _text_gen():
			async for tok in text_ch:
				yield tok

		async def _modalities():
			return ["text", "audio"]

		msg_id = str(uuid.uuid4())
		msg = _llm.MessageGeneration(
			message_id=msg_id,
			text_stream=_text_gen(),
			audio_stream=_audio_gen(),
			modalities=asyncio.ensure_future(_modalities()),
		)

		async def _msg_stream():
			yield msg

		async def _fn_stream():
			return
			yield  # type: ignore  # make it an async generator

		return _llm.GenerationCreatedEvent(
			message_stream=_msg_stream(),
			function_stream=_fn_stream(),
			user_initiated=user_initiated,
			response_id=msg_id,
		)

	# ── WebSocket connection lifecycle ───────────────────────────────────────

	def _build_ws_url(self) -> str:
		m = self._model
		parsed = urlparse(m._ws_url)
		params: dict = {
			"text_prompt": self._instructions or m._text_prompt,
			"voice_prompt": m._voice_prompt,
			"text_temperature": str(m._text_temperature),
			"text_topk": str(m._text_topk),
			"audio_temperature": str(m._audio_temperature),
			"audio_topk": str(m._audio_topk),
			"pad_mult": str(m._pad_mult),
			"repetition_penalty_context": str(m._repetition_penalty_context),
			"repetition_penalty": str(m._repetition_penalty),
		}
		if m._seed != -1:
			params["seed"] = str(m._seed)
		q = urlencode(params)
		return urlunparse(parsed._replace(query=q))

	async def _run(self) -> None:
		"""Main task: connect to PersonaPlex WS, await handshake, run I/O loops."""
		url = _upgrade_ws_scheme(self._build_ws_url())
		logger.info("PersonaPlex session connecting to %s", url)
		try:
			async with websockets.connect(url, max_size=None) as ws:
				# PersonaPlex sends 0x00 once model is warmed up and ready
				while True:
					msg = await ws.recv()
					if isinstance(msg, bytes) and len(msg) >= 1 and msg[0] == 0x00:
						break
				logger.info("PersonaPlex handshake received — full-duplex session starting")

				# Announce the generation so the LiveKit pipeline starts consuming.
				# PersonaPlex is full-duplex: audio flows in both directions from now on.
				evt = self._make_generation_event(user_initiated=False)
				self.emit("generation_created", evt)

				send_t = asyncio.create_task(self._send_loop(ws))
				flush_t = asyncio.create_task(self._flush_loop(ws))
				recv_t = asyncio.create_task(self._recv_loop(ws))
				done, pending = await asyncio.wait(
					[send_t, flush_t, recv_t], return_when=asyncio.FIRST_COMPLETED
				)
				for t in pending:
					t.cancel()
					try:
						await t
					except asyncio.CancelledError:
						pass
		except asyncio.CancelledError:
			raise
		except Exception as exc:
			logger.exception("PersonaPlex session error: %s", exc)
			self.emit(
				"error",
				_llm.RealtimeModelError(
					timestamp=time.monotonic(),
					label="personaplex_session",
					error=exc,
					recoverable=False,
				),
			)
		finally:
			if self._audio_ch and not self._audio_ch.closed:
				self._audio_ch.close()
			if self._text_ch and not self._text_ch.closed:
				self._text_ch.close()

	async def _send_loop(self, ws) -> None:
		"""Drain the output queue and forward Opus packets to PersonaPlex.

		 Uses an asyncio.Event so the coroutine wakes the moment push_audio
		 enqueues a packet, eliminating the previous 50 ms polling timeout.
		 After waking, all queued packets are drained in a single pass to avoid
		 head-of-line blocking when multiple frames have accumulated.
		"""
		sent = 0
		while not self._closed:
			# Wait until at least one packet is available (or sentinel)
			await self._out_event.wait()
			self._out_event.clear()
			# Drain everything that is currently queued
			while True:
				try:
					opus = self._out_queue.get_nowait()
				except asyncio.QueueEmpty:
					break
				if opus is None:  # sentinel from aclose()
					logger.info("PersonaPlex _send_loop exited (sent %d packets)", sent)
					return
				await ws.send(b"\x01" + opus)
				sent += 1
				if sent == 1:
					logger.info("PersonaPlex: first audio packet sent to server (%d bytes)", len(opus))
				elif sent % 500 == 0:
					logger.debug("PersonaPlex: %d audio packets sent to server", sent)
		logger.info("PersonaPlex _send_loop exited (sent %d packets)", sent)

	async def _flush_loop(self, ws) -> None:
		"""Periodically flush any bytes buffered inside sphn.OpusStreamWriter.

		 sphn produces Ogg/Opus pages which may span several 20 ms Opus frames
		 before a page boundary is reached.  Without this loop those bytes sit
		 inside the writer until the next append_pcm() call — potentially adding
		 60-80 ms of extra latency on top of the model's own 80 ms frame rate.
		 Polling at 10 ms (half an Opus frame) keeps worst-case buffering ≤ 10 ms.
		"""
		while not self._closed:
			await asyncio.sleep(0.01)  # 10 ms — well below one Opus frame (20 ms)
			if self._writer is None:
				continue
			try:
				chunk = self._writer.read_bytes()
			except Exception:
				continue
			if chunk:
				try:
					await ws.send(b"\x01" + chunk)
				except Exception:
					break

	async def _recv_loop(self, ws) -> None:
		"""Receive Opus audio and text tokens from PersonaPlex and forward to LiveKit."""
		recvd = 0
		async for msg in ws:
			if not isinstance(msg, bytes) or len(msg) < 2:
				continue
			kind, payload = msg[0], msg[1:]
			if kind == 0x01:  # model audio output
				recvd += 1
				if recvd == 1:
					logger.info("PersonaPlex: first audio packet received from server (%d bytes)", len(payload))
				for lk_frame in self._decode_opus(payload):
					if self._audio_ch and not self._audio_ch.closed:
						self._audio_ch.send_nowait(lk_frame)
			elif kind == 0x02:  # model text token
				text = payload.decode("utf-8", errors="ignore")
				if self._text_ch and not self._text_ch.closed:
					self._text_ch.send_nowait(text)
		logger.info("PersonaPlex _recv_loop exited (received %d audio packets)", recvd)


# ── PersonaPlex Realtime Model ────────────────────────────────────────────────

class PersonaPlexRealtimeModel(_llm.RealtimeModel):
	"""LiveKit RealtimeModel that routes audio through a PersonaPlex endpoint.

	PersonaPlex (https://github.com/NVIDIA/personaplex) is a full-duplex,
	speech-to-speech model served via a Moshi-compatible WebSocket server.
	This class implements the LiveKit RealtimeModel interface so that
	PersonaPlex acts as the sole end-to-end audio brain — no separate STT,
	LLM, or TTS stages are needed.

	Key env vars (all optional, sensible defaults provided):
	  PERSONAPLEX_WS_URL          WebSocket endpoint of the running server
	                              (default: ws://127.0.0.1:8998/api/chat)
	  PERSONAPLEX_TEXT_PROMPT     Persona / system prompt
	  PERSONAPLEX_VOICE_PROMPT    Voice embedding file (e.g. NATF0.pt)
	  PERSONAPLEX_SEED            RNG seed (-1 = random)
	"""

	def __init__(
		self,
		*,
		ws_url: str | None = None,
		text_prompt: str = _LK_DEFAULT_TEXT_PROMPT,
		voice_prompt: str = _LK_DEFAULT_VOICE_PROMPT,
		text_temperature: float = 0.7,
		text_topk: int = 25,
		audio_temperature: float = 0.8,
		audio_topk: int = 250,
		pad_mult: int = 0,
		repetition_penalty_context: int = 64,
		repetition_penalty: float = 1.0,
		seed: int = -1,
	) -> None:
		super().__init__(
			capabilities=_llm.RealtimeCapabilities(
				message_truncation=False,
				# PersonaPlex handles its own full-duplex turn detection
				turn_detection=False,
				user_transcription=False,
				auto_tool_reply_generation=False,
				audio_output=True,
				manual_function_calls=False,
			)
		)
		self._ws_url = ws_url or _LK_WS_URL
		self._text_prompt = text_prompt
		self._voice_prompt = voice_prompt
		self._text_temperature = text_temperature
		self._text_topk = text_topk
		self._audio_temperature = audio_temperature
		self._audio_topk = audio_topk
		self._pad_mult = pad_mult
		self._repetition_penalty_context = repetition_penalty_context
		self._repetition_penalty = repetition_penalty
		self._seed = seed

	@property
	def model(self) -> str:
		return "personaplex-7b-v1"

	@property
	def provider(self) -> str:
		return urlparse(self._ws_url).netloc

	def session(self) -> PersonaPlexRealtimeSession:
		return PersonaPlexRealtimeSession(self)

	async def aclose(self) -> None:
		pass


# ── LiveKit Agent & Server ────────────────────────────────────────────────────

_livekit_server = _AgentServer(port=0)  # port=0 → OS picks a free ephemeral port


@_livekit_server.rtc_session()
async def livekit_entrypoint(ctx: _JobContext) -> None:
	await ctx.connect()
	session = _AgentSession(
		llm=PersonaPlexRealtimeModel(
			text_prompt=os.getenv("PERSONAPLEX_TEXT_PROMPT", _LK_DEFAULT_TEXT_PROMPT),
			voice_prompt=os.getenv("PERSONAPLEX_VOICE_PROMPT", _LK_DEFAULT_VOICE_PROMPT),
			seed=int(os.getenv("PERSONAPLEX_SEED", "-1")),
		),
		# PersonaPlex is a continuous full-duplex model: user audio must flow to
		# the server AT ALL TIMES, not just during detected speech windows.
		# Disabling the discard-if-uninterruptible flag ensures push_audio is
		# called unconditionally so the server hears the user continuously.
		discard_audio_if_uninterruptible=False,
		# Remove the default 500ms end-of-speech silence gate.  PersonaPlex
		# generates audio the instant it has something to say; there is no
		# need to wait for a VAD-confirmed turn boundary before re-subscribing
		# to the model audio channel.
		min_endpointing_delay=0.0,
		max_endpointing_delay=0.5,
		# Allow interruptions with the shortest detectable speech burst so the
		# pipeline stops gating audio early rather than waiting for a full word.
		min_interruption_duration=0.05,
	)
	await session.start(
		agent=_Agent(
			instructions=os.getenv("PERSONAPLEX_TEXT_PROMPT", _LK_DEFAULT_TEXT_PROMPT)
		),
		room=ctx.room,
	)


def build_livekit_server():
	return _livekit_server, _lk_cli


def main() -> None:
	parser = argparse.ArgumentParser(description="PersonaPlex bridge and LiveKit integration")
	sub = parser.add_subparsers(dest="mode", required=True)

	http_parser = sub.add_parser("serve-http")
	http_parser.add_argument("--host", default="0.0.0.0")
	http_parser.add_argument("--port", default=8000, type=int)

	sub.add_parser("livekit")

	args, passthrough = parser.parse_known_args()

	if args.mode == "serve-http":
		uvicorn.run(create_app(), host=args.host, port=args.port)
		return

	if args.mode == "livekit":
		server, cli = build_livekit_server()
		# Env vars may not be set at module-import time (when AgentServer is
		# constructed at module level), so push them in now at actual run time.
		# Only pass a kwarg if the env var is present — update_options treats
		# None as an explicit value (distinct from NOT_GIVEN).
		_lk_overrides: dict = {}
		if os.environ.get("LIVEKIT_URL"):
			_lk_overrides["ws_url"] = os.environ["LIVEKIT_URL"]
		if os.environ.get("LIVEKIT_API_KEY"):
			_lk_overrides["api_key"] = os.environ["LIVEKIT_API_KEY"]
		if os.environ.get("LIVEKIT_API_SECRET"):
			_lk_overrides["api_secret"] = os.environ["LIVEKIT_API_SECRET"]
		if _lk_overrides:
			server.update_options(**_lk_overrides)
		sys.argv = [sys.argv[0], *passthrough]
		cli.run_app(server)


if __name__ == "__main__":
	main()
