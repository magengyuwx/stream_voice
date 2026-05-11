import os
import uuid
import wave
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import dashscope
from dashscope.audio.tts_v2 import AudioFormat, ResultCallback, SpeechSynthesizer
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    import sounddevice as sd
except ImportError:
    sd = None


def ts() -> str:
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S.%f]")


class SynthesizeRequest(BaseModel):
    text_segments: List[str] = Field(
        ...,
        description="Text chunks to stream into TTS. Useful for LLM token/segment output.",
    )
    model: str = "cosyvoice-v3-flash"
    voice: str = "longanyang"
    sample_rate: int = 24000
    output_filename: Optional[str] = None
    output_dir: str = "outputs"
    play_realtime: bool = False


class SynthesizeResponse(BaseModel):
    request_id: str
    first_package_delay_ms: float
    wav_path: str
    total_bytes: int
    chunk_count: int


class RealtimeWavCallback(ResultCallback):
    def __init__(self, player, wav_file: wave.Wave_write):
        self.player = player
        self.wav_file = wav_file
        self.chunk_index = 0
        self.total_bytes = 0

    def on_open(self) -> None:
        print(f"{ts()} TTS连接已建立")

    def on_complete(self) -> None:
        print(f"{ts()} TTS合成已完成")

    def on_error(self, message) -> None:
        print(f"{ts()} TTS出现异常: {message}")

    def on_close(self) -> None:
        print(f"{ts()} TTS连接已关闭")

    def on_event(self, message: str) -> None:
        pass

    def on_data(self, data: bytes) -> None:
        self.chunk_index += 1
        chunk_len = len(data)
        self.total_bytes += chunk_len
        if self.player is not None:
            self.player.write(data)
        self.wav_file.writeframes(data)
        print(f"{ts()} chunk={self.chunk_index}, size={chunk_len}, total={self.total_bytes}")


def _pcm_format_for_sample_rate(sample_rate: int) -> AudioFormat:
    format_map = {
        8000: AudioFormat.PCM_8000HZ_MONO_16BIT,
        16000: AudioFormat.PCM_16000HZ_MONO_16BIT,
        22050: AudioFormat.PCM_22050HZ_MONO_16BIT,
        24000: AudioFormat.PCM_24000HZ_MONO_16BIT,
        44100: AudioFormat.PCM_44100HZ_MONO_16BIT,
        48000: AudioFormat.PCM_48000HZ_MONO_16BIT,
    }
    if sample_rate not in format_map:
        supported = ", ".join(str(rate) for rate in sorted(format_map))
        raise HTTPException(status_code=400, detail=f"Unsupported sample_rate: {sample_rate}. Supported: {supported}")
    return format_map[sample_rate]


def _build_output_path(output_dir: str, output_filename: Optional[str]) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if output_filename:
        filename = output_filename
        if not filename.lower().endswith(".wav"):
            filename += ".wav"
    else:
        filename = f"novel_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.wav"
    return out_dir / filename


def _configure_api_key() -> None:
    load_dotenv()
    api_key = os.getenv("QWEN_TTS_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Missing API key: set QWEN_TTS_API_KEY or DASHSCOPE_API_KEY in .env",
        )
    dashscope.api_key = api_key


def synthesize_segments(request: SynthesizeRequest) -> SynthesizeResponse:
    _configure_api_key()

    cleaned_segments = [seg.strip() for seg in request.text_segments if seg and seg.strip()]
    if not cleaned_segments:
        raise HTTPException(status_code=400, detail="text_segments must contain at least one non-empty segment")

    if request.play_realtime and sd is None:
        raise HTTPException(status_code=400, detail="sounddevice is not installed, cannot play realtime audio")

    wav_path = _build_output_path(request.output_dir, request.output_filename)

    player_cm = (
        sd.RawOutputStream(samplerate=request.sample_rate, channels=1, dtype="int16")
        if request.play_realtime
        else nullcontext(None)
    )

    print(f"{ts()} 开始流式合成: segments={len(cleaned_segments)}, play={request.play_realtime}")
    if request.play_realtime and sd is not None:
        print(f"{ts()} 输出设备: {sd.default.device}")
    with player_cm as player:
        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(request.sample_rate)

            callback = RealtimeWavCallback(player, wav_file)
            synthesizer = SpeechSynthesizer(
                model=request.model,
                voice=request.voice,
                format=_pcm_format_for_sample_rate(request.sample_rate),
                callback=callback,
            )

            for segment in cleaned_segments:
                synthesizer.streaming_call(segment)

            synthesizer.streaming_complete()

    print(f"{ts()} 播放结束，已写入 {wav_path}，总字节={callback.total_bytes}")
    return SynthesizeResponse(
        request_id=synthesizer.get_last_request_id(),
        first_package_delay_ms=synthesizer.get_first_package_delay(),
        wav_path=str(wav_path.resolve()),
        total_bytes=callback.total_bytes,
        chunk_count=callback.chunk_index,
    )


def synthesize_from_segments(
    text_segments: List[str],
    model: str = "cosyvoice-v3-flash",
    voice: str = "longanyang",
    sample_rate: int = 24000,
    output_filename: Optional[str] = None,
    output_dir: str = "outputs",
    play_realtime: bool = False,
) -> SynthesizeResponse:
    """Non-HTTP entrypoint for local script calls."""
    request = SynthesizeRequest(
        text_segments=text_segments,
        model=model,
        voice=voice,
        sample_rate=sample_rate,
        output_filename=output_filename,
        output_dir=output_dir,
        play_realtime=play_realtime,
    )
    return synthesize_segments(request)


app = FastAPI(title="Qwen TTS Streaming API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/tts/synthesize", response_model=SynthesizeResponse)
def synthesize(request: SynthesizeRequest) -> SynthesizeResponse:
    return synthesize_segments(request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("tts_api_server:app", host="127.0.0.1", port=8000, reload=False)
