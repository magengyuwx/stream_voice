import os
import wave
from datetime import datetime

import dashscope
import sounddevice as sd
from dashscope.audio.tts_v2 import AudioFormat, ResultCallback, SpeechSynthesizer
from dotenv import load_dotenv


def ts() -> str:
	return datetime.now().strftime("[%Y-%m-%d %H:%M:%S.%f]")


class RealtimeWavCallback(ResultCallback):
	def __init__(self, player: sd.RawOutputStream, wav_file: wave.Wave_write):
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
		self.player.write(data)
		self.wav_file.writeframes(data)
		print(f"{ts()} chunk={self.chunk_index}, size={chunk_len}, total={self.total_bytes}")


def main() -> None:
	load_dotenv()

	api_key = os.getenv("QWEN_TTS_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
	if not api_key:
		raise RuntimeError("Missing API key: set QWEN_TTS_API_KEY or DASHSCOPE_API_KEY in .env")
	dashscope.api_key = api_key

	model = "cosyvoice-v3-flash"
	voice = "longanyang"
	sample_rate = 24000
	output_path = "output_stream_realtime.wav"
	text_segments = [
		"现在演示边生成边播放的流式语音合成。",
		"你会在音频分片到达时立即听到声音，",
		"这比等整段音频合成结束后再播放更实时。",
	]

	print(f"{ts()} 开始流式合成并实时播放")
	print(f"{ts()} 输出设备: {sd.default.device}")

	with sd.RawOutputStream(samplerate=sample_rate, channels=1, dtype="int16") as player:
		with wave.open(output_path, "wb") as wav_file:
			wav_file.setnchannels(1)
			wav_file.setsampwidth(2)
			wav_file.setframerate(sample_rate)
			callback = RealtimeWavCallback(player, wav_file)

			synthesizer = SpeechSynthesizer(
				model=model,
				voice=voice,
				format=AudioFormat.PCM_24000HZ_MONO_16BIT,
				callback=callback,
			)

			for segment in text_segments:
				if not segment:
					continue
				synthesizer.streaming_call(segment)

			synthesizer.streaming_complete()

	print(f"{ts()} 播放结束，已写入 {output_path}，总字节={callback.total_bytes}")
	print(
		f"[Metric] requestId={synthesizer.get_last_request_id()}, first_package_delay_ms={synthesizer.get_first_package_delay()}"
	)


if __name__ == "__main__":
	main()
