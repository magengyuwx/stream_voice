import requests
import os
from dotenv import load_dotenv
import dashscope
from dashscope.audio.http_tts.http_speech_synthesizer import HttpSpeechSynthesizer

# 加载 .env 文件中的环境变量
load_dotenv()

dashscope.api_key = os.getenv("QWEN_TTS_API_KEY")

# HTTP 接口默认走 dashscope.base_http_api_url（北京地域）
# 若使用新加坡地域，可设置为：https://dashscope-intl.aliyuncs.com/api/v1

# 模型
model = "cosyvoice-v3-flash"
# 音色
voice = "longanyang"

result = HttpSpeechSynthesizer.call(
	model=model,
	text="今天天气怎么样？",
	voice=voice,
	audio_format="mp3",
	sample_rate=24000,
	stream=False,
)

if not result.audio_url:
	raise RuntimeError("TTS succeeded but no audio_url was returned")

audio_resp = requests.get(result.audio_url, timeout=30)
audio_resp.raise_for_status()

with open("output.mp3", "wb") as f:
	f.write(audio_resp.content)

print("Synthesis completed: output.mp3")