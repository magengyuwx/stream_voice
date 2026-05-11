import os
from datetime import datetime

import dashscope
from dashscope.audio.http_tts.http_speech_synthesizer import HttpSpeechSynthesizer
from dotenv import load_dotenv

def get_timestamp() -> str:
    now = datetime.now()
    formatted_timestamp = now.strftime("[%Y-%m-%d %H:%M:%S.%f]")
    return formatted_timestamp


def main() -> None:
    load_dotenv()

    # 优先读取你前面脚本使用的变量名，同时兼容下载示例里的变量名
    api_key = os.environ.get("QWEN_TTS_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key: set QWEN_TTS_API_KEY or DASHSCOPE_API_KEY in .env")
    dashscope.api_key = api_key

    # 默认北京地域。若使用新加坡地域可切换为：https://dashscope-intl.aliyuncs.com/api/v1
    # dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

    model = "cosyvoice-v3-flash"
    voice = "longanyang"
    output_path = "output_stream.mp3"
    text = (
        "流式文本语音合成可以边生成边返回音频分片，"
        "显著降低首包等待时间，"
        "适合和大语言模型流式输出联动。"
    )


    stream_results = HttpSpeechSynthesizer.call(
        model=model,
        text=text,
        voice=voice,
        audio_format="mp3",
        sample_rate=24000,
        stream=True,
    )

    chunk_index = 0
    total_bytes = 0
    final_audio_url = None

    print(f"{get_timestamp()} 开始流式接收音频")
    with open(output_path, "wb") as f:
        for part in stream_results:
            is_final_event = bool(part.audio_url)

            # 结束事件可能包含聚合后的整段音频，避免与前面分片重复写入。
            if part.audio_data and not is_final_event:
                chunk_index += 1
                chunk_len = len(part.audio_data)
                total_bytes += chunk_len
                f.write(part.audio_data)
                print(
                    f"{get_timestamp()} chunk={chunk_index}, size={chunk_len}, total={total_bytes}",
                )

            # 结束事件里通常带有最终音频 url，可用于回溯或二次下载
            if is_final_event:
                final_audio_url = part.audio_url

    print(f"{get_timestamp()} 流式合成完成，已写入 {output_path}，总字节={total_bytes}")
    if final_audio_url:
        print(f"final_audio_url={final_audio_url}")


if __name__ == "__main__":
    main()