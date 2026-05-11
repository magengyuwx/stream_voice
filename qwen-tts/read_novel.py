import argparse
import json
import re
from pathlib import Path
from typing import List

from tts_api_server import synthesize_from_segments


def read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Failed to decode file with supported encodings: {path}")


def split_for_tts(text: str, max_chars: int) -> List[str]:
    text = re.sub(r"\r\n?", "\n", text)
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    segments: List[str] = []
    sentence_splitter = re.compile(r"([^。！？!?；;]+[。！？!?；;]?)")

    for para in paragraphs:
        sentence_parts = [s for s in sentence_splitter.findall(para) if s.strip()]
        if not sentence_parts:
            sentence_parts = [para]

        current = ""
        for sentence in sentence_parts:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > max_chars:
                if current:
                    segments.append(current)
                    current = ""
                for i in range(0, len(sentence), max_chars):
                    segments.append(sentence[i : i + max_chars])
                continue

            if len(current) + len(sentence) <= max_chars:
                current += sentence
            else:
                if current:
                    segments.append(current)
                current = sentence

        if current:
            segments.append(current)

    return segments


def split_batches_by_total_chars(segments: List[str], max_total_chars: int) -> List[List[str]]:
    if max_total_chars <= 0:
        raise ValueError("max_total_chars must be > 0")

    batches: List[List[str]] = []
    current_batch: List[str] = []
    current_total = 0

    for segment in segments:
        seg_len = len(segment)
        # A single segment can already be larger than the per-request limit.
        if seg_len > max_total_chars:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_total = 0
            batches.append([segment])
            continue

        if current_total + seg_len > max_total_chars:
            batches.append(current_batch)
            current_batch = [segment]
            current_total = seg_len
        else:
            current_batch.append(segment)
            current_total += seg_len

    if current_batch:
        batches.append(current_batch)

    return batches


def build_part_filename(base_filename: str, part_index: int, total_parts: int) -> str:
    if total_parts <= 1:
        return base_filename
    base_path = Path(base_filename)
    stem = base_path.stem
    suffix = base_path.suffix or ".wav"
    return f"{stem}_part{part_index:03d}{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read a novel via direct TTS synthesizer call")
    parser.add_argument("--novel-path", required=True, help="Path to novel text file")
    parser.add_argument("--model", default="cosyvoice-v3-flash")
    parser.add_argument("--voice", default="longqiang_v3")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--max-segment-chars", type=int, default=120)
    parser.add_argument("--max-request-chars", type=int, default=190000)
    parser.add_argument("--max-input-chars", type=int, default=0)
    parser.add_argument("--output-filename", default="novel_output.wav")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--play-realtime", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    novel_path = Path(args.novel_path)
    if not novel_path.exists():
        raise FileNotFoundError(f"Novel file not found: {novel_path}")

    text = read_text_file(novel_path)
    if args.max_input_chars > 0:
        text = text[: args.max_input_chars]

    segments = split_for_tts(text, max_chars=args.max_segment_chars)
    if not segments:
        raise RuntimeError("No valid text segment found in input file")

    batches = split_batches_by_total_chars(segments, max_total_chars=args.max_request_chars)

    print(f"Prepared segments: {len(segments)}")
    print(f"Prepared batches: {len(batches)} (max_request_chars={args.max_request_chars})")

    results = []
    total_parts = len(batches)
    for idx, batch in enumerate(batches, start=1):
        batch_chars = sum(len(seg) for seg in batch)
        part_filename = build_part_filename(args.output_filename, idx, total_parts)
        print(
            f"Start batch {idx}/{total_parts}: segments={len(batch)}, chars={batch_chars}, output={part_filename}"
        )
        result = synthesize_from_segments(
            text_segments=batch,
            model=args.model,
            voice=args.voice,
            sample_rate=args.sample_rate,
            output_filename=part_filename,
            output_dir=args.output_dir,
            play_realtime=args.play_realtime,
        )
        results.append(result.model_dump())

    print("Synthesis done.")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
