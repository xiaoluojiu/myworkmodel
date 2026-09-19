"""LCCC-base 中文对话数据预处理脚本。

支持 Hugging Face 流式读取和本地 JSON/JSONL，避免一次性把数百万条
对话加载进内存；使用 re 完成文本清洗，pandas 负责最终统计整理。
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import pandas as pd

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

HTML_RE = re.compile(r"<[^>]+>")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SPACE_RE = re.compile(r"\s+")
REPEATED_PUNCT_RE = re.compile(r"([！？。,.，；;：:])\1{3,}")


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    text = HTML_RE.sub("", text)
    text = CONTROL_RE.sub("", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = REPEATED_PUNCT_RE.sub(r"\1\1\1", text)
    return SPACE_RE.sub(" ", text).strip()


def load_sensitive_words(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines()
            if x.strip() and not x.strip().startswith("#")]


def contains_sensitive(text: str, sensitive_words: Iterable[str]) -> bool:
    return any(word and word in text for word in sensitive_words)


def extract_utterances(record: Any) -> List[str]:
    if isinstance(record, list):
        return [str(x) for x in record]
    if not isinstance(record, dict):
        return []

    value = record.get("dialog")
    if value is None:
        for key in ("conversation", "conversations", "utterances"):
            if record.get(key) is not None:
                value = record[key]
                break
    if value is None and isinstance(record.get("messages"), list):
        value = [m.get("content", "") for m in record["messages"] if isinstance(m, dict)]
    if not isinstance(value, list):
        return []

    result = []
    for item in value:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            content = item.get("text", item.get("content", item.get("utterance", "")))
            if content:
                result.append(str(content))
    return result


def record_to_messages(record: Any, sensitive_words: List[str]) -> Optional[Dict[str, Any]]:
    utterances = [clean_text(x) for x in extract_utterances(record)]
    utterances = [x for x in utterances if x]
    if len(utterances) < 2 or contains_sensitive("\n".join(utterances), sensitive_words):
        return None
    return {"messages": [
        {"role": "user" if i % 2 == 0 else "assistant", "content": text}
        for i, text in enumerate(utterances)
    ]}


def iter_local_records(path: Path) -> Iterator[Any]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("data", "train", "conversations", "dialogues"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("本地 JSON 文件必须最终解析为 list。")
    yield from data


def iter_hf_records(args: argparse.Namespace) -> Iterator[Any]:
    if load_dataset is None:
        raise RuntimeError("未安装 datasets，请执行 pip install -r requirements.txt")
    dataset = load_dataset(args.dataset_name, "base", split=args.split, streaming=True)
    yield from dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="LCCC-base 数据清洗与对话格式转换")
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="silver/lccc")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--sensitive_words", type=str, default="sensitive_words.txt")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer_size", type=int, default=500)
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.valid_ratio <= 0 or args.train_ratio + args.valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio 必须小于 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "train.jsonl",
        "valid": output_dir / "valid.jsonl",
        "test": output_dir / "test.jsonl",
    }
    for path in paths.values():
        path.write_text("", encoding="utf-8")

    sensitive_words = load_sensitive_words(Path(args.sensitive_words))
    records = iter_local_records(Path(args.input_file)) if args.input_file else iter_hf_records(args)
    rng = random.Random(args.seed)
    buffers = {name: [] for name in paths}

    stats = {
        "raw_records": 0,
        "kept_records": 0,
        "removed_empty_or_short": 0,
        "removed_sensitive": 0,
        "removed_invalid": 0,
        "sensitive_word_count": len(sensitive_words),
        "train_records": 0,
        "valid_records": 0,
        "test_records": 0,
    }

    handles = {name: path.open("a", encoding="utf-8") for name, path in paths.items()}
    try:
        for record in records:
            if args.max_samples > 0 and stats["raw_records"] >= args.max_samples:
                break
            stats["raw_records"] += 1

            utterances = [clean_text(x) for x in extract_utterances(record)]
            utterances = [x for x in utterances if x]
            if len(utterances) < 2:
                stats["removed_empty_or_short"] += 1
                continue
            if contains_sensitive("\n".join(utterances), sensitive_words):
                stats["removed_sensitive"] += 1
                continue

            converted = record_to_messages(utterances, sensitive_words)
            if converted is None:
                stats["removed_invalid"] += 1
                continue

            r = rng.random()
            split_name = "train" if r < args.train_ratio else "valid" if r < args.train_ratio + args.valid_ratio else "test"
            buffers[split_name].append(converted)
            stats[f"{split_name}_records"] += 1
            stats["kept_records"] += 1

            if len(buffers[split_name]) >= args.buffer_size:
                for item in buffers[split_name]:
                    handles[split_name].write(json.dumps(item, ensure_ascii=False) + "\n")
                buffers[split_name].clear()
    finally:
        for name, items in buffers.items():
            for item in items:
                handles[name].write(json.dumps(item, ensure_ascii=False) + "\n")
            handles[name].close()

    stats["removed_total"] = stats["raw_records"] - stats["kept_records"]
    stats["retention_rate"] = round(stats["kept_records"] / max(stats["raw_records"], 1), 6)

    summary_df = pd.DataFrame([stats])
    stats = summary_df.iloc[0].to_dict()
    integer_keys = [
        "raw_records", "kept_records", "removed_empty_or_short", "removed_sensitive",
        "removed_invalid", "sensitive_word_count", "train_records", "valid_records",
        "test_records", "removed_total"
    ]
    for key in integer_keys:
        stats[key] = int(stats[key])
    stats["retention_rate"] = float(stats["retention_rate"])

    (output_dir / "statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n处理完成，输出目录：{output_dir.resolve()}")


if __name__ == "__main__":
    main()
