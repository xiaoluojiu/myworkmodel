"""LCCC-base 中文对话数据预处理脚本。"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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
    """兼容 LCCC 的 dialog 字段及常见本地格式。"""
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
    if len(utterances) < 2:
        return None
    if contains_sensitive("\n".join(utterances), sensitive_words):
        return None

    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": text}
        for i, text in enumerate(utterances)
    ]
    return {"messages": messages}


def load_records(args: argparse.Namespace) -> List[Any]:
    if args.input_file:
        path = Path(args.input_file)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in ("data", "train", "conversations", "dialogues"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError("本地 JSON 文件必须最终解析为 list。")
        return data

    if load_dataset is None:
        raise RuntimeError("未安装 datasets，请执行 pip install -r requirements.txt")

    # silver/lccc 是 Hugging Face 上可直接读取的 LCCC 镜像，字段为 dialog。
    dataset = load_dataset(args.dataset_name, "base", split=args.split)
    return [dataset[i] for i in range(len(dataset))]


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_data(records: List[Dict[str, Any]], train_ratio: float, valid_ratio: float, seed: int):
    records = list(records)
    random.Random(seed).shuffle(records)
    n = len(records)
    train_end = int(n * train_ratio)
    valid_end = train_end + int(n * valid_ratio)
    return records[:train_end], records[train_end:valid_end], records[valid_end:]


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
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.valid_ratio <= 0 or args.train_ratio + args.valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio 必须小于 1")

    output_dir = Path(args.output_dir)
    sensitive_words = load_sensitive_words(Path(args.sensitive_words))
    raw_records = load_records(args)
    if args.max_samples > 0:
        raw_records = raw_records[:args.max_samples]

    cleaned: List[Dict[str, Any]] = []
    stats = {
        "raw_records": len(raw_records),
        "kept_records": 0,
        "removed_empty_or_short": 0,
        "removed_sensitive": 0,
        "removed_invalid": 0,
        "sensitive_word_count": len(sensitive_words),
    }

    for record in raw_records:
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
        cleaned.append(converted)

    stats["kept_records"] = len(cleaned)
    stats["removed_total"] = stats["raw_records"] - stats["kept_records"]
    stats["retention_rate"] = round(stats["kept_records"] / max(stats["raw_records"], 1), 6)

    train, valid, test = split_data(cleaned, args.train_ratio, args.valid_ratio, args.seed)
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "valid.jsonl", valid)
    write_jsonl(output_dir / "test.jsonl", test)

    stats.update({"train_records": len(train), "valid_records": len(valid), "test_records": len(test)})
    (output_dir / "statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n处理完成，输出目录：{output_dir.resolve()}")


if __name__ == "__main__":
    main()
