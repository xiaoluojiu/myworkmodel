"""LCCC-base 对话数据预处理脚本。

功能：
1. 从 Hugging Face datasets 加载 LCCC-base，或读取本地 JSON。
2. 清洗 HTML 标签、特殊字符、异常空白。
3. 使用可配置敏感词表过滤对话。
4. 将多轮 utterances 转换为 user/assistant 交替角色格式。
5. 划分 train/valid/test 并保存为 JSONL。
6. 保存处理统计信息。
"""

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
    """清理单条文本。"""
    if text is None:
        return ""
    text = str(text).strip()
    text = HTML_RE.sub("", text)
    text = CONTROL_RE.sub("", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = REPEATED_PUNCT_RE.sub(r"\1\1\1", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def load_sensitive_words(path: Path) -> List[str]:
    if not path.exists():
        return []
    words = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.append(line)
    return words


def contains_sensitive(text: str, sensitive_words: Iterable[str]) -> bool:
    return any(word and word in text for word in sensitive_words)


def extract_utterances(record: Dict[str, Any]) -> List[str]:
    """兼容 LCCC 常见的 conversations / utterances 字段。"""
    value = record.get("conversation")
    if value is None:
        value = record.get("conversations")
    if value is None:
        value = record.get("utterances")
    if value is None:
        # 某些本地格式直接使用 list 作为记录本身，这里交由调用方处理。
        return []
    if isinstance(value, dict):
        value = list(value.values())
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
    if isinstance(record, list):
        raw_utterances = record
    elif isinstance(record, dict):
        raw_utterances = extract_utterances(record)
        if not raw_utterances and isinstance(record.get("messages"), list):
            raw_utterances = [m.get("content", "") for m in record["messages"] if isinstance(m, dict)]
    else:
        return None

    utterances = [clean_text(x) for x in raw_utterances]
    utterances = [x for x in utterances if x]

    # 对话至少需要一问一答；奇数轮可以保留，最后一轮作为 assistant 回复。
    if len(utterances) < 2:
        return None

    joined = "\n".join(utterances)
    if contains_sensitive(joined, sensitive_words):
        return None

    messages = []
    for index, utterance in enumerate(utterances):
        role = "user" if index % 2 == 0 else "assistant"
        messages.append({"role": role, "content": utterance})

    return {"messages": messages}


def load_records(args: argparse.Namespace) -> List[Any]:
    if args.input_file:
        path = Path(args.input_file)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # 兼容 {"data": [...]} 等常见包装格式。
            for key in ("data", "train", "conversations", "dialogues"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError("本地 JSON 文件必须最终解析为 list。")
        return data

    if load_dataset is None:
        raise RuntimeError("未安装 datasets，请执行 pip install -r requirements.txt")

    dataset = load_dataset("lccc", "base", split=args.split)
    return [dataset[i] for i in range(len(dataset))]


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_data(records: List[Dict[str, Any]], train_ratio: float, valid_ratio: float, seed: int):
    random.Random(seed).shuffle(records)
    n = len(records)
    train_end = int(n * train_ratio)
    valid_end = train_end + int(n * valid_ratio)
    return records[:train_end], records[train_end:valid_end], records[valid_end:]


def main() -> None:
    parser = argparse.ArgumentParser(description="LCCC-base 数据清洗与对话格式转换")
    parser.add_argument("--input_file", type=str, default=None, help="本地 JSON；不提供时从 HF 加载 LCCC-base")
    parser.add_argument("--split", type=str, default="train", help="Hugging Face 数据集 split")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--sensitive_words", type=str, default="sensitive_words.txt")
    parser.add_argument("--max_samples", type=int, default=0, help=">0 时限制处理数量，便于课堂演示")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.train_ratio < 1 or not 0 < args.valid_ratio < 1 or args.train_ratio + args.valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio 必须小于 1")

    output_dir = Path(args.output_dir)
    sensitive_words = load_sensitive_words(Path(args.sensitive_words))
    raw_records = load_records(args)
    if args.max_samples > 0:
        raw_records = raw_records[:args.max_samples]

    cleaned = []
    stats = {
        "raw_records": len(raw_records),
        "kept_records": 0,
        "removed_empty_or_short": 0,
        "removed_sensitive": 0,
        "removed_invalid": 0,
        "sensitive_word_count": len(sensitive_words),
    }

    for record in raw_records:
        before_count = len(cleaned)
        if isinstance(record, list):
            raw_texts = record
        elif isinstance(record, dict):
            raw_texts = extract_utterances(record)
            if not raw_texts and isinstance(record.get("messages"), list):
                raw_texts = [m.get("content", "") for m in record["messages"] if isinstance(m, dict)]
        else:
            raw_texts = []

        cleaned_texts = [clean_text(x) for x in raw_texts]
        cleaned_texts = [x for x in cleaned_texts if x]
        if len(cleaned_texts) < 2:
            stats["removed_empty_or_short"] += 1
            continue
        if contains_sensitive("\n".join(cleaned_texts), sensitive_words):
            stats["removed_sensitive"] += 1
            continue

        converted = record_to_messages(record, sensitive_words)
        if converted is None:
            stats["removed_invalid"] += 1
            continue
        cleaned.append(converted)
        if len(cleaned) == before_count:
            stats["removed_invalid"] += 1

    stats["kept_records"] = len(cleaned)
    stats["removed_total"] = stats["raw_records"] - stats["kept_records"]
    stats["retention_rate"] = round(stats["kept_records"] / max(stats["raw_records"], 1), 6)

    train, valid, test = split_data(cleaned, args.train_ratio, args.valid_ratio, args.seed)
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "valid.jsonl", valid)
    write_jsonl(output_dir / "test.jsonl", test)

    stats["train_records"] = len(train)
    stats["valid_records"] = len(valid)
    stats["test_records"] = len(test)
    (output_dir / "statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n处理完成，输出目录：{output_dir.resolve()}")


if __name__ == "__main__":
    main()
