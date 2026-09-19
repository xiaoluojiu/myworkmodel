# 数据目录说明

完整 LCCC-base 数据集规模较大，本仓库不直接提交原始数据和完整处理结果。

## 数据来源

推荐使用 Hugging Face 的 `silver/lccc` 数据集：

```python
from datasets import load_dataset

dataset = load_dataset("silver/lccc", "base", split="train")
```

LCCC-base 的字段为 `dialog`，每条记录是一个由多条 utterance 组成的列表。

## 本地数据

如果课程环境无法联网，可下载 LCCC-base 后放置到：

```text
data/raw/LCCC-base.json
```

然后运行：

```bash
python src/prepare_lccc.py --input_file data/raw/LCCC-base.json
```

## 输出

程序会生成：

```text
data/processed/train.jsonl
data/processed/valid.jsonl
data/processed/test.jsonl
data/processed/statistics.json
```

`.gitignore` 中默认忽略大规模原始数据和处理结果，避免把数据集本身提交到代码仓库。
