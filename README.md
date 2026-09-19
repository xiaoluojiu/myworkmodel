# 作业一：LCCC-base 中文对话数据预处理

本项目以清华大学 CoAI 团队公开的 **LCCC-base（Large-scale Cleaned Chinese Conversation）** 为例，完成从原始对话数据到适用于对话模型微调的数据集构建流程。

## 作业目标

- 使用 Conda 创建独立 Python 环境并安装依赖
- 加载 LCCC-base 原始数据
- 清洗空文本、HTML 标签、特殊字符及异常空白
- 根据可配置敏感词表过滤敏感内容
- 将原始多轮列表转换为 `user / assistant` 角色对话格式
- 按训练集、验证集、测试集进行划分
- 保存为 JSONL，便于后续使用 Transformers / SFT 类训练框架
- 输出清洗统计信息，便于在实验报告中分析处理效果

## 项目结构

```text
myworkmodel/
├── README.md
├── report.md                         # 可直接整理为 Word 的实验报告
├── requirements.txt
├── environment.yml
├── sensitive_words.txt               # 可自行扩展的敏感词表
├── data/
│   └── README.md                     # 数据不直接提交到 GitHub 的说明
└── src/
    └── prepare_lccc.py               # 完整数据预处理程序
```

## 1. 创建环境

```bash
conda env create -f environment.yml
conda activate lccc-preprocess
```

也可以：

```bash
conda create -n lccc-preprocess python=3.10 -y
conda activate lccc-preprocess
pip install -r requirements.txt
```

## 2. 执行处理

默认从 Hugging Face Datasets 加载 LCCC-base：

```bash
python src/prepare_lccc.py --output_dir data/processed
```

如果已经有本地 JSON 文件，也可以：

```bash
python src/prepare_lccc.py \
  --input_file data/raw/LCCC-base.json \
  --output_dir data/processed
```

如果只想快速生成一个小规模实验样本：

```bash
python src/prepare_lccc.py \
  --output_dir data/processed \
  --max_samples 10000
```

## 输出格式

每行一个 JSON 对话样本，例如：

```json
{"messages":[{"role":"user","content":"你好，今天过得怎么样？"},{"role":"assistant","content":"挺好的，你呢？"},{"role":"user","content":"我也不错。"},{"role":"assistant","content":"那就好。"}]}
```

输出文件：

```text
data/processed/
├── train.jsonl
├── valid.jsonl
├── test.jsonl
└── statistics.json
```

## 数据来源

LCCC 由清华大学 CoAI 团队发布，官方项目为 `thu-coai/CDial-GPT`。官方说明显示，LCCC-base 主要来自微博对话，并经过规则与分类器相结合的数据清洗流程；官方项目也说明可以通过 `datasets` 加载：`load_dataset("lccc", "base")`。

论文：Yida Wang et al., *A Large-Scale Chinese Short-Text Conversation Dataset*, 2020。

## 说明

原始数据文件体积较大，因此本仓库只保存**处理程序、配置、敏感词表示例和实验报告**，不直接把完整 LCCC-base 数据上传到 GitHub。这样既避免仓库膨胀，也方便复现实验。
