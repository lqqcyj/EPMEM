<div align="center">

# 🧠 EPMEM

> **Preserving Context, Retracing Evidence: Training-Free Emotion Understanding in Multimodal Large Language Models**

[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2%2B-orange)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-blue)](https://huggingface.co/transformers)

</div>

---

## 📌 Overview

**EPMEM** is a **training-free** inference framework for multimodal emotion understanding built on top of LLaVA-style MLLMs. It addresses two core challenges:

- 🔴 **Visual Grounding Decay** — Visual attention weakens during deep autoregressive decoding, causing models to over-rely on language priors and miss subtle emotional cues.
- 🔵 **Global-Local Semantic Interplay** — Emotions arise from interactions among multiple visual elements, not isolated regions.

<div align="center">
<img src="assets/framework.png" width="90%" alt="EPMEM Framework"/>
</div>

### How It Works

EPMEM operates in two stages:

| Stage | Name | Key Components |
|---|---|---|
| **Stage 1** | Emotional Perception Augmentation | Confidence-Guided Polarity Classification + EEG-guided EVE extraction |
| **Stage 2** | Visual-Retracing Augmented Inference | ATC Prompting + Entropy-Triggered EFVR |

---

## 📊 Main Results

Accuracy (%) on fine-grained emotion benchmarks:

| Model | Method | Emotion6 | EmoSet | WebEmo7 | WebEmo25 | Artphoto | Avg |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| LLaVA-1.5-7B | Zero-shot | 46.13 | 53.95 | 25.10 | 19.00 | 39.21 | 36.68 |
| | Zero-shot-CoT | 47.64 | 54.41 | 28.90 | 19.15 | 39.70 | 37.96 |
| | SEPM | 54.21 | 56.04 | 42.39 | 18.26 | 43.67 | 42.91 |
| | **EPMEM (Ours)** | **55.56** | **56.39** | **45.85** | **20.70** | **46.90** | **45.08** |
| LLaVA-1.5-13B | Zero-shot | 55.56 | 58.90 | 35.70 | 22.50 | 44.79 | 43.49 |
| | **EPMEM (Ours)** | **61.45** | **61.15** | **43.25** | **23.54** | **47.02** | **47.28** |
| Qwen2.5-VL-7B | Zero-shot | 55.89 | 58.65 | 42.25 | 21.45 | 40.69 | 43.79 |
| | **EPMEM (Ours)** | **59.60** | **60.93** | **44.55** | **22.30** | **43.67** | **46.21** |

---

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/EPMEM.git
cd EPMEM

# 2. Create conda environment
conda create -n epmem python=3.10
conda activate epmem

# 3. Install dependencies
pip install -r requirements.txt
```

### Core Dependencies

```
torch>=2.1.2
torchvision>=2.1.2
transformers
tokenizers
accelerate
peft
pillow
sentencepiece
einops
tqdm
numpy
shortuuid
```

---

## 📁 Project Structure

```
EPMEM/
├── data/
│   └── stage_1.jsonl              # Evaluation data (JSONL format)
├── llava/
│   ├── model/
│   │   ├── language_model/
│   │   │   ├── qwen_vl.py         # Qwen-VL model integration
│   │   │   └── modelling_logits_llama.py  # Logits processing
│   │   ├── multimodal_encoder/    # CLIP vision backbone
│   │   ├── multimodal_projector/
│   │   │   └── builder.py         # MLP / Linear projector
│   │   └── llava_arch.py          # Base LLaVA architecture
│   └── eval/
│       ├── model_vqa_loader_qwen.py   # VQA data loader (Qwen)
│       ├── utils_zero_shot.py         # Zero-shot inference
│       ├── utils_zero_shot_cot.py     # Chain-of-Thought inference
│       ├── utils_prompt2.py           # ATC prompt variant
│       ├── utils_prompt3.py
│       ├── utils_prompt4.py
│       ├── utils_prompt5.py
│       └── utils_prompt6.py
├── qwen.sh                        # Launch script
├── requirements.txt
└── README.md
```

---

## ⚙️ Usage

### Quick Start

```bash
bash qwen.sh
```

### Manual Inference

```bash
python llava/eval/model_vqa_loader_qwen.py \
  --model-path /path/to/pretrained/weights \
  --image-folder /path/to/images \
  --question-file data/stage_1.jsonl \
  --answers-file outputs/answers.jsonl \
  --beta 0.3 \
  --alpha 0.12 \
  --gamma 0.7 \
  --rho 0.1
```

### Prompting Modes

| Script | Mode | Description |
|---|---|---|
| `utils_zero_shot.py` | Zero-Shot | Direct inference, no CoT |
| `utils_zero_shot_cot.py` | Zero-Shot CoT | "Let's think step by step" |
| `utils_prompt2~6.py` | ATC Variants | Analyze-Then-Choose prompting |

---

## ⚙️ Key Hyperparameters

| Parameter | Symbol | Description | Default |
|---|---|---|---|
| `--alpha` | $$\alpha$$ | Confidence threshold for polarity classification | `0.12` |
| `--beta` | $$\beta$$ | Redundancy ratio for EVE token masking | `0.3` |
| `--gamma` | $$\gamma$$ | Entropy threshold for EFVR trigger | `0.7` |
| `--rho` | $$\rho$$ | Injection strength for visual memory refreshing | `0.1` |
| `--top_k` | $$K$$ | Top-K tokens for entropy computation | `6` |
| `--l_start` | $$l_{start}$$ | Start layer for EFVR triggering | `11` |
| `--l_end` | $$l_{end}$$ | End layer for EFVR triggering | `20` |

---

## 📦 Supported Datasets

| Dataset | Classes | Download |
|---|---|---|
| Emotion6 | 6 | [Link](http://chenlab.ece.cornell.edu/downloads.html) |
| EmoSet | 8 | [Link](https://vcc.tech/EmoSet) |
| WebEmo (7 / 25) | 7 / 25 | [Link](https://rpand002.github.io/emotion.html) |
| Artphoto | 8 | [Link](https://dl.acm.org/doi/10.1145/1873951.1873965) |

Place downloaded datasets under the `data/` directory.

---

## 🏗️ Supported Backbones

| Backbone | Size | HuggingFace |
|---|---|---|
| LLaVA-1.5 | 7B | [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf) |
| LLaVA-1.5 | 13B | [llava-hf/llava-1.5-13b-hf](https://huggingface.co/llava-hf/llava-1.5-13b-hf) |
| Qwen2.5-VL | 7B | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{epmem2026,
  title     = {Beyond Visual Forgetting: Emotion-Focused Visual Retracing for
               Training-Free Multimodal Emotion Understanding},
  author    = {Anonymous},
  booktitle = {Proceedings of the International Joint Conference on
               Artificial Intelligence (IJCAI)},
  year      = {2026}
}
```

---

## 🙏 Acknowledgements

This project builds upon:
- [LLaVA](https://github.com/haotian-liu/LLaVA) — Large Language and Vision Assistant
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) — Qwen Vision-Language Model
- [CLIP](https://github.com/openai/CLIP) — Vision backbone

---

<div align="center">
<sub>⭐ If you find EPMEM helpful, please consider giving it a star!</sub>
</div>