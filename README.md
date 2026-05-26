<div align="center">

# 🧠 EPMEM

> **Preserving Context, Retracing Evidence: Training-Free Emotion Understanding in Multimodal Large Language Models**

</div>

---

## 🛠️ Installation

### 1. Clone the repository

> 🔒 This is an anonymous repository for peer review. The link will be updated upon paper acceptance.

```bash
# Clone via the anonymous link provided in the paper submission
git clone https://anonymous.4open.science/r/EPMEM
cd EPMEM
```

### 2. Create conda environment
```bash
conda create -n epmem python=3.10
conda activate epmem
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install flash-attention (required, install separately)
```bash
# Option A: Install from prebuilt wheel (recommended)
pip install flash_attn==2.8.3+cu12torch2.5 \
  --find-links https://github.com/Dao-AILab/flash-attention/releases

# Option B: Build from source
pip install flash-attn --no-build-isolation
```

> ⚠️ `flash_attn` requires **CUDA 12.1** and **PyTorch 2.5.1**. Make sure your environment matches before installing.

### Environment Summary

| Package | Version |
|---|---|
| Python | 3.10 |
| torch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| transformers | 5.5.1 |
| accelerate | 1.13.0 |
| tokenizers | 0.22.2 |
| einops | 0.8.2 |
| pillow | 12.2.0 |
| qwen-vl-utils | 0.0.14 |
| flash_attn | 2.8.3+cu12torch2.5 |
| CUDA | 12.1 |

---

## 📁 Project Structure

```
EPMEM/
├── data/
│   ├── stage_1.jsonl                    # Evaluation data (JSONL format)
│   └── artphoto/                        # Artphoto dataset
│       └── image/
├── llava/
│   ├── model/
│   │   ├── language_model/
│   │   │   ├── qwen_vl.py              # Qwen-VL model integration
│   │   │   └── modelling_logits_llama.py  # Logits processing
│   │   ├── multimodal_encoder/         # CLIP vision backbone
│   │   ├── multimodal_projector/
│   │   │   └── builder.py              # MLP / Linear projector
│   │   └── llava_arch.py               # Base LLaVA architecture
│   ├── eval/
│   │   ├── model_vqa_loader_qwen.py    # Main inference entry
│   │   ├── utils_zero_shot.py          # Zero-shot inference
│   │   ├── utils_zero_shot_cot.py      # Chain-of-Thought inference
│   │   └── eval_moset                  # eval accuracy
│   ├── utils_zero_shot.py              # Zero-shot inference
│   ├── utils_zero_shot_cot.py          # Chain-of-Thought inference
│   └── utils.py                           # ATC prompt variants
│
├── qwen.sh                             # Launch script
├── requirements.txt
└── README.md
```

---
## 📦 Supported Datasets

| Dataset | Classes | Download |
|---|---|---|
| Artphoto | 8 | [Link](https://www.imageemotion.org/) |

Place downloaded datasets under the `data/` directory (e.g., `data/artphoto/`).

## ⚙️ Usage

### Quick Start

```bash
bash qwen.sh
```

### Manual Inference

```bash
python llava/eval/model_vqa_loader_qwen.py \
  --model-path /path/to/pretrained/weights \
  --image-folder /path/to/image \
  --question-file data/stage_1.jsonl \
  --answers-file outputs/answers.jsonl \
  --alpha 0.12 \
  --beta 0.1 \
  --gamma 0.7 \
  --rho 0.1
```

### Prompting Modes

| Script | Mode | Description |
|---|---|---|
| `utils_zero_shot.py` | Zero-Shot | Direct inference without CoT |
| `utils_zero_shot_cot.py` | Zero-Shot CoT | "Let's think step by step" |
| `utils.py` | ATC Variants | Analyze-Then-Choose prompting |

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
