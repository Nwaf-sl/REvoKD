# REvoKD
**Self-Evolutionary Reinforced Knowledge Distillation for Multi-Modal Tool-Use Agents**


## 🛠 Installation & Environment Setup

### Prerequisites
Before running the code, ensure:
- **Python** version ≥ 3.10
- **PyTorch** installed with CUDA support
- Installed dependencies:
  - [ms-swift](https://github.com/modelscope/ms-swift)
  - [Verl](https://github.com/verl-project/verl)

---

## 📦 Installation Steps

### 1. Clone the repository
```bash
git clone https://github.com/Nwaf-sl/ReEvoKD.git
cd ReEvoKD
```


 
### 2. Install dependencies

```bash
pip install -r requirements.txt
```


## 🚀 Usage
1️⃣ Supervised Fine-Tuning (SFT)

Run the SFT training script:

```bash
sh train_lora_agent.sh
```

2️⃣ Reinforcement Learning (RL)

Before starting RL training: Configure Search Tool API credentials in the appropriate config files.
Run:

```bash
sh run_qwen3vl-4b_multiturn.sh
```
3️⃣ Inference & Evaluation

Before running evaluation:Edit configs/eval.yaml
Set your API key and Search Tool API credentials
Run:

```bash
python eval.py
```


