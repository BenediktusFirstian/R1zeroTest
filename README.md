# DeepSeek R1 Zero Testing

![Python](https://img.shields.io/badge/Python-3.10-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange) ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)

This repository contains a step-by-step implementation of the DeepSeek R1 training methodology as described in the DeepSeek technical report. The implementation is designed to be easy to understand and provides a practical example of training a reasoning-focused language model.

## Project Structure

```
deepseek_r1/
├── __init__.py          # Package initialization
├── imports.py           # Common imports
├── config.py            # Configuration settings and classes
├── data_processing.py   # Dataset loading and preprocessing
├── reward_functions.py  # Reward functions for reinforcement learning
├── model_training.py    # Training functions for R1 Zero and R1
├── inference.py         # Functions for model inference
└── main.py              # Main script to run training and demos
```

## Overview

The DeepSeek R1 training process consists of multiple stages:

1. **R1 Zero**: Train a base model using only reinforcement learning (GRPO algorithm) without supervised fine-tuning
2. **R1 Cold Start**: Create high-quality, structured reasoning data for supervised training
3. **R1 SFT Stage 1**: Fine-tune the base model on cold start data to improve reasoning clarity
4. **Reasoning-Oriented RL**: Apply reinforcement learning to enhance reasoning and fix language mixing
5. **Rejection Sampling**: Generate and filter reasoning examples for a second SFT stage
6. **SFT Stage 2**: Further fine-tune with both reasoning and non-reasoning data
7. **Final RL**: Align the model with human preferences for helpfulness and harmlessness

This implementation simplifies the process to focus on the key stages and make it accessible for educational purposes.

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- Transformers
- TRL (Transformers Reinforcement Learning)

### Installation

```bash
pip install torch transformers datasets trl
```

Additional requirements for math verification:
```bash
pip install latex2sympy2-extended math-verify
```

### Usage

To test the base model:
```bash
python main.py --mode test_base
```

To train the R1 Zero model:
```bash
python main.py --mode train_r1_zero
```

To train the R1 SFT model:
```bash
python main.py --mode train_r1_sft
```

To run demos of the trained models:
```bash
python main.py --mode demo_r1_zero
python main.py --mode demo_r1_sft
```

To run the entire pipeline:
```bash
python main.py --mode all
```

## Training Datasets

This implementation uses the following datasets:
- [AI-MO/NuminaMath-TIR](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR) - Math problems for R1 Zero
- [bespokelabs/Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) - Reasoning data for R1 SFT

## Base Model

The implementation uses [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) as the base model for simplicity and to enable local training on consumer hardware. For better results, consider using a larger model if your hardware permits.

## References

- [DeepSeek-R1 Technical Report](https://arxiv.org/abs/2501.12948)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
