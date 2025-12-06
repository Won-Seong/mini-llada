# 🇰🇷 Mini-LLaDA: Korean Small Language Diffusion Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-Apache_2.0-blue)

## 📖 Introduction
**Mini-LLaDA**는 기존의 Autoregressive(GPT 방식) 생성 모델이 아닌, **Masked Diffusion** 방식을 적용한 0.3B 규모의 한국어 소형 언어 모델(SLM)입니다.

본 프로젝트는 [LLaDA: Large Language Diffusion Models](https://arxiv.org/abs/2502.09992) 논문의 핵심 아이디어를 기반으로 하며, 사전 학습된 Encoder 기반 모델인 **RoBERTa**를 Diffusion Generator로 전환(Upcycling)하여 텍스트 생성 능력을 부여하는 실험적인 연구 프로젝트입니다.

## ✨ Key Features
- **Generative Adaptation:** 판별(Discriminative) 모델인 RoBERTa를 생성(Generative) 모델로 성공적으로 전환.
- **Efficient SLM (0.3B):** 3억 개의 파라미터로 구성된 경량 모델로, 제한된 컴퓨팅 자원에서의 학습 및 추론 최적화.
- **Custom Diffusion Sampler:**
  - **Monotonic Unmasking:** 생성된 토큰의 일관성을 유지하기 위해, 이미 예측된 토큰을 보존하는 단조 언마스킹 로직 구현.
  - **Low-confidence Remasking:** 신뢰도가 낮은 토큰을 재조정하여 0.3B 모델의 생성 품질을 보완.
- **EOS-Aware Training:** SFT 단계에서 EOS 토큰을 강제 패딩(Mask=1)하여 학습시킴으로써, Diffusion 모델이 문장 종료 시점을 스스로 제어하도록 설계.

## 🏗️ Architecture & Methodology
### 1. Model Structure
- **Backbone:** `klue/roberta-small` (or similar BERT-based models)
- **Framework:** Continuous Pre-training → Supervised Fine-Tuning (SFT)
- **Mechanism:** Bidirectional Context를 활용한 Masked Diffusion Process ($t=1 \to t=0$)

### 2. Training Strategy
- **Pre-training:** Wikipedia 및 News Corpus를 활용하여 Diffusion 프로세스 적응 (Adaptation).
- **SFT (Supervised Fine-Tuning):** Q&A 데이터셋을 활용하여 Instruction Following 능력 주입.
- **Length Control:** EOS 토큰 비중이 높은 데이터의 편향을 제어하기 위해 추론 단계에서의 Logit Suppression 기법 적용.

## 📊 Experiments & Data
Base Data: wikimedia/wikipedia (Korean subset), AI-Hub Text Data.

SFT Data: Custom QA Datasets.

Evaluation: 모델의 크기(0.3B) 한계로 인해 복잡한 추론보다는 문장 완성도(Fluency)와 문법적 정확성(Grammatical Correctness), 그리고 Diffusion 기반 생성 가능성 검증에 초점을 맞춤.

## ⚠️ Limitations
Model Capacity: 0.3B의 작은 파라미터 수로 인해 복잡한 논리 추론이나 긴 문맥 유지에는 한계가 있습니다.

Inference Speed: Diffusion 특성상 Autoregressive 모델 대비 생성 속도가 느릴 수 있습니다 (Iterative Denoising).

EOS Bias: 학습 데이터의 패딩 비중으로 인해 추론 시 EOS 토큰 생성 경향이 강할 수 있으며, 이는 Sampler 파라미터로 제어 가능합니다.

## 📚 References
This project is heavily inspired by the following paper:

@article{nie2024llada,
  title={LLaDA: Large Language Diffusion Models},
  author={Nie, Shen and Zhu, Fengqi and others},
  journal={arXiv preprint arXiv:2502.09992},
  year={2024}
}

## 👨‍💻 Author
[Sungwon Kim] - Project Lead & Implementation
Interest: LLM, Diffusion Models, NLP
