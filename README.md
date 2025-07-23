# CAEM-MoE
Official Code for Complexity-Aware Expert Merging for Efficient Multi-Task Learning in Mixture of Experts

## 1. Introduction

This repository contains the official implementation for the 2025 AACL paper, **CAEM-MOE: Complexity-Aware Expert Merging for Efficient Multi-Task Learning in Mixture of Experts**. 
It provides all the necessary code, scripts, and instructions to reproduce the experimental results presented in the paper.

Paper Link: [Link to Paper (to be updated with arXiv/ACL Anthology link)]

<img width="1290" height="507" alt="截圖 2025-07-24 凌晨3 05 39" src="https://github.com/user-attachments/assets/b067b067-cb7a-4624-b1e2-7cf476f791ed" />



## 2. Abstract of Core Contribution
We introduce a novel, one-shot strategy for merging Mixture-of-Experts (MoE) models, termed **Complexity-Aware Expert Merging (CAEM)**. The core mechanism of CAEM is to use the entropy of expert utilization during single-task fine-tuning as a quantitative indicator of task complexity. This principle enables a strategic and principled allocation of expert capacity when consolidating multiple single-task models into a unified multi-task model.   

This approach allows for a strategic allocation of expert resources, overcoming common MTL bottlenecks. Our method achieves a significant improvement over standard MTL baselines, exemplified by a 6.47\% ROUGE-L gain on the complex XSum task with only negligible trade-offs on other simpler tasks (8-experts case) because of the superior starting point in the loss landscape and path dependency in optimization. This observation led us to identify a more general principle: a "Founder Effect" in model merging. CAEM not only provides a resource-efficient path to high-performance MTL but also provides insights into the mechanisms of model merging.

## 3. Environment Setup
### 3-1 Prerequisites

- Python (version 3.9+ recommended)

- NVIDIA GPU with CUDA and corresponding drivers (VRAM >= 48 recommended).

- To ensure full reproducibility, all experiments were conducted on an NVIDIA RTX A6000 GPU.

### 3-2 Create Virtual Environment and Install Dependencies

It is **highly recommended** to use a virtual environment (e.g., ``venv`` or ``conda``) to avoid package conflicts.  
(Cause we will directly modify the underlying files of the ``transformer``)

1. Create and activate the virtual environment:
   
   ```
   python -m venv venv
   source venv/bin/activate
   # Or using conda
   # conda create -n caem python=3.9
   # conda activate caem
   ```
2. Create a ``requirements.txt`` file with the following content. Note that the ``transformers`` package version is pinned, which is critical for the patching step below.

   ```
   # requirements.txt
   torch
   transformers==4.46.3
   datasets
   accelerate
   sentencepiece
   evaluate
   rouge_score
   scikit-learn
   numpy
   ```
3. Install all dependencies.

   ```
   pip install -r requirements.txt
   ```
   





