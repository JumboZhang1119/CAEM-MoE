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
2. Install all dependencies.

   ```
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   conda install -c conda-forge datasets tqdm transformers==4.46.3
   conda install numpy evaluate
   pip install rouge-score
   ```

### 3-3 Patching the transformers Library

> WARNING: Please ensure you are operating within a dedicated virtual environment.

The core logic of CAEM is implemented by modifying two source files within the Hugging Face ``transformers`` library. This step is necessary to inject the custom expert selection and merging functions into the Switch Transformer's architecture. This approach is highly sensitive to the library version, which is why we have locked the ``transformers`` version in ``requirements.txt``.   
1. Locate the transformers library installation path.
  
   > This command finds the installation directory of the transformers library in your environment
   ```
   TRANSFORMERS_PATH=$(python -c "import transformers; import os; print(os.path.dirname(transformers.__file__))")
   ```
2. Apply the patch.

   > Copy ``modeling_switch_transformers.py`` and ``configuration_switch_transformers.py`` to the corresponding directory in the transformers library, replacing the original files.
   ```
   cp./caem_files/modeling_switch_transformers.py $TRANSFORMERS_PATH/models/switch_transformers/
   cp./caem_files/configuration_switch_transformers.py $TRANSFORMERS_PATH/models/switch_transformers/
   ```
   
## 4. Data Preparation

> Script file: ``datasets_preprocessing.py``
   
   ### 4-1 Script Functionality
   
   - Downloads the AG News, SQuAD v1.1, and XSum datasets from the Hugging Face Hub.

   - Unifies all tasks into a sequence-to-sequence format, prepending inputs with task-specific prompts (e.g., ``Classify:``, ``Question:``, ``Context:``, ``Summarize:``).

   - Tokenizes the datasets using T5TokenizerFast with pre-trained weight: ``google/switch-base-8``.

   - Saves the processed datasets to local disk for fast loading during the training phase.

   ### 4-2 Pre-execution settings

   Set the following path: ``AG_NEWS_PATH``, ``SQUAD_PATH``, ``XSUM_PATH``.

   ### 4-3 Execution Command

   ```
   python datasets_preprocessing.py
   ```

## 5. Train Single-Task Models (Prerequisite for CAEM)

   ### 5-1 Pre-execution settings

   Set the following path: ``DATA_DIR``, ``PICKLE_PATH``, ``MODEL_SAVE_PATH`` and global config: ``batch_size``, ``num_expert``, ``gradient_accumulation_steps``, ``task_type``.

   ### 5-2 Execution Command

   ```
   python single_task_finetune.py
   ```

## 6. Baseline Multi-Task Models

   ### 6-1 Pre-execution settings

   Set the following path: ``DATA_DIR``, ``PICKLE_PATH``, ``MODEL_SAVE_PATH`` and global config: ``batch_size``, ``gradient_accumulation_steps``, ``experts_num``, ``current_case_name``.

   > The resulting file name be like: ``{current_case_name}_[{experts_num}]``. (e.g. Baseline_[8])

   ### 6-2 Execution Command

   ```
   python Baseline_multi_task_finetune.py
   ```

## 7. CAEM Multi-Task Models

   ### 7-1 Pre-execution settings

   Set the following path: ``DATA_DIR``, ``SOURCE_MODEL_WEIGHT_PATH``, ``PICKLE_PATH``, ``MODEL_SAVE_PATH`` and global config: ``batch_size``, ``gradient_accumulation_steps``, ``source_experts_num``, ``target_experts_num``, ``current_case_name``, ``alpha``(set to 0 for ablation studies).

   > The resulting file name be like: ``{current_case_name}_[{source_experts_num}|{target_experts_num}]``. (e.g. CAEM_[8|16])

   ### 7-2 Execution Command

   ```
   python CAEM_multi_task_finetune.py
   ```

## 8. Citation

   ```
   @inproceedings{anonymous2024caemmoe,
       title={{CAEM-MOE}: Complexity-Aware Expert Merging for Efficient Multi-Task Learning in Mixture of Experts},
       author={Anonymous},
       booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics},
       year={2024},
       publisher={Association for Computational Linguistics},
       url={...to be updated...},
       note={Code available at: [Link to this GitHub repository]}
   }
   ```




