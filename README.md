## Multilingual Sentiment Analysis (Unsloth & QLoRA)

This project contains the complete code for fine-tuning a Large Language Model (LLM) to perform binary sentiment classification on multi-lingual text. The model is built using the Unsloth library and leverages Parameter-Efficient Fine-Tuning (PEFT) techniques to run on consumer hardware.

## 🚀 Key Features & Goals

The pipeline is engineered for maximum memory efficiency and speed within a Kaggle Docker environment, utilizing **4-bit Quantization** and **Low-Rank Adaptation (LoRA)**.

* **SOTA Architecture:** Utilizes Meta's **LLaMA 3.1 8B Instruct** model as the foundational backbone.
* **Hyper-Optimization:** leverages **Unsloth** to achieve 2x faster training and significantly lower memory usage compared to standard Hugging Face implementations.
* **Parameter Efficiency:** Implements **QLoRA** to fine-tune only a small fraction of the model's parameters while freezing the core weights.
* **Precision:** Uses **Bfloat16** (or Float16) mixed precision training to accelerate computation.

## 📈 Methodology

The core of this solution is **Instruction Tuning** using the SFTTrainer. Instead of retraining the full model, we inject trainable rank decomposition matrices into specific layers of the Transformer architecture.

### 1. Data Formatting (Prompt Engineering)

To align the model with the classification task, the raw CSV data is transformed into a structured instruction format:
* **Structure:** The model is fed a prompt containing an instruction, the input sentence, and the expected label.
* **Template:**
    ```text
    ### Input: {sentence}
    ### Response: {label}
    ```
* **Tokenization:** Data is tokenized with a sequence length of 2048 to handle varying text lengths.

### 2. The Model (LoRA Configuration)

The model used is **unsloth/llama-3-8b-instruct-bnb-4bit**.
* **Quantization:** Loaded in 4-bit precision to fit within GPU memory constraints.
* **Target Modules:** LoRA adapters are attached to all linear layers: `q_proj`, `k_proj`, `v_proj`, `up_proj`, `down_proj`, `o_proj`, and `gate_proj`.
* **Rank & Alpha:** Configured with `r=16` and `lora_alpha=16` to balance plasticity and stability.
* **Gradient Checkpointing:** Enabled to further reduce VRAM usage during backpropagation.

### 3. Training & Inference

* **Optimizer:** `adamw_8bit` (Paged AdamW) to optimize memory.
* **Trainer:** Hugging Face `SFTTrainer` with `packing=True` to maximize training throughput.
* **Inference:** The notebook utilizes `FastLanguageModel.for_inference` which enables native 2x faster inference speeds. It generates tokens, decodes the output, and parses the text to extract "Positive" or "Negative" labels.

## 🛠️ Tech Stack

* **Core:** Python 3
* **LLM Engine:** Unsloth, Transformers (Hugging Face)
* **Fine-Tuning:** PEFT, TRL (SFTTrainer)
* **Hardware Acceleration:** CUDA 12.1, Xformers
* **Data Handling:** Pandas, Datasets
* **Metrics:** Scikit-learn (Accuracy, Confusion Matrix)

## 🏃 Running the Project

### 1. Dependencies

It is highly recommended to run this in a **Kaggle Notebook** (GPU T4 x2 or P100) to ensure compatibility with the pre-compiled CUDA kernels.

```bash
pip install unsloth "xformers<0.0.27" --no-deps trl peft accelerate bitsandbytes
```

### 2. Dataset
   
This model was trained on the multilingual sentiment analysis dataset as part of a university challenge. The data consists of a train.csv and test.csv containing mixed-language sentences and their corresponding sentiment labels. Due to privacy and access restrictions, the dataset is not publicly available and is not included in this repository.

Therefore, the script cannot be run out-of-the-box without downloading the specific competition data separately and placing it in the correct directory structure.


### 3. Notebook Review 

The provided code serves as a reference implementation for efficient LLM fine-tuning, including:


* Model Initialization: Loading 4-bit models and attaching adapters.

* Training Loop: Configuring TrainingArguments for gradient accumulation and learning rate scheduling.

* Submission Generation: A dedicated prediction loop that iterates through the test set, parses the LLM response, and handles edge cases before saving to submission.csv.
