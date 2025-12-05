# Multilingual Sentiment Analysis with Unsloth & QLoRA

This project demonstrates how to fine-tune the **LLaMA 3.1 8B Instruct** model for a Multi-Lingual Sentiment Analysis task. It utilizes the **Unsloth** library to achieve faster training and memory efficiency through 4-bit quantization and LoRA (Low-Rank Adaptation).

## 🚀 Key Features

* **Model:** LLaMA 3.1 8B (Instruct version).
* **Optimization:** Uses `unsloth` for 2x faster training and 4-bit quantization (`load_in_4bit`).
* **Fine-Tuning Method:** QLoRA (Quantized Low-Rank Adaptation) targeting specific attention modules (`q_proj`, `k_proj`, etc.).
* **Training Library:** Hugging Face `trl` (SFTTrainer) and `transformers`.
* **Task:** Binary Sentiment Classification (Positive/Negative).

## 🛠️ Prerequisites & Installation

This environment is optimized for a CUDA-enabled GPU (specifically designed to run in a Kaggle Docker environment).

### Dependencies

The script automatically installs the necessary dependencies, including specific CUDA 12.1 compatible versions of PyTorch and Xformers.

```bash
pip install pip3-autoremove
pip-autoremove torch torchvision torchaudio -y
pip install torch xformers --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install unsloth
pip install --no-deps trl peft accelerate bitsandbytes datasets
```

### 📂 Dataset Structure

The script expects a dataset (CSV format) located in /kaggle/input/multi-lingual-sentiment-analysis/.

Required Columns:

sentence: The input text to be analyzed.

label: The sentiment label (e.g., 'Positive', 'Negative').

⚙️ Model Configuration
Base Model

Path: /kaggle/input/llama-3.1/transformers/8b-instruct/2

Sequence Length: 2048 tokens

Dtype: Auto-detected (Float16 or Bfloat16 depending on GPU)

LoRA Configuration

Parameter-Efficient Fine-Tuning is applied with the following settings:

Rank (r): 16

Alpha: 16

Target Modules: q_proj, k_proj, v_proj, up_proj, down_proj, o_proj, gate_proj

Dropout: 0

Bias: None

🧠 Training Pipeline
Prompt Formatting: The input text is wrapped in a structured instruction prompt:

Plaintext
Classify the text into 'Positive', 'Negative', and return the answer as the predicted sentiment.
### Input: {sentence}
### Response: {label} <|end_of_text|>
SFTTrainer Settings:

Optimizer: adamw_8bit (reduces memory usage).

Batch Size: 2 per device (with gradient accumulation steps = 4).

Learning Rate: 3e-4 (Linear scheduler).

Steps: Capped at max_steps=70 for this run (can be adjusted for full convergence).

Packing: Enabled (packs multiple short sequences into the context window).

🔮 Inference & Prediction
The inference process generates labels for the test set:

Fast Inference Mode: FastLanguageModel.for_inference(model) enables native 2x faster inference.

Generation: The model generates a response based on the prompt.

Parsing: The script parses the text after ### Response: to extract the class label.

Fallback: If the model generates an unknown category, it defaults to "Positive".

📊 Output
The script generates a submission.csv file containing:

ID: Row identifier.

label: The predicted sentiment (Positive/Negative).

