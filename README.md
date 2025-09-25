# P²-DPO Project
![alt text](method.png)
This repository contains the code for generating preference pair data and training a model using the **P²-DPO** method. It also includes evaluation scripts for the **AMBER** and **POPE** benchmarks.

---

## Project Structure

The codebase is organized as follows:

- `Data_gen/`: Contains all scripts for generating preference pair data.  
- `Training/`: Contains the scripts and code for the model training pipeline.  
- `AMBER/`: Contains evaluation scripts for the AMBER benchmark.  
- `POPE/`: Contains evaluation scripts for the POPE benchmark.  

---

## Workflow Guide

Follow these steps to set up the environment, generate data, and run the training and evaluation.

---

### 1. Environment Setup

First, create a virtual environment (e.g., using `conda` or `venv`) and install the required dependencies from the `requirements.txt` file.

<details>
<summary><code>bash</code></summary>

```bash
# Create and activate a conda environment (optional but recommended)
# conda create -n p2dpo python=3.10
# conda activate p2dpo

# Install dependencies
pip install -r requirements.txt
```

</details>

---

### 2. Download Dataset

This project uses the **RLHF-V-Dataset** to generate preference data. Download the dataset from the following link:

🔗 [RLHF-V-Dataset on Hugging Face](https://huggingface.co/datasets/openbmb/RLHF-V-Dataset)

The images and questions from this dataset will be used as the basis for generating preference pairs in the next step.

---

### 3. Preference Pair Generation

The data generation process is divided into two parts: **"Rob" pairs** and **"Focus" pairs**.

---

#### 3.1 Generating "Visual Robustness Preference Pairs"

To generate the "Rob" preference pairs, run the `gendata_llava.py` script. You need to configure the paths and parameters accordingly.

<details>
<summary><code>bash</code></summary>

```bash
python P2_DPO/Data_gen/rob_pair/gen/gendata_llava.py \
    --model-path "/path/to/your/llava_model" \
    --model-base "/path/to/your/model_base" \
    --image-folder "/path/to/downloaded/images" \
    --question-file "/path/to/downloaded/questions.json" \
    --answers-file "/path/to/save/generated_pairs.jsonl" \
    --conv-mode "llava_v1" \
    --noise_step 600 \
    --use_cd \
    --cd_alpha 1 \
    --cd_beta 0.1 \
    --seed 42 \
    --temperature 0.2 \
    --top_p 0.9
```

</details>

**Required Arguments:**

- `--model-path`: Path to the LLaVA model checkpoint.  
- `--model-base`: Path to the base model (if separate).  
- `--image-folder`: Path to the directory containing images from the RLHF-V-Dataset.  
- `--question-file`: Path to the JSON file containing dataset questions.  
- `--answers-file`: Output path for generated preference pairs.  

**Configuration Arguments:**

- `--conv-mode`: Conversation template (default: `llava_v1`)  
- `--noise_step`: Step for adding noise (default: `600`)  
- `--use_cd`: Enable contrastive decoding  
- `--cd_alpha`, `--cd_beta`: Contrastive decoding parameters  
- `--seed`: Random seed  
- `--temperature`, `--top_p`: Sampling parameters  

---

#### 3.2 Generating "Focus-and-Enhance Preference Pairs"

To generate the "Focus" preference pairs, run the `generate.py` script:

<details>
<summary><code>bash</code></summary>

```bash
python P2_DPO/Data_gen/focus_pair/generate.py
```

</details>

> 💡 *Note: Be sure to configure the necessary paths directly inside the `generate.py` script before running.*

---

### 4. Training the Model

Once the preference data is generated, you can start training the model using the provided shell script.

> ⚠️ Before running, open the script and configure the paths to your dataset, model, and output directories.

<details>
<summary><code>bash</code></summary>

```bash
bash P2_DPO/Training/scripts/v1_5/p2_dpo.sh
```

</details>

---

### 5. Evaluation

The project includes evaluation scripts for the **POPE** and **AMBER** benchmarks:

- `P2_DPO/AMBER/`
- `P2_DPO/POPE/`

For detailed usage, refer to the scripts within each directory and the official documentation for each benchmark. The provided scripts are self-contained and can be run directly to evaluate a trained model.
