# NLP Assignment 1 README

This assignment implements a complete pipeline for text cleaning, tokenization (Whitespace, Regex, and BPE), and N-gram Language Modelling with Witten-Bell and Kneser-Ney smoothing.

## Getting Started

### 1. Setup (Optional but Recommended)
It is recommended to use a virtual environment (only for ease as I use tqdm library).
```bash
# Create venv
python3 -m venv venv

# Activate venv
source venv/bin/activate  

# Install dependencies (mainly tqdm for progress bars)
pip install -r requirements.txt
```

### 2. Data Preparation
First, you must run the cleaning script. This script handles the "detritus"(trash characters) (unicode artifacts, excessive spacing) and splits the raw CC-100 data into Training, Validation, and Testing sets.
```bash
python3 clean.py
```
**Note:** This will generate the `.txt` partitions in the `/data/datasets/` directory.

### 3. Training and Evaluation
The `main.py` file is the **"Control centre and the tweaking place"** Open this file to configure your parameters (Language, Training Limits, BPE Merges, etc.) via the toggles at the top of the script.
```bash
python3 main.py
```
This script will:
- Train the BPE tokenizer (if rules don't exist).
- Train/Load all 9 variations of the Language Models.
- Calculate and report Perplexity on the test set.

### 4. Testing and Inference
If you want to see the models in action (Autocomplete), use the `verify_test.py` script. It picks random sentences from the test set and generates predictions.
```bash
python3 verify_test.py
```

---

## Project Structure & Files

I have deviated slightly from the suggested structure to allow for better modularity and for a eased "Control Centre" lilke environment.

### Required Files:
- **`tokenizers.py`**: Contains the logic for the Whitespace, Regex, and BPE tokenizers.
- **`language_models.py`**: Contains the 4-gram Language Model class and the smoothing logic (Witten-Bell/Kneser-Ney).
- **`report.pdf`**: Detailed analysis of results and tokenization behavior.
- **`README.md`**: This file.

### Custom Files:
- **`main.py`**: Instead of a simple `utils.py`, this is the main centre. You can set variables, train, and evaluate everything from here.
- **`verify_test.py`**: A dedicated script for loading saved `.pkl` models and running inference on test data to check for "Sense vs Non-sense" outputs.
- **`clean.py`**: Handles the initial data cleaning and partitioning.

### Data Storage:
- **/data/datasets/**: Stores the cleaned `.txt` files (Train/Val/Test).
- **/data/models/**: Stores the trained `.pkl` Language Models and the BPE `.json` merge rules. *This prevents needing to retrain 800k+ lines on every run.*

---

## Configurations in `main.py`
You can toggle the following variables at the top of `main.py`:
- `LANG`: Switch between English (`en`) and Mongolian (`mn`).
- `LM_TRAIN_LIMIT`: Set the number of lines to train on (e.g., 100,000 or the full 800,000).
- `NUM_MERGES`: Change the BPE merge count (default 3000).
- `TRAIN_MODELS`: Set to `False` if you only want to load existing models and check results.

---
## Few of the results

### Mongolian Dataset (MN)
| Tokenizer | No Smoothing | Witten-Bell | Kneser-Ney |
| :--- | :--- | :--- | :--- |
| Whitespace | 49,377,498.29 | 475.47 | 399.68 |
| Regex | 49,377,498.29 | 475.47 | 399.68 |
| BPE | 22,384.27 | 21.78 | **18.50** |

### English Dataset (EN)
| Tokenizer | No Smoothing | Witten-Bell | Kneser-Ney |
| :--- | :--- | :--- | :--- |
| Whitespace | 154,031,639.96 | 430.53 | 314.47 |
| Regex | 154,031,639.96 | 430.53 | 314.47 |
| BPE | 596,398.92 | 46.44 | **36.45** | 

PS: (Honestly one of the worse ones - very small dataset for training and testing)

PPS: .pkl files and BPE.jsons are present at [Google Drive](https://drive.google.com/drive/folders/1e1enfJz-e9sylHNXA699vf5IKDMHNb29?usp=sharing)