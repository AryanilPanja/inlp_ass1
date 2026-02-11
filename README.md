## README

### Instructions to run:
1. Preferably source `.venv/bin/activate` (recommended for tqdm)
2. Run `python3 clean.py` to clean and split datasets
3. Run `python3 main.py` to train all models
4. Run `python3 verify_test.py` to verify generated outputs

---

## Results

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