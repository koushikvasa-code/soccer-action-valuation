# Valuing Soccer Player Actions Using Machine Learning: A Simplified VAEP Approach

CMPE 252, San Jose State University
Authors: Koushik Vasa, Varsana Ilango Gomathynaveena

## Overview

Traditional soccer statistics (goals, assists, shots, tackles) miss many actions that create or prevent scoring chances. This project builds a simplified version of the **VAEP** framework (Decroos et al., KDD 2019). It trains models that predict whether the team in possession will **score** or **concede** within the next 10 actions, given the current game state. The change in these probabilities across consecutive actions is then used to value each action.

Three models are compared: **Logistic Regression** (baseline), **Decision Tree**, and **Gradient Boosting**. They are evaluated with ROC-AUC, Brier score, and log loss, with precision, recall, and F1 as secondary metrics.

**Current status:** the data pipeline and **Experiment 1 (model comparison)** are complete. Experiments 2 to 4 (feature groups, action-type valuation, player-level analysis) are in progress.

## Data

- [StatsBomb Open Data](https://github.com/hudl/open-data), English Premier League 2015/16 (`competition_id=2`, `season_id=27`), 380 matches.
- The data is **not stored in this repository**. The notebook downloads only the files it needs into `open-data/` (about 1 to 2 GB).

## Method

1. Convert StatsBomb events to the SPADL action format (`socceraction`).
2. Build game states from the 3 most recent actions: S_i = {a_(i-2), a_(i-1), a_i}.
3. Label each state: `scores` / `concedes` = 1 if the team scores / concedes within the next 10 actions.
4. Split **by match**: 70% train / 15% validation / 15% test (avoids leakage between actions of the same match).
5. Train the three models for both targets. Select on validation, report the final numbers once on the test set.

## Repository contents

| File | Description |
|---|---|
| `soccer_action_valuation.ipynb` | Full pipeline through Experiment 1 |
| `requirements.txt` | Python dependencies |
| `results/` | Metric tables and plots written by the notebook |
| `.gitignore` | Excludes data, caches, and virtual environments |

## Setup

`socceraction` 1.5.3 supports **Python 3.9 to 3.12**, so use Python 3.12 (not 3.13).

```bash
git clone <your-repo-url>
cd <repo-folder>

python3.12 -m venv .venv312
source .venv312/bin/activate        # Windows: .venv312\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name soccer312 --display-name "Python 3.12 (soccer)"
```

Open the notebook, select the **Python 3.12 (soccer)** kernel, and run all cells. The first run downloads the data and converts 380 matches (a few minutes). Both steps are resumable, so re-running skips finished work.

**Known compatibility note:** `socceraction` 1.5.3 references the old `pandera.SchemaModel` name. The first cell of the notebook aliases it to `pandera.DataFrameModel` when needed.

## Results (Experiment 1)

Fill these in from `results/exp1_test.csv` after running the notebook.

| Target | Model | ROC-AUC | Brier | Log loss |
|---|---|---|---|---|
| scores | Logistic Regression | | | |
| scores | Decision Tree | | | |
| scores | Gradient Boosting | | | |
| concedes | Logistic Regression | | | |
| concedes | Decision Tree | | | |
| concedes | Gradient Boosting | | | |

## Limitations

- One competition and season; results may not generalize to other leagues.
- Event data contains only on-ball actions (no off-ball movement or pressure).
- Goals are rare, so the labels are heavily imbalanced; the project emphasizes probability metrics (AUC, Brier, log loss).
- Action values depend on the quality of the probability models and are not a perfect measure of player quality.

## References

1. T. Decroos, L. Bransen, J. Van Haaren, J. Davis. *Actions Speak Louder than Goals: Valuing Player Actions in Soccer.* KDD 2019. doi:10.1145/3292500.3330758
2. M. Van Roy, P. Robberechts, T. Decroos, J. Davis. *Valuing On-the-Ball Actions in Soccer: A Critical Comparison of xT and VAEP.* AAAI-20 Workshop on AI in Team Sports, 2020.
3. StatsBomb Open Data: https://github.com/hudl/open-data
4. `socceraction`, KU Leuven DTAI Sports Analytics Lab: https://github.com/ML-KULeuven/socceraction
