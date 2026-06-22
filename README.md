# Fairness-Aware Implementation of Machine Learning Models

This project demonstrates an integrated framework for evaluating and mitigating bias
in machine learning (ML) models, with a focus on demographic fairness across different
population groups. The implementation includes:

- **Bias quantification** using Earth Mover's Distance (EMD) and KL divergence.
- **Fairness evaluation** based on metrics such as demographic parity, equalized odds,
  equal opportunity, treatment equality, disparate impact, and balanced error rate.
- **Statistical testing** of bias via bootstrap and permutation tests (p-values).
- **Bias mitigation** through targeted data rebalancing (instance removal).
- **Performance comparison** of fairness-aware vs. non-fairness-aware models.

## Models Used

- Logistic Regression (LR)
- k-Nearest Neighbors (KNN)
- Decision Tree (DT)
- Random Forest (RF)
- Support Vector Machine (SVM)
- Artificial Neural Network (ANN / MLP)

## Datasets

The primary analysis uses the
[Heart Disease (Framingham) Dataset](https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression)
from Kaggle (4,238 samples with demographic and medical attributes), expected locally
as `framingham.csv`.

The ablation study (`ablation.py`) additionally references three datasets — `D7.csv`,
`D58.csv`, and `D73.csv` — covering cardiovascular, diabetes, and sepsis outcomes.

> **Note:** The CSV data files are **not** included in this repository. Download them
> from their respective sources and place them in the project root before running.

## File Descriptions

| File | Description |
| --- | --- |
| `main.py` | Fairness-aware ML pipeline: group-wise k-fold evaluation, EMD bias test, fairness metrics, and permutation-based p-values. Writes results to `results.xlsx`. |
| `ablation.py` | EMD bias analysis across the D7/D58/D73 datasets after targeted instance removal. Writes `emd_bias_table_after_specified_removal.csv`/`.xlsx`. |
| `jupyter_notebook.ipynb` | Exploratory notebook covering the same analyses, including Fairlearn-based demographic-parity evaluation. |

## Setup

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone the repository.
2. Install the required packages: `pip install -r requirements.txt`.
3. Place the required dataset CSV(s) in the project root (see **Datasets** above).
4. Run a script:
   ```bash
   python main.py        # main fairness-aware pipeline -> results.xlsx
   python ablation.py    # ablation study -> emd_bias_table_after_specified_removal.csv
   ```

## Reference

This work is based on the study titled *"Fairness-Aware Implementation of Machine
Learning Models"*.

## License

[MIT License](LICENSE)
