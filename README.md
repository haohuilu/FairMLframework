
# Fairness-Aware Implementation of Machine Learning Models

This project demonstrates an integrated framework for evaluating and mitigating bias in machine learning (ML) models, particularly focusing on demographic fairness across different population groups. The implementation includes:

- **Bias quantification** using Earth Mover’s Distance (EMD) and KL divergence.
- **Fairness evaluation** based on metrics such as demographic parity, equalized odds, and treatment equality.
- **Bias mitigation** through targeted data rebalancing.
- **Performance comparison** of fairness-aware vs. non-fairness-aware models.

## Dataset

The analysis uses the [Heart Disease Prediction Dataset](https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression) from Kaggle, which includes 4,238 samples with demographic and medical attributes.

## Models Used

- Logistic Regression (LR)
- k-Nearest Neighbors (KNN)
- Decision Tree (DT)
- Random Forest (RF)
- Support Vector Machine (SVM)

## File Descriptions

- `Untitled26.py`: Python script implementing the fairness-aware ML pipeline.
- `fairness_ml_implementation.ipynb`: Jupyter Notebook version of the same pipeline.

## How to Run

1. Clone the repository.
2. Install the required packages (e.g., `pandas`, `scikit-learn`, `numpy`).
3. Run the Python script or open the notebook to explore the fairness evaluation workflow.

## Reference

This work is based on the study titled *"Fairness-Aware Implementation of Machine Learning Models"*.

## License

MIT License
