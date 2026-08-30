# Bank Deposit Subscription Prediction

This repository contains a machine-learning study of a Portuguese bank's phone marketing data. The objective is to estimate whether a contacted customer will open a term deposit. The full analysis, charts, experiments, and interpretation are in [notebooks/mid_term_project.ipynb](notebooks/mid_term_project.ipynb).

## Data

The project uses the [Bank Marketing dataset from Kaggle](https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv). Each record includes customer details, campaign-contact information, previous marketing outcomes, and the target variable `y`, which shows whether the customer subscribed to a term deposit.

The analysis removes duplicate rows, checks completeness, investigates numeric distributions and possible outliers, and creates `previously_contacted` from the `pdays` sentinel value. The `duration` column is deliberately excluded from modelling: call duration is only known after the call ends, so it would not be available for a pre-call prediction.

## Modelling Workflow

- Split the data with stratification to preserve the minority subscription class.
- Transform features inside scikit-learn pipelines: numerical variables are scaled where needed and categorical variables are one-hot encoded.
- Compare Logistic Regression, k-Nearest Neighbors, Decision Tree, and Gradient Boosting using F1-score as the primary measure.
- Report accuracy, precision, recall, balanced accuracy, ROC-AUC, and PR-AUC alongside F1.
- Tune the Decision Tree with `RandomizedSearchCV` and tune Gradient Boosting using both `RandomizedSearchCV` and Hyperopt.
- Choose a classification threshold from validation probabilities, then inspect feature importance, SHAP explanations, and the confusion matrix.

## Current Results

The initial model comparison used 10-fold stratified cross-validation on the training partition. The baseline Decision Tree was the strongest of the four candidate models:

| Model | CV F1 | Validation F1 | Validation ROC-AUC |
|---|---:|---:|---:|
| Decision Tree | 0.454 | 0.487 | 0.797 |
| Logistic Regression | 0.451 | 0.462 | 0.800 |
| k-Nearest Neighbors | 0.363 | 0.377 | 0.756 |
| Gradient Boosting | 0.328 | 0.300 | 0.806 |

Decision Tree randomized search improved validation F1 to `0.523`. Its selected settings are `criterion='entropy'`, `max_depth=6`, `min_samples_split=20`, and `min_samples_leaf=2`. The threshold sweep selected `0.55`, yielding validation precision `0.464`, recall `0.602`, and F1 `0.524`.

For Gradient Boosting, RandomizedSearchCV achieved validation F1 `0.370` and Hyperopt achieved `0.346`. Both methods produced the same validation ROC-AUC of `0.811`.

These values are development results, not a final unbiased estimate. The validation partition was used to compare models, select parameters, and choose the operating threshold. A separate, untouched test partition should be added and evaluated once after all development decisions are complete.

## Interpretation

The fitted Decision Tree attributes most of its importance to macroeconomic context, particularly `nr.employed`, followed by `cons.conf.idx`. SHAP plots complement this global view by showing how transformed variables influence individual predictions. These explanations describe the model's associations and must not be interpreted as causal effects.

## Repository Layout

```text
data/
	bank-additional-full.csv       Source dataset
notebooks/
	mid_term_project.ipynb         Analysis and modelling workflow
src/
	helper_functions.py            Reusable plots and preprocessing helpers
```

## Requirements

Run the notebook with Python and Jupyter installed. Its main dependencies are:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
hyperopt
shap
```

## Running the Analysis

1. Create and activate a Python environment.
2. Install the listed dependencies.
3. Open `notebooks/mid_term_project.ipynb` in Jupyter or VS Code.
4. Run the cells in order from top to bottom.

Running sequentially is important because later sections reuse the chosen pipeline, threshold, transformed feature names, and predictions created earlier in the notebook.