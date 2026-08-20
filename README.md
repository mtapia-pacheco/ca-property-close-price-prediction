# California Property Close Price Prediction

This project predicts California residential property close prices from CRMLS sold-listing data. It includes exploratory notebooks, preprocessing, model comparison, final model artifacts, and a small Streamlit app for local predictions.

## Dataset

The raw dataset comes from monthly CRMLS sold-property CSV exports stored outside this repo:

- `CRMLSSold202512.csv`
- `CRMLSSold202601.csv`
- `CRMLSSold202602.csv`
- `CRMLSSold202603.csv`
- `CRMLSSold202604.csv`
- `CRMLSSold202605.csv`
- `CRMLSSold202606.csv`

The target column is `ClosePrice`. Column notes are in `key_columns_desc.pdf`.

## Preprocessing

`02_preprocessing.ipynb` merges the monthly files, parses `CloseDate`, creates `CloseMonth`, and builds a time-based split:

- Train: January 2026 through May 2026
- Test: June 2026

Feature engineering adds simple property features such as `PropertyAge`, `BedBathRatio`, and `LivingAreaPerBedroom`, plus location and school-district fields when available.

## Models Tested

The notebooks compare:

- Linear Regression
- Decision Tree
- Random Forest
- XGBoost
- CatBoost

The final saved model is an XGBoost pipeline in `final_model_outputs/final_model.joblib`.

## Best Results

Best model on the June 2026 test set: **XGBoost**

| Metric | Value |
| --- | ---: |
| Test R² | 0.852 |
| Test MAE | $224,815 |
| Test RMSE | $400,200 |
| Test MAPE | 19.04% |
| Test MdAPE | 12.91% |

The full model comparison is saved in `final_model_outputs/metrics_summary.csv`.

## Re-run the Code

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the raw CRMLS CSV files in `../data/`, then run the notebooks in order:

```bash
jupyter notebook
```

1. `01_exploration.ipynb`
2. `02_preprocessing.ipynb`
3. `03_baseline_model.ipynb`
4. `04_model_comparison.ipynb`
5. `feature_engineering.ipynb`
6. `05_advanced_models.ipynb`
7. `06_evaluation.ipynb`

`06_evaluation.ipynb` saves the final model and metrics into `final_model_outputs/`.

## Launch the App

After installing dependencies, run:

```bash
streamlit run app.py
```

The app loads `final_model_outputs/final_model.joblib` and predicts a close price from basic property inputs.
