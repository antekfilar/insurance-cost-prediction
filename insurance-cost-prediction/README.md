# 💡 Insurance Cost Prediction

This project uses regression models to predict medical insurance charges based on personal and lifestyle-related features.

## 📁 Dataset
Source: [Kaggle – Insurance Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)  
File: `insurance.csv`

Features include:
- age, sex, BMI, number of children,
- smoker status,
- region,
- and resulting `charges`.

## 🔍 Goal
To predict the `charges` (insurance cost) using multiple regression models and compare their performance.

## 📊 Models Used
- Linear Regression
- Ridge Regression
- Lasso Regression
- XGBoost
- LightGBM
- CatBoost

## ✅ Best model
📈 XGBoost showed the lowest RMSE among the tested models.

## 🖼️ Visualizations
- Distributions of features
- Boxplots of categorical variables vs charges
- Feature importance from XGBoost
- Model residuals

## 📦 Dependencies

Install with:
```bash
pip install -r requirements.txt
```

## ▶️ How to run

1. Place `insurance.csv` inside `data/`.
2. Run the script:
```bash
python main.py
```

## 📈 Metrics printed:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score
