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
📈 CatBoost showed the best scores in metrics among the tested models.

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

## 📌 Conclusions
- All linear models (Linear Regression, Ridge, Lasso) performed similarly, with RMSE around 5625, indicating they capture basic trends but miss nonlinear patterns.
- CatBoost outperformed all other models, achieving the lowest RMSE, suggesting it effectively captures complex interactions between categorical and numerical features.
- XGBoost and LightGBM also performed better than linear models, but slightly worse than CatBoost.
- Key factors influencing medical insurance charges:
  - Smoking is the strongest cost driver — smokers pay significantly more.
  - Age correlates positively with cost — older individuals pay more.
  - BMI affects cost especially in class 2 and 3 obesity groups.
- Using models that handle categorical features natively (like CatBoost) provides an advantage in this dataset.
- Future improvements may include stacking ensemble methods, hyperparameter tuning, and feature engineering.
