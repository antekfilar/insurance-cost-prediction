import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor, plot_importance
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Importing data
data = pd.read_csv(r"C:\Users\User\Downloads\insurance.csv")

# EDA
# Getting familiar with data
print(data.head())
print(data.dtypes)
print(data.nunique()) # there are 2 sexes and 4 regions
print(data["region"].unique())
print(data.isna().sum())

# Adding BMI group to data
data["bmi_group"] = pd.cut(data["bmi"], bins=[0, 18.5, 25, 30, 35, 40, 60],
                           labels=["underweight", "healthy weight", "overweight",
                                   "class 1 obesity", "class2 obesity", "class 3 obesity"])

# Variables' distributions
fig, axes = plt.subplots(3, 2)

sns.histplot(data["age"], ax=axes[0, 0])
axes[0, 0].set_title("Age distribution")
sns.histplot(data["sex"], ax=axes[0, 1])
axes[0, 1].set_title("Sex distribution")
sns.histplot(data["bmi"], ax=axes[1, 0])
axes[1, 0].set_title("BMI distribution")
sns.histplot(data["children"], ax=axes[1, 1])
axes[1, 1].set_title("Children distribution")
sns.histplot(data["smoker"], ax=axes[2, 0])
axes[2, 0].set_title("Smoker distribution")
sns.histplot(data["region"], ax=axes[2, 1])
axes[2, 1].set_title("Region distribution")
plt.tight_layout()
plt.show()

sns.histplot(data["charges"])
plt.title("Medical cost distribution")
plt.show()

fig1, axes = plt.subplots(nrows=2, ncols=1)
sns.boxplot(data, x="age", y="charges",ax=axes[0])
axes[0].set_title("Age vs charges")
sns.boxplot(data, x="sex", y="charges", ax=axes[1])
axes[1].set_title("Sex vs charges")
plt.tight_layout()
plt.grid(True)
plt.show()
fig2, axes = plt.subplots(nrows=2, ncols=1)
sns.boxplot(data, x="bmi_group", y="charges", ax=axes[0])
axes[0].set_title("BMI group vs charges")
sns.boxplot(data, x="children", y="charges", ax=axes[1])
axes[1].set_title("Children vs charges")
plt.tight_layout()
plt.grid(True)
plt.show()
fig3, axes = plt.subplots(nrows=2, ncols=1)
sns.boxplot(data, x="smoker", y="charges", ax=axes[0])
axes[0].set_title("Smoker vs charges")
sns.boxplot(data, x="region", y="charges", ax=axes[1])
axes[1].set_title("Region vs charges")
plt.tight_layout()
plt.grid(True)
plt.show()

# After plots we can see, that older you are, more you pay.
# Smokers pay much more than no smokers.
# Southeast is charged a little bit more.
# Men are charged a little bit more.
# The leap in charges at class 2 obesity is noticeable. In class 1 obesity many outliers.

# Correlations - I have to change categorical data into numerical ones
ordinal_encoder = OrdinalEncoder()
label_data = data.copy()
label_data[["sex", "smoker", "region", "bmi_group"]] = ordinal_encoder.fit_transform(data[["sex", "smoker", "region", "bmi_group"]])
sns.heatmap(label_data.corr())
plt.title("Correlation heatmap")
plt.show()

# Target and feature data
y = data.charges
X = data.drop("charges", axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)
# Ordinal encoding
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()
label_X_train[["sex", "smoker", "region", "bmi_group"]] = ordinal_encoder.fit_transform(X_train[["sex", "smoker", "region", "bmi_group"]])
label_X_valid[["sex", "smoker", "region", "bmi_group"]] = ordinal_encoder.transform(X_valid[["sex", "smoker", "region", "bmi_group"]])

# Models
# Linear Regression
ss = StandardScaler()
my_model_1 = LinearRegression()
my_model_1.fit(ss.fit_transform(label_X_train), y_train)
preds_1 = my_model_1.predict(ss.transform(label_X_valid))

# Lasso
my_model_2 = Lasso()
my_model_2.fit(ss.fit_transform(label_X_train), y_train)
preds_2 = my_model_2.predict(ss.transform(label_X_valid))

# XGBoost
my_model_3 = XGBRegressor(n_estimators=25, learning_rate=0.1)
my_model_3.fit(label_X_train, y_train)
preds_3 = my_model_3.predict(label_X_valid)
plot_importance(my_model_3, importance_type="gain")
plt.title("Feature importance due XGBoost, importance_type=gain")
plt.show()

# Ridge
my_model_4 = Ridge(alpha=0.01)
my_model_4.fit(ss.fit_transform(label_X_train), y_train)
preds_4 = my_model_4.predict(ss.transform(label_X_valid))

# LGBM
my_model_5 = LGBMRegressor(verbose=0)
my_model_5.fit(label_X_train, y_train)
preds_5 = my_model_5.predict(label_X_valid)

# CatBoost
my_model_6 = CatBoostRegressor(verbose=False)
my_model_6.fit(X_train, y_train, cat_features=["sex", "smoker", "region", "bmi_group"])
preds_6 = my_model_6.predict(X_valid)

# Plotting the outcomes
plt.plot([i for i in range(1, len(y_valid)+1)], y_valid - preds_1, color='red')
plt.plot([i for i in range(1, len(y_valid)+1)], y_valid - preds_2, color='blue')
plt.plot([i for i in range(1, len(y_valid)+1)], y_valid - preds_3, color='purple')
plt.plot([i for i in range(1, len(y_valid)+1)], y_valid - preds_4, color='green')
plt.show()

# Showing accuracy
acc = {
    'Model': ['Linear Regression', 'Lasso', 'XGBoost', 'Ridge', 'LGBM', 'CatBoost'],
    'MAE': [mean_absolute_error(preds_1, y_valid), mean_absolute_error(preds_2, y_valid),
            mean_absolute_error(preds_3, y_valid), mean_absolute_error(preds_4, y_valid),
            mean_absolute_error(preds_5, y_valid), mean_absolute_error(preds_6, y_valid)],
    'MSE': [mean_squared_error(preds_1, y_valid), mean_squared_error(preds_2, y_valid),
            mean_squared_error(preds_3, y_valid), mean_squared_error(preds_4, y_valid),
            mean_squared_error(preds_5, y_valid), mean_squared_error(preds_6, y_valid)],
    'R2 score': [r2_score(preds_1, y_valid), r2_score(preds_2, y_valid), r2_score(preds_3, y_valid),
                 r2_score(preds_4, y_valid), r2_score(preds_5, y_valid), r2_score(preds_6, y_valid)],
    'RMSE': [root_mean_squared_error(preds_1, y_valid), root_mean_squared_error(preds_2, y_valid),
             root_mean_squared_error(preds_3, y_valid), root_mean_squared_error(preds_4, y_valid),
             root_mean_squared_error(preds_5, y_valid), root_mean_squared_error(preds_6, y_valid)]
}
df_acc = pd.DataFrame(acc)
print(df_acc)