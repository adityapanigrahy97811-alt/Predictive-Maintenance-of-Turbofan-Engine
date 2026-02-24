# ==============================================
# MACHINE LEARNING PROJECT
# Predictive Maintenance - Turbofan Engine
# ==============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

column_names = ['unit_id','cycle','op1','op2','op3'] + \
               [f'sensor_{i}' for i in range(1,22)]

train = pd.read_csv('train_FD001.txt', sep='\s+', header=None)
train.columns = column_names

# --------------------------------------------------
# 2. CREATE RUL
# --------------------------------------------------

rul = train.groupby('unit_id')['cycle'].max().reset_index()
rul.columns = ['unit_id', 'max_cycle']

train = train.merge(rul, on='unit_id')
train['RUL'] = train['max_cycle'] - train['cycle']
train.drop('max_cycle', axis=1, inplace=True)

# --------------------------------------------------
# 3. DROP UNNECESSARY COLUMNS
# --------------------------------------------------

train.drop(['op1','op2','op3'], axis=1, inplace=True)

X = train.drop(['RUL','unit_id'], axis=1)
y = train['RUL']

# --------------------------------------------------
# 4. SCALE FEATURES
# --------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# 5. TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# 6. DEFINE MODELS
# --------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42)
}

# --------------------------------------------------
# 7. TRAIN & EVALUATE
# --------------------------------------------------

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results.append([name, rmse, mae, r2])

# --------------------------------------------------
# 8. DISPLAY RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results, columns=["Model","RMSE","MAE","R2"])
print("\nModel Comparison:")
print(results_df)

# --------------------------------------------------
# 9. VISUALIZE BEST MODEL
# --------------------------------------------------

best_model_name = results_df.sort_values("RMSE").iloc[0]["Model"]
print("\nBest Model:", best_model_name)

best_model = models[best_model_name]
best_predictions = best_model.predict(X_test)

plt.figure(figsize=(10,5))
plt.plot(y_test.values[:200], label="Actual RUL")
plt.plot(best_predictions[:200], label="Predicted RUL")
plt.legend()
plt.title(f"Actual vs Predicted RUL ({best_model_name})")
plt.show()
