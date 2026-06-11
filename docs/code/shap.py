# pip install shap xgboost 

import xgboost as xgb
import shap
import sklearn.datasets
from sklearn.model_selection import train_test_split

# 1. Prepare data
X, y = shap.datasets.california()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train an XGBoost model
model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# 3. Initialize the TreeSHAP Explainer
# Passing the model directly allows SHAP to optimize computation via TreeSHAP
explainer = shap.Explainer(model, X_train)

# 4. Compute SHAP values for the test set
# This returns an Explanation object containing .values, .base_values, and .data
shap_values = explainer(X_test)

# Multi-row global summary
shap.plots.beeswarm(shap_values)

# Explaining the prediction for the first instance in the test set
shap.plots.waterfall(shap_values[0])