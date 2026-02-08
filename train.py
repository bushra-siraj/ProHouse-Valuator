from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load the dataset
data = pd.read_csv('train.csv')

# --- 1. CLEANING STEP ---
# A. Drop columns that are completely empty
data = data.dropna(axis=1, how='all')

# B. Handle Duplicate Columns
data = data.loc[:, ~data.columns.duplicated()]

# C. Fill missing values in target column
data = data.dropna(subset=['SalePrice'])

# D. Define Features Strictly
num_features = ['GrLivArea', 'FullBath', 'YearBuilt'] # Only numeric
selected_cat_features = [
    'Neighborhood', 'HouseStyle', 'Foundation', 
    'KitchenQual', 'CentralAir', 'LotShape'
]
final_features = num_features + selected_cat_features

# --- 2. Outlier Handling (IQR Clipping) ---
# Ensure only numeric columns are clipped
for col in num_features:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    data[col] = np.clip(data[col], lower, upper)

print("Outliers handled using IQR clipping")

# --- 3. Preprocessing Pipelines ---
# Numerical Pipeline: Impute missing values with median, then scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical Pipeline: Impute missing values with mode, then one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Apply to columns
full_preprocessor = ColumnTransformer(
    transformers=[
        ('num_features', numeric_transformer, num_features),
        ('selected_cat_features', categorical_transformer, selected_cat_features)
    ],
    remainder='drop' 
)

# --- 4. Split Data ---
X = data[final_features] 
y = data['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# --- 5. Create and Train Pipeline ---
model_lr = Pipeline(steps=[
    ('preprocessor', full_preprocessor),
    ('regressor', LinearRegression())
])

# Fit the entire pipeline
model_lr.fit(X_train, y_train)
print("Pipeline fitted successfully!")

# --- 6. Evaluation ---
train_preds = model_lr.predict(X_train)
test_preds = model_lr.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

print(f"Train RMSE (Dollars): ${train_rmse:,.2f}")
print(f"Test RMSE (Dollars): ${test_rmse:,.2f}")

# --- 7. Visualization ---
plt.figure(figsize=(8, 6))
plt.scatter(y_train, train_preds, alpha=0.5)
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted (Training Data)')
plt.plot([y_train.min(), y_train.max()],
         [y_train.min(), y_train.max()],
         'r--', lw=2)
plt.tight_layout()
plt.show()

# --- 8. Save the Model ---
joblib.dump(model_lr, 'house_model.pkl')
print("Model saved as house_model.pkl")