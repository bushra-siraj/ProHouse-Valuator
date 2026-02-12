import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data= pd.read_csv('train.csv')
data.head()

# Basic Data Exploration
print(data.shape)
print(data.info())
print(data.describe())
print(data.isnull().sum())
print("\n🧾 Data Types by Column:")
print(data.dtypes.to_string())

# Identify numerical and categorical features
num_features = data.select_dtypes(include=[np.number]).columns.tolist()
cat_features = data.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Numerical Features ({len(num_features)}):")
print(num_features)

print("-"*180)

print(f"Categorical Features ({len(cat_features)}):")
print(cat_features)

missing= data.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print("\nMissing Values per Column (sorted):")
if missing.empty:
    print("No missing values found!")
else:
    print(missing.to_string())

# Visualize missing values
plt.figure(figsize=(10,6))
missing.plot(kind='bar', color='salmon')
plt.title("Columns with Missing Values")
plt.ylabel("Count of Missing Values")
plt.show()

#univariate analysis for num_features

for col in num_features:
    plt.figure(figsize=(6,4))
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

#univariate analysis for cat_features

selected_cat_features = [
    'Neighborhood', 'HouseStyle', 'Foundation', 
    'KitchenQual', 'CentralAir', 'LotShape'
]

n_cols = 3
n_rows = int(np.ceil(len(selected_cat_features) / n_cols))

# Make the figure large enough
plt.figure(figsize=(n_cols * 6, n_rows * 5))

for idx, col in enumerate(selected_cat_features, 1):
    plt.subplot(n_rows, n_cols, idx)
    
    # Check if column exists
    if col in data.columns:
        sns.countplot(data=data, x=col, order=data[col].value_counts().index, palette='viridis')
        plt.title(f'{col} Distribution', fontsize=14)
        
        plt.xticks(rotation=45, ha='right') # Rotate labels and align right
        plt.xlabel("")
        plt.ylabel("Count", fontsize=12)
    
# Automatically adjust subplot parameters for better fit
plt.tight_layout()

#Bivariate analysis

# --- 1. Categorical vs Numerical (Boxplot) ---
# Visualize how Neighborhood affects SalePrice distribution
plt.figure(figsize=(14, 6))
sns.boxplot(x='Neighborhood', y='SalePrice', data=data, palette='Set3')
plt.xticks(rotation=90)
plt.title('SalePrice Distribution by Neighborhood')
plt.tight_layout()
plt.grid(True)
plt.savefig('plot1.png') 
plt.close()

# --- 2. Numerical vs Numerical (Scatterplot) ---
# Goal: Visualize the relationship between Above Ground Living Area and SalePrice
plt.figure(figsize=(8, 6))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=data, color='teal')
plt.title('SalePrice vs. Above Ground Living Area')
plt.tight_layout()
plt.grid(True)
plt.savefig('plot2.png')
plt.close()

# --- 3. Categorical vs Numerical (Barplot) ---
# Goal: Compare average SalePrice across different House Styles
plt.figure(figsize=(10, 6))
sns.barplot(x='HouseStyle', y='SalePrice', data=data, hue='HouseStyle', palette='Blues')
plt.title('Average SalePrice by House Style')
plt.tight_layout()
plt.savefig('plot3.png')
plt.grid(True)
plt.close()

# --- 5. Categorical vs Categorical (Countplot) ---
# Goal: Distribution of House Styles across different Foundation types
plt.figure(figsize=(12, 6))
sns.countplot(x='Foundation', hue='HouseStyle', data=data, palette='Set2')
plt.title('Distribution of House Styles by Foundation Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plot4.png')
plt.grid(True)
plt.close()

# --- 6. Pairplot for Key Features ---
# See relationships between key numerical features simultaneously

subset_cols = ['GrLivArea', 'TotalBsmtSF', 'GarageCars', 'FullBath', 'YearBuilt', 'SalePrice']
pairplot = sns.pairplot(data[subset_cols], height=2.0)
pairplot.fig.suptitle('Pairplot of Key Numerical Features', y=1.02)
plt.grid(True)
plt.savefig('plot5.png')
plt.close()

# --- 7. Violin Plot (Distribution) ---

plt.figure(figsize=(12, 6))
sns.violinplot(x='OverallQual', y='SalePrice', data=data, palette='Pastel1')
plt.title('SalePrice Distribution by Overall Quality')
plt.tight_layout()
plt.savefig('plot6.png')
plt.close()