import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

# 1. Read your uploaded file
df = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")

# 2. Advanced Text Processing: Extract Brand and Model Series
# Example: "Volkswagen Vento 1.5 TDI" -> Brand: "Volkswagen", Model: "Vento"
df['Brand'] = df['name'].str.split().str[0]
df['Model'] = df['name'].str.split().str[1]

# Drop rows where the model couldn't be parsed properly (rare edge cases)
df.dropna(subset=['Brand', 'Model'], inplace=True)

# Compute Car Age relative to the current year (2026)
current_year = 2026
df['Car_Age'] = current_year - df['year']

# 3. Handle heavy outliers to prevent skewing the linear weights
q_high = df['selling_price'].quantile(0.95)
df_cleaned = df[df['selling_price'] < q_high]

# 4. Create the Hierarchical Brand-to-Model Mapping for the Frontend
brand_model_map = {}
for brand in df_cleaned['Brand'].unique():
    # Get all unique models that belong to this specific brand
    models_for_brand = df_cleaned[df_cleaned['Brand'] == brand]['Model'].unique().tolist()
    brand_model_map[brand] = sorted(models_for_brand)

# Save unique options and the brand-model hierarchy layout
categorical_options = {
    'brand_model_map': brand_model_map,
    'fuel': sorted(df_cleaned['fuel'].unique().tolist()),
    'seller_type': sorted(df_cleaned['seller_type'].unique().tolist()),
    'transmission': sorted(df_cleaned['transmission'].unique().tolist()),
    'owner': sorted(df_cleaned['owner'].unique().tolist())
}
joblib.dump(categorical_options, 'dropdown_options.pkl')

# 5. Drop original raw columns to prevent leakage or redundant features
X = df_cleaned.drop(columns=['name', 'year', 'selling_price'])
y = df_cleaned['selling_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Build pre-processing architecture
numeric_features = ['km_driven', 'Car_Age']
categorical_features = ['Brand', 'Model', 'fuel', 'seller_type', 'transmission', 'owner']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# 7. Assemble Ridge Pipeline and optimize parameters
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge())
])

param_grid = {'regressor__alpha': [0.1, 1.0, 10.0, 100.0]}
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# 8. Serialize the completed ML pipeline logic out to file
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'cardekho_ridge_model.pkl')
print("File system update: 'train.py' has completed successfully with Brand-Model hierarchy!")