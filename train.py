import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

# 1. Read the uploaded dataset file
df = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")

# 2. Extract Brand name and Model name
# Example: "Maruti 800 AC" -> Brand: "Maruti", Model: "800"
df['Brand'] = df['name'].str.split().str[0]
df['Model'] = df['name'].str.split().str[1]

# Drop any row where the model split failed (edge cases)
df.dropna(subset=['Brand', 'Model'], inplace=True)

# Compute Car Age relative to the current year (2026)
current_year = 2026
df['Car_Age'] = current_year - df['year']

# 3. Handle heavy outliers to keep linear weights steady
q_high = df['selling_price'].quantile(0.95)
df_cleaned = df[df['selling_price'] < q_high]

# 4. Create the Hierarchical Brand-to-Model Mapping for the app dropdowns
brand_model_map = {}
for brand in df_cleaned['Brand'].unique():
    models_for_brand = df_cleaned[df_cleaned['Brand'] == brand]['Model'].unique().tolist()
    brand_model_map[brand] = sorted(models_for_brand)

# Save unique dropdown options for the web server to pull
categorical_options = {
    'brand_model_map': brand_model_map,
    'fuel': sorted(df_cleaned['fuel'].unique().tolist()),
    'seller_type': sorted(df_cleaned['seller_type'].unique().tolist()),
    'transmission': sorted(df_cleaned['transmission'].unique().tolist()),
    'owner': sorted(df_cleaned['owner'].unique().tolist())
}
joblib.dump(categorical_options, 'dropdown_options.pkl')

# 5. Define Features (X) and Target (y)
# Notice 'Model' is kept here so the pipeline trains on it!
X = df_cleaned[['km_driven', 'Brand', 'Model', 'fuel', 'seller_type', 'transmission', 'owner', 'Car_Age']]
y = df_cleaned['selling_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Build the Preprocessing Matrix Configuration
numeric_features = ['km_driven', 'Car_Age']
categorical_features = ['Brand', 'Model', 'fuel', 'seller_type', 'transmission', 'owner']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# 7. Assemble Ridge Pipeline and train across Grid Search
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge())
])

param_grid = {'regressor__alpha': [0.1, 1.0, 10.0, 100.0]}
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# 8. Export the perfect end-to-end matching pipeline configuration
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'cardekho_ridge_model.pkl')
print("Success: 'train.py' has been fully updated with the 'Model' feature.")
