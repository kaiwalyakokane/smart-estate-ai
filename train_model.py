import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Loading dataset...")

data = pd.read_csv("train.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

# ==========================================
# 2. SELECT TARGET COLUMN
# ==========================================

target = "Price_INR_Lakhs"

# Remove rows where target price is missing
data = data.dropna(subset=[target])

# ==========================================
# 3. SELECT INPUT FEATURES
# ==========================================

features = [
    "City",
    "Locality_Type",
    "Property_Type",
    "BHK",
    "Bathrooms",
    "Super_Area_SqFt",
    "Carpet_Area_SqFt",
    "Floor_Number",
    "Total_Floors",
    "Age_of_Property",
    "Furnishing_Status",
    "Parking",
    "Lift_Available",
    "Gated_Community",
    "Distance_to_Metro_km",
    "Distance_to_City_Center_km"
]

# Keep only columns available in the dataset
features = [column for column in features if column in data.columns]

X = data[features].copy()
y = pd.to_numeric(data[target], errors="coerce")

# Remove rows with invalid target values
valid_rows = y.notna() & (y >= 0)

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

print("Features used:", features)
print("Usable records:", len(X))

# ==========================================
# 4. IDENTIFY COLUMN TYPES
# ==========================================

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

# ==========================================
# 5. PREPROCESS DATA
# ==========================================

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# ==========================================
# 6. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# ==========================================
# 7. CREATE RANDOM FOREST MODEL
# ==========================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2
    ))
])

# ==========================================
# 8. TRAIN MODEL
# ==========================================

print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

print("Training completed!")

# ==========================================
# 9. EVALUATE MODEL
# ==========================================

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(y_test, predictions)

print("\n========== MODEL PERFORMANCE ==========")
print(f"MAE:  {mae:.2f} lakhs")
print(f"RMSE: {rmse:.2f} lakhs")
print(f"R2 Score: {r2:.4f}")

# ==========================================
# 10. SAVE MODEL AND METADATA
# ==========================================

model_data = {
    "model": model,
    "features": features,
    "cities": sorted(data["City"].dropna().unique().tolist()),
    "property_types": sorted(data["Property_Type"].dropna().unique().tolist()),
    "locality_types": sorted(data["Locality_Type"].dropna().unique().tolist()),
    "furnishing_options": sorted(data["Furnishing_Status"].dropna().unique().tolist()),
    "metrics": {
        "mae_lakhs": float(mae),
        "rmse_lakhs": float(rmse),
        "r2": float(r2)
    }
}

with open("house_model.pkl", "wb") as file:
    pickle.dump(model_data, file)

print("\nModel saved successfully as house_model.pkl")
print("Available cities:", model_data["cities"])