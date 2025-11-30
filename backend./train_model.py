import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib
import numpy as np

print(" Starting training script...")

# 1. Load dataset
print(" Loading dataset stroke_data.csv ...")
df = pd.read_csv(r"C:\Users\raksh\Downloads\archive\stroke_data.csv")

# Drop id column if present
if "id" in df.columns:
    df = df.drop(columns=["id"])

# 2. Handle missing BMI values
df["bmi"] = df["bmi"].fillna(df["bmi"].median())

# 3. Features and target
X = df.drop(columns=["stroke"])
y = df["stroke"]

# 4. Column types
categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
numeric_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]

# 5. Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# Class imbalance handling
classes = np.unique(y)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
class_weight_dict = {cls: w for cls, w in zip(classes, class_weights)}
print(" Class weights:", class_weight_dict)

# 6. Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight=class_weight_dict
)

# 7. Pipeline
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# 8. Train-test split
print(" Splitting data into train and test ...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 9. Train model
print(" Training model ...")
clf.fit(X_train, y_train)

# 10. Evaluate
print(" Evaluation on test set:")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# 11. Save model
joblib.dump(clf, "stroke_model.joblib")
print(" Model saved as stroke_model.joblib")