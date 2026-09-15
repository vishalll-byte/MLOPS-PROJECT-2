# Titanic Survival Prediction
# Decision Tree Classification + Hyperparameter Tuning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# 1. LOAD DATASET

DATA_PATH = "titanic_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("\n" + "=" * 60)
print("TITANIC SURVIVAL PREDICTION")
print("=" * 60)

# 2. BASIC DATA EXPLORATION

print("\n FIRST 5 ROWS")
print(df.head())

print("\n LAST 5 ROWS")
print(df.tail())

print("\nDATASET SHAPE")
print(df.shape)

print("\nDATASET INFORMATION")
df.info()

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

print("\n========== SURVIVAL DISTRIBUTION ==========")
print(df["Survived"].value_counts())

# 3. DATA PREPROCESSING

# Handle Gender/Sex column
# Some Titanic datasets use "Sex", while others use "Gender".

if "Gender" in df.columns:
    gender_column = "Gender"

elif "Sex" in df.columns:
    df["Gender"] = df["Sex"]
    gender_column = "Gender"

else:
    raise ValueError(
        "Dataset must contain either a 'Gender' or 'Sex' column."
    )


# Convert Gender into numerical values
df["Gender"] = df[gender_column].map({
    "male": 0,
    "female": 1
})


# Check if any gender values could not be mapped
if df["Gender"].isnull().any():
    print("\nWarning: Some Gender values could not be encoded.")


# Handle missing Embarked values
if df["Embarked"].isnull().any():
    df["Embarked"] = df["Embarked"].fillna(
        df["Embarked"].mode()[0]
    )


# Convert Embarked into numerical values
df["Embarked"] = df["Embarked"].map({
    "S": 0,
    "C": 1,
    "Q": 2
})


# Check remaining missing values
print("\n========== MISSING VALUES AFTER PREPROCESSING ==========")
print(df.isnull().sum())

# 4. FEATURE SELECTION

features = [
    "Pclass",
    "Gender",
    "SibSp",
    "Parch",
    "Embarked"
]

X = df[features]
y = df["Survived"]


print("\n========== FEATURES ==========")
print(features)

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

# 5. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n========== TRAIN-TEST SPLIT ==========")
print("Training features:", X_train.shape)
print("Testing features :", X_test.shape)
print("Training target  :", y_train.shape)
print("Testing target   :", y_test.shape)

# 6. DEFAULT DECISION TREE MODEL

default_model = DecisionTreeClassifier(
    random_state=42
)

default_model.fit(X_train, y_train)

# 7. DEFAULT MODEL PREDICTION

y_pred_default = default_model.predict(X_test)

print("\n========== DEFAULT MODEL PREDICTIONS ==========")
print(y_pred_default)

# 8. DEFAULT MODEL EVALUATION

default_accuracy = accuracy_score(
    y_test,
    y_pred_default
)

default_confusion = confusion_matrix(
    y_test,
    y_pred_default
)

print("\n========== DEFAULT MODEL RESULTS ==========")
print("Accuracy:", default_accuracy)

print("\nConfusion Matrix:")
print(default_confusion)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_default
    )
)

# 9. HYPERPARAMETER TUNING
grid_param = {

    "criterion": [
        "gini",
        "entropy"
    ],

    "max_depth": [
        3,
        5,
        7,
        None
    ],

    "min_samples_split": [
        2,
        5,
        10
    ],

    "min_samples_leaf": [
        1,
        3,
        5
    ]
}


print("\n" + "=" * 60)
print("GRID SEARCH HYPERPARAMETER TUNING")
print("=" * 60)

print("\nParameters being tested:")
print(grid_param)

# 10. GRID SEARCH CV
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(
        random_state=42
    ),
    param_grid=grid_param,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)


grid_search.fit(X_train, y_train)

# 11. BEST PARAMETERS
print("\n========== BEST PARAMETERS ==========")
print(grid_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(grid_search.best_score_)

# 12. BEST/TUNED MODEL

tuned_model = grid_search.best_estimator_

print("\n========== TUNED MODEL ==========")
print(tuned_model)

# 13. TUNED MODEL PREDICTION
y_pred_tuned = tuned_model.predict(X_test)

print("\n========== TUNED MODEL PREDICTIONS ==========")
print(y_pred_tuned)

# 14. TUNED MODEL EVALUATION
tuned_accuracy = accuracy_score(
    y_test,
    y_pred_tuned
)

tuned_confusion = confusion_matrix(
    y_test,
    y_pred_tuned
)

print("\n========== TUNED MODEL RESULTS ==========")

print("Tuned Accuracy:", tuned_accuracy)

print("\nTuned Confusion Matrix:")
print(tuned_confusion)

print("\nTuned Classification Report:")
print(
    classification_report(
        y_test,
        y_pred_tuned
    )
)

# 15. MODEL COMPARISON
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    f"Default Decision Tree Accuracy : "
    f"{default_accuracy:.4f}"
)

print(
    f"Tuned Decision Tree Accuracy   : "
    f"{tuned_accuracy:.4f}"
)

improvement = tuned_accuracy - default_accuracy

print(
    f"Accuracy Improvement           : "
    f"{improvement:.4f}"
)

# 16. CONFUSION MATRIX VISUALIZATION

plt.figure(figsize=(6, 5))

plt.imshow(
    tuned_confusion,
    interpolation="nearest"
)

plt.title("Tuned Decision Tree - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    [0, 1],
    ["Did Not Survive", "Survived"]
)

plt.yticks(
    [0, 1],
    ["Did Not Survive", "Survived"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            tuned_confusion[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()
plt.tight_layout()
plt.show()

# 17. DECISION TREE VISUALIZATION
plt.figure(figsize=(18, 10))

plot_tree(
    tuned_model,
    feature_names=features,
    class_names=[
        "Did Not Survive",
        "Survived"
    ],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Tuned Decision Tree")
plt.tight_layout()
plt.show()

# 18. FEATURE IMPORTANCE
feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": tuned_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(feature_importance)

# 19. FEATURE IMPORTANCE VISUALIZATION

plt.figure(figsize=(8, 5))

plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Decision Tree Feature Importance")

plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# 20. FINAL SUMMARY
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"Dataset Size          : {df.shape}")
print(f"Number of Features    : {len(features)}")
print(f"Default Accuracy      : {default_accuracy:.4f}")
print(f"Tuned Accuracy        : {tuned_accuracy:.4f}")
print(f"Best Parameters       : {grid_search.best_params_}")
print(f"Best CV Accuracy      : {grid_search.best_score_:.4f}")

print("\nProject execution completed successfully!")