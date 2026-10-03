# Re-trains the Elastic Net churn model and saves it to model/customer_churn_elastic_net.joblib
# (Same pipeline as your notebook. Only needed if the saved model fails to load on your machine.)
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

DATA_PATH = "data/Telco-Customer-Churn.csv"
MODEL_PATH = "model/customer_churn_elastic_net.joblib"
RANDOM_STATE = 42


def train():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.drop_duplicates().reset_index(drop=True)

    y = df["Churn"].map({"No": 0, "Yes": 1})
    X = df.drop(columns=["Churn", "customerID"])

    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            penalty="elasticnet", solver="saga", max_iter=5000,
            class_weight="balanced", random_state=RANDOM_STATE,
        )),
    ])

    param_grid = {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__l1_ratio": [0, 0.25, 0.5, 0.75, 1],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(pipeline, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    pred = best.predict(X_test)
    prob = best.predict_proba(X_test)[:, 1]
    print("Best parameters:", grid.best_params_)
    print("Accuracy :", round(accuracy_score(y_test, pred), 4))
    print("Precision:", round(precision_score(y_test, pred), 4))
    print("Recall   :", round(recall_score(y_test, pred), 4))
    print("F1-score :", round(f1_score(y_test, pred), 4))
    print("ROC-AUC  :", round(roc_auc_score(y_test, prob), 4))

    joblib.dump(best, MODEL_PATH)
    print("Saved:", MODEL_PATH)
    return best


if __name__ == "__main__":
    train()
