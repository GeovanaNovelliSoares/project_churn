from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import joblib

from preprocessing import load_data, preprocess


def train():
    # Load the churn dataset from disk.
    df = load_data('data/churn.csv')

    # Preprocess the dataset and fit a scaler.
    X, y = preprocess(df, fit_scaler=True)

    # Split the data into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    # Train and evaluate Logistic Regression.
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)
    log_auc = roc_auc_score(y_test, log_model.predict_proba(X_test)[:, 1])
    results['logistic'] = (log_model, log_auc)

    # Train and evaluate Random Forest.
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
    results['random_forest'] = (rf_model, rf_auc)

    # Train and evaluate XGBoost.
    xgb_model = XGBClassifier(eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
    results['xgboost'] = (xgb_model, xgb_auc)

    # Select the model with the highest ROC-AUC score.
    best_model_name = max(results, key=lambda x: results[x][1])
    best_model, best_score = results[best_model_name]

    print(f"Best model: {best_model_name} | ROC-AUC: {best_score:.4f}")

    # Save the best model to disk for later use.
    joblib.dump(best_model, 'src/model.pkl')


if __name__ == "__main__":
    train()