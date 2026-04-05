from sklearn.metrics import classification_report, roc_auc_score
import joblib

from preprocessing import load_data, preprocess


def evaluate():
    """Evaluate the trained churn model using the full dataset."""

    # Load the dataset used for evaluation.
    df = load_data('data/churn.csv')

    # Preprocess the data using the saved scaler without refitting.
    X, y = preprocess(df, fit_scaler=False)

    # Load the trained model from disk.
    model = joblib.load('src/model.pkl')

    # Generate class predictions and probability estimates.
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Print evaluation metrics to the console.
    print("Classification Report:")
    print(classification_report(y, y_pred))
    print("ROC-AUC:", roc_auc_score(y, y_prob))


if __name__ == "__main__":
    evaluate()