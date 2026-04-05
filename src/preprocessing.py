import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


def load_data(path):
    """Load a CSV dataset into a pandas DataFrame."""
    df = pd.read_csv(path)
    return df


def preprocess(df, fit_scaler=True):
    df = df.copy()

    # Convert the target column to numeric labels.
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Remove the customer ID column because it is not predictive.
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # Convert TotalCharges to numeric, coercing invalid entries to NaN.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill missing TotalCharges values with the median value.
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Create a new average ticket feature for monthly revenue per tenure month.
    df['avg_ticket'] = df['TotalCharges'] / (df['tenure'] + 1)

    # Create discrete tenure groups for categorical splitting.
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 60, 100],
        labels=['0-1yr', '1-2yr', '2-4yr', '4-5yr', '5+yr']
    )

    # Convert categorical columns to one-hot encoded features.
    df = pd.get_dummies(df, drop_first=True)

    # Split features and label.
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    scaler = StandardScaler()

    if fit_scaler:
        # Fit a new scaler on the training data and save it.
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, 'src/scaler.pkl')
    else:
        # Load the saved scaler and apply it to incoming data.
        scaler = joblib.load('src/scaler.pkl')
        X_scaled = scaler.transform(X)

    return X_scaled, y