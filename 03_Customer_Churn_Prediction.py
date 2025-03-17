import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/datablist/sample-csv-files/main/files/customers/customers-100.csv')

# Prepare data
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
X = df[['Age', 'Gender', 'Annual Income ($)']]
y = (df['Age'] > 40).astype(int)  # Simulating churn based on age

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
print("Churn Predictions:", model.predict(X_test[:5]))
