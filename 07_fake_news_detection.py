import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')

# Load dataset
url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
df = pd.read_csv(url)[['label', 'tweet']]  # Use only label and tweet columns
df.rename(columns={'tweet': 'text'}, inplace=True)  # Rename for consistency

# Convert labels (0 = Real, 1 = Fake)
df['label'] = df['label'].map({0: 0, 1: 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Create a text processing & classification pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer(stop_words=stopwords.words('english'))),
    ('tfidf', TfidfTransformer()),
    ('classifier', LogisticRegression())  # Logistic Regression classifier
])

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Test on new articles
new_articles = [
    "Breaking: Scientists discover a new way to reverse climate change!",
    "Shocking! Government is hiding aliens in Area 51!"
]
predictions = model.predict(new_articles)
for article, label in zip(new_articles, predictions):
    print(f"Article: {article} → {'Fake News' if label else 'Real News'}")
