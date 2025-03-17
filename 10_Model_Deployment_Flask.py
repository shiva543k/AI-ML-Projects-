Here’s the Model Deployment using Flask script. This project demonstrates how to deploy a trained Machine Learning model as a REST API using Flask. The model predicts whether an email is spam or not using Naïve Bayes.


---

🚀 Steps to Run:

1️⃣ Install dependencies:

pip install flask pandas scikit-learn nltk

2️⃣ Save the script as app.py
3️⃣ Run the Flask app:

python app.py

4️⃣ Test the API in your browser or Postman:

Open http://127.0.0.1:5000/predict?text=Congratulations! You won a lottery!

It should return: {"prediction": "Spam"}



---

📌 Model Deployment with Flask Script

from flask import Flask, request, jsonify
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# Download stopwords
nltk.download('stopwords')

# Load pre-trained model
with open("spam_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load vectorizer
with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    text = request.args.get('text')
    if not text:
        return jsonify({"error": "No input text provided"}), 400

    # Transform input text
    text_vectorized = vectorizer.transform([text])

    # Predict using model
    prediction = model.predict(text_vectorized)[0]
    
    return jsonify({"prediction": "Spam" if prediction else "Ham"})

if __name__ == "__main__":
    app.run(debug=True)




🚀 Enhancements:

Use Docker to containerize the Flask app.

Deploy the model to AWS, Google Cloud, or Heroku.

Convert this into a full-stack app with a React/HTML frontend.


Let me know if you need help running this! 🚀

