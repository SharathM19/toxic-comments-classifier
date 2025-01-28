from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        comment_text = request.form['comment']
        
        # Transform the comment text using the vectorizer
        comment_vector = vectorizer.transform([comment_text])
        
        # Predict the probabilities
        prediction = model.predict(comment_vector)
        
        # Format the output as specified
        result = {
            "id": 1,  # Assuming a dummy ID, replace with actual ID if available
            "comment_text": comment_text,
            "toxic": int(prediction[0][0]),
            "severe_toxic": int(prediction[0][1]),
            "obscene": int(prediction[0][2]),
            "threat": int(prediction[0][3]),
            "insult": int(prediction[0][4]),
            "identity_hate": int(prediction[0][5])
        }

        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)


