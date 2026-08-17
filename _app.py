from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from _model import SentimentLSTM
from _data_prep import word_to_index, MAX_LEN

app = Flask(__name__)
CORS(app)

# 1. Initialize & Load Trained PyTorch Model
VOCAB_SIZE = len(word_to_index)
EMBEDDING_DIM = 16
HIDDEN_DIM = 32
OUTPUT_DIM = 1

model = SentimentLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
model.load_state_dict(torch.load('sentiment_model.pth'))
model.eval()

# 2. Text Processing Pipeline
def process_input_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    clean = [word for word in tokens if word not in string.punctuation and word not in stop_words]
    
    indexed = [word_to_index.get(word, 1) for word in clean]  # 1 = <UNK>
    
    if len(indexed) < MAX_LEN:
        indexed = indexed + [0] * (MAX_LEN - len(indexed))    # 0 = <PAD>
    else:
        indexed = indexed[:MAX_LEN]
        
    return torch.tensor([indexed], dtype=torch.long)

# 3. API Routes
@app.route('/')
def home():
    return render_template('sentiment_dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review = data.get('text', '')

    input_tensor = process_input_text(review)

    with torch.no_grad():
        prediction = model(input_tensor)
        prob = prediction.item()  # 1.0 = Max Pos, 0.0 = Max Neg

    print(f"\n--- INFERENCE RAW PROBABILITY: {prob:.4f} ---")

    if prob >= 0.5:
        sentiment = "Positive"
        confidence_val = prob * 100
    else:
        sentiment = "Negative"
        confidence_val = (1.0 - prob) * 100

    return jsonify({
        'sentiment': sentiment,
        'score': prob,
        'confidence': round(confidence_val, 1)
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)