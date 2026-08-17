import string
import pandas as pd
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# 1. Dataset with 50 distinct reviews for proper training convergence
raw_data = {
    'review': [
        "I absolutely love this product it is amazing", "Best item I have ever bought fantastic quality",
        "Great experience works perfectly and fast", "Super happy with this purchase highly recommend",
        "Excellent product excellent service love it", "Wonderful item exceeded my expectations awesome",
        "Really good quality fast shipping love it", "Perfect choice works great and looks nice",
        "Five stars incredible product super useful", "Awesome experience would definitely buy again",
        "Loved the item great value for money", "Brilliant product works like a charm",
        "Extremely happy best quality product ever", "Top quality item super satisfied with it",
        "Great quality item very durable and nice", "Fantastic product came super fast loved it",
        "Amazing build quality best item on amazon", "Very good product happy customer here",
        "Love love love this product works great", "Perfect condition super fast shipping awesome",
        "Great deal highly recommend this awesome item", "Loved using this works wonderfully well",
        "Very satisfied good quality item loved it", "Incredible performance works awesome and fast",
        "Best purchase ever extremely happy with this",
        
        "Terrible experience item arrived broken and useless", "Worst product ever total waste of money",
        "Horrible quality broke on the first day", "Very disappointed shipping was super slow and bad",
        "Trash item do not buy total scam", "Bad experience item was damaged and broken",
        "Awful quality stopped working immediately", "Defective product terrible customer service",
        "Waste of money item felt super cheap", "Hated this purchase completely broken on arrival",
        "Poor quality item broke right away terrible", "Extremely bad product waste of time and money",
        "Worst experience ever item never worked", "Cheap quality completely useless product",
        "Super disappointed item was broken into pieces", "Unusable item arrived late and damaged",
        "Horrible service item broke in two minutes", "Total garbage do not waste your money",
        "Very poor performance terrible quality item", "Broken product useless item waste of cash",
        "Awful experience item stopped working fast", "Bad quality item arrived cracked and damaged",
        "Disappointed with this product terrible build", "Scam item completely broken waste of money",
        "Hated it worst product ever bought"
    ],
    'label': [1]*25 + [0]*25  # 1 = Positive, 0 = Negative
}

df = pd.DataFrame(raw_data)

# 2. Text Preprocessing
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    clean = [word for word in tokens if word not in string.punctuation and word not in stop_words]
    return clean

df['cleaned_tokens'] = df['review'].apply(preprocess_text)

# 3. Build Vocabulary (0 = <PAD>, 1 = <UNK>)
vocab = sorted(set(word for tokens in df['cleaned_tokens'] for word in tokens))
word_to_index = {word: idx + 2 for idx, word in enumerate(vocab)}
word_to_index['<PAD>'] = 0
word_to_index['<UNK>'] = 1

# 4. Numericalize and Pad Sequences to Fixed Length (10)
MAX_LEN = 10

def encode_and_pad(tokens):
    indexed = [word_to_index.get(word, 1) for word in tokens]
    if len(indexed) < MAX_LEN:
        indexed = indexed + [0] * (MAX_LEN - len(indexed))
    else:
        indexed = indexed[:MAX_LEN]
    return indexed

df['padded_sequence'] = df['cleaned_tokens'].apply(encode_and_pad)

print("✅ Data prep complete! Dataset size:", len(df))