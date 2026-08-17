import torch
import torch.nn as nn
import torch.optim as optim
from _model import SentimentLSTM
from _data_prep import df, word_to_index

# Model Parameters
VOCAB_SIZE = len(word_to_index)
EMBEDDING_DIM = 16
HIDDEN_DIM = 32
OUTPUT_DIM = 1
LEARNING_RATE = 0.01
EPOCHS = 150

model = SentimentLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("--- TRAINING PYTORCH LSTM ---")
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for _, row in df.iterrows():
        input_tensor = torch.tensor([row['padded_sequence']], dtype=torch.long)
        target_tensor = torch.tensor([[float(row['label'])]], dtype=torch.float)

        optimizer.zero_grad()
        prediction = model(input_tensor)
        loss = criterion(prediction, target_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 30 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# Save trained weights
torch.save(model.state_dict(), 'sentiment_model.pth')
print("✅ MODEL TRAINED AND SAVED AS sentiment_model.pth!")
import json
with open('word_to_index.json', 'w') as f:
    json.dump(word_to_index, f)