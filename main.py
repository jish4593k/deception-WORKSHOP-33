import tkinter as tk
from tkinter import Text, Label, Button
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import re

# Load and preprocess your text dataset
# Replace this with your actual dataset loading and preprocessing
# In this example, we'll use some dummy data
data = pd.DataFrame({'text': ["Positive text 1", "Positive text 2", "Negative text 1", "Negative text 2"],
                     'label': [1, 1, 0, 0]})


def preprocess_text(text):
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

data['text'] = data['text'].apply(preprocess_text)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Tokenize the text and convert it to sequences
tokenizer = Tokenizer(num_words=1000, lower=True)
tokenizer.fit_on_texts(X_train)
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Padding sequences
max_sequence_length = 50  # Adjust this based on your dataset and model
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_sequence_length)
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_sequence_length)

# Build and train a Keras LSTM model
model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=32, input_length=max_sequence_length))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_padded, y_train, epochs=10, batch_size=32, validation_data=(X_test_padded, y_test))

# Evaluate the model
y_pred = model.predict(X_test_padded)
y_pred_binary = (y_pred > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred_binary)
classification_rep = classification_report(y_test, y_pred_binary)

# Create a Tkinter GUI for text classification
def classify_text():
    input_text = text_input.get("1.0", "end-1c")
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_padded = pad_sequences(input_sequence, maxlen=max_sequence_length)
    prediction = model.predict(input_padded)
    result_label.config(text=f'Prediction: {"Positive" if prediction > 0.5 else "Negative"}')

root = tk.Tk()
root.title("Text Classification")

text_input_label = Label(root, text="Enter Text:")
text_input_label.pack()
text_input = Text(root, height=5, width=30)
text_input.pack()
classify_button = Button(root, text="Classify", command=classify_text)
classify_button.pack()
result_label = Label(root, text="")
result_label.pack()

root.mainloop()
