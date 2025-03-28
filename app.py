import torch
import torch.nn as nn
import streamlit as st
from transformers import BertTokenizer, BertModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import pickle
import plotly.express as px
import pandas as pd
import os
import gc

# Define the LSTM Classifier
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Use the last hidden state
        out = self.softmax(out)  # Output probabilities for each class
        return out

# Load models function
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load LSTM model
    if not os.path.exists('model.pth'):
        st.error("model.pth not found. Please upload the LSTM model file to your Streamlit Cloud repository.")
        return None, None, None, None, None, None
    try:
        model = LSTMClassifier(input_size=257, hidden_size=512, num_classes=3).to(device)
        model.load_state_dict(torch.load('model.pth', map_location=device))
        model.eval()
    except Exception as e:
        st.error(f"Error loading LSTM model: {e}")
        return None, None, None, None, None, None
    
    # Load BERT tokenizer and model
    try:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
        bert_model.eval()
    except Exception as e:
        st.error(f"Error loading BERT model: {e}")
        return None, None, None, None, None, None
    
    # Load PCA model
    if not os.path.exists('pca_model.pkl'):
        st.error("pca_model.pkl not found. Please upload the PCA model file to your Streamlit Cloud repository.")
        return None, None, None, None, None, None
    try:
        with open('pca_model.pkl', 'rb') as f:
            pca = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading PCA model: {e}")
        return None, None, None, None, None, None
    
    # Sentiment analyzer (VADER)
    analyzer = SentimentIntensityAnalyzer()
    
    return model, tokenizer, bert_model, pca, analyzer, device

# Function for sentiment classification
def predict_sentiment(model, tokenizer, bert_model, pca, analyzer, device, text):
    # Tokenize and process text for BERT
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        sentence_embedding = outputs.last_hidden_state.mean(dim=1)  # Take the mean of the hidden states

    # Apply PCA transformation on BERT sentence embedding
    sentence_embedding_pca = pca.transform(sentence_embedding.cpu().numpy())

    # Convert PCA embedding to tensor
    sentence_embedding_tensor = torch.tensor(sentence_embedding_pca).to(device)

    # Get sentiment prediction from LSTM model
    with torch.no_grad():
        prediction = model(sentence_embedding_tensor)
    sentiment = prediction.argmax(dim=1).item()

    # Use VADER for additional sentiment analysis (optional)
    vader_score = analyzer.polarity_scores(text)
    return sentiment, vader_score

# Streamlit UI
def main():
    st.title("Sentiment Analysis with LSTM and BERT")
    
    # Load models
    model, tokenizer, bert_model, pca, analyzer, device = load_models()
    if model is None:
        return
    
    # Text input for sentiment analysis
    user_input = st.text_area("Enter text for sentiment analysis:")
    
    if st.button("Analyze Sentiment"):
        if user_input:
            sentiment, vader_score = predict_sentiment(model, tokenizer, bert_model, pca, analyzer, device, user_input)
            
            # Display sentiment result
            if sentiment == 0:
                sentiment_label = "Negative"
            elif sentiment == 1:
                sentiment_label = "Neutral"
            else:
                sentiment_label = "Positive"
            
            st.write(f"Sentiment: {sentiment_label}")
            st.write(f"VADER Sentiment Score: {vader_score}")
            
            # Optionally plot sentiment analysis result
            sentiment_data = {'Sentiment': ['Negative', 'Neutral', 'Positive'], 'Score': [vader_score['neg'], vader_score['neu'], vader_score['pos']]}
            df = pd.DataFrame(sentiment_data)
            fig = px.bar(df, x='Sentiment', y='Score', title='Sentiment Analysis')
            st.plotly_chart(fig)
        else:
            st.error("Please enter some text.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
