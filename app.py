# app.py
import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # Corrected import
import numpy as np
import pickle
from sklearn.decomposition import PCA
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    st.error(f"Error downloading NLTK data: {e}")

# LSTMClassifier class (unchanged)
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, dropout=0.3, bidirectional=True):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * num_directions, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size * num_directions)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size).to(x.device)
        lstm_out, (hidden, _) = self.lstm(x, (h0, c0))
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        out = self.batch_norm(hidden)
        out = self.dropout(out)
        out = self.fc(out)
        return out

# Initialize models and components
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = LSTMClassifier(input_size=257, hidden_size=512, num_classes=3).to(device)
        model.load_state_dict(torch.load('model.pth', map_location=device))
        model.eval()
    except Exception as e:
        st.error(f"Error loading LSTM model: {e}")
        return None, None, None, None, None, None
    
    try:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
        bert_model.eval()
    except Exception as e:
        st.error(f"Error loading BERT model: {e}")
        return None, None, None, None, None, None
    
    try:
        with open('pca_model.pkl', 'rb') as f:
            pca = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading PCA model: {e}")
        return None, None, None, None, None, None
    
    analyzer = SentimentIntensityAnalyzer()
    
    return model, tokenizer, bert_model, pca, analyzer, device

# Preprocessing and prediction functions
def preprocess_sentence(sentence, tokenizer, bert_model, device):
    tokens = tokenizer(
        [sentence],
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = bert_model(**tokens)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
    return cls_embeddings

def predict_sentence(sentence, model, tokenizer, bert_model, pca, analyzer, device):
    try:
        cls_embeddings = preprocess_sentence(sentence, tokenizer, bert_model, device)
        cls_embeddings_np = cls_embeddings.cpu().numpy()
        pca_embeddings = pca.transform(cls_embeddings_np)
        vader_score = analyzer.polarity_scores(sentence)['compound']
        sentence_features = np.concatenate([[vader_score], pca_embeddings[0]])
        sentence_features_tensor = torch.tensor(sentence_features, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
        with torch.no_grad():
            outputs = model(sentence_features_tensor)
            _, predicted = torch.max(outputs.data, 1)
        sentiment_label = predicted.item()
        sentiment = ['Negative', 'Neutral', 'Positive'][sentiment_label]
        return sentiment
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Word-level sentiment analysis
def get_word_sentiments(text, analyzer):
    words = word_tokenize(text.lower())
    word_sentiments = {}
    for word in words:
        score = analyzer.polarity_scores(word)['compound']
        if score != 0:  # Only include words with non-neutral sentiment
            word_sentiments[word] = score
    return word_sentiments

# Visualization functions
def plot_sentiment_bar(word_sentiments):
    if not word_sentiments:
        return None
    words = list(word_sentiments.keys())
    scores = list(word_sentiments.values())
    colors = ['red' if s < 0 else 'green' for s in scores]
    
    fig = px.bar(
        x=words,
        y=scores,
        color=colors,
        color_discrete_map={'red': 'red', 'green': 'green'},
        labels={'x': 'Words', 'y': 'Sentiment Score'},
        title='Word-Level Sentiment Distribution'
    )
    fig.update_layout(showlegend=False)
    return fig

def generate_wordcloud(word_sentiments):
    if not word_sentiments:
        return None
    wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_sentiments)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig

# Streamlit interface
def main():
    st.title("Sentiment Analysis Application")
    st.write("Enter a sentence to analyze its sentiment and see word-level insights")

    # Load models
    model, tokenizer, bert_model, pca, analyzer, device = load_models()
    
    if model is None:
        return

    # User input
    user_input = st.text_area("Enter your text here:", height=150)
    
    if st.button("Analyze Sentiment"):
        if user_input:
            with st.spinner("Analyzing..."):
                # Overall sentiment
                sentiment = predict_sentence(user_input, model, tokenizer, bert_model, pca, analyzer, device)
                if sentiment:
                    if sentiment == "Positive":
                        st.success(f"Overall Sentiment: {sentiment} 😊")
                    elif sentiment == "Negative":
                        st.error(f"Overall Sentiment: {sentiment} 😔")
                    else:
                        st.info(f"Overall Sentiment: {sentiment} 😐")

                    # Word-level analysis
                    word_sentiments = get_word_sentiments(user_input, analyzer)
                    
                    # Bar chart
                    bar_fig = plot_sentiment_bar(word_sentiments)
                    if bar_fig:
                        st.plotly_chart(bar_fig)
                    else:
                        st.info("No significant word-level sentiments detected for bar chart.")

                    # Word cloud
                    wc_fig = generate_wordcloud(word_sentiments)
                    if wc_fig:
                        st.pyplot(wc_fig)
                    else:
                        st.info("No significant word-level sentiments detected for word cloud.")
        else:
            st.warning("Please enter some text to analyze!")

if __name__ == "__main__":
    main()
