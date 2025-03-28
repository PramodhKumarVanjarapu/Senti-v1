# app.py
import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, pipeline  # Use BertTokenizer instead of BertTokenizerFast
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import pickle
from sklearn.decomposition import PCA
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from io import StringIO
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from functools import lru_cache

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
        return None, None, None, None, None, None, None, None
    
    try:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
        bert_model.eval()
    except Exception as e:
        st.error(f"Error loading BERT model: {e}")
        return None, None, None, None, None, None, None, None
    
    try:
        with open('pca_model.pkl', 'rb') as f:
            pca = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading PCA model: {e}")
        return None, None, None, None, None, None, None, None
    
    analyzer = SentimentIntensityAnalyzer()
    sentence_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    absa_classifier = pipeline("text-classification", model="yangheng/deberta-v3-base-absa-v1.1")
    
    st.write("Models loaded successfully!")  # Debug message
    
    return model, tokenizer, bert_model, pca, analyzer, device, sentence_model, absa_classifier

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
        cls_embeddings = preprocess_sentence(str(sentence), tokenizer, bert_model, device)
        cls_embeddings_np = cls_embeddings.cpu().numpy()
        pca_embeddings = pca.transform(cls_embeddings_np)
        vader_score = analyzer.polarity_scores(str(sentence))['compound']
        sentence_features = np.concatenate([[vader_score], pca_embeddings[0]])
        sentence_features_tensor = torch.tensor(sentence_features, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
        with torch.no_grad():
            outputs = model(sentence_features_tensor)
            _, predicted = torch.max(outputs.data, 1)
        sentiment_label = predicted.item()
        sentiment = ['Negative', 'Neutral', 'Positive'][sentiment_label]
        return sentiment
    except Exception as e:
        st.error(f"Prediction error for '{sentence}': {e}")
        return "Error"

# Word-level sentiment analysis
def get_word_sentiments(text, analyzer):
    words = word_tokenize(text.lower())
    word_sentiments = {}
    for word in words:
        score = analyzer.polarity_scores(word)['compound']
        if score != 0:
            word_sentiments[word] = score
    return word_sentiments

# ABSA Functions
def load_language_model(text):
    lang = detect(text)
    model_name = f"{lang}_core_web_sm" if lang != "en" else "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except OSError:
        return spacy.load("en_core_web_sm")

STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "it", "this", "that", "of", "to",
    "for", "on", "with", "as", "by", "at", "in", "and", "but", "or"
}

def predict_aspects(text, previous_aspects=None):
    nlp = load_language_model(text)
    doc = nlp(text)
    aspects = []
    previous_aspects = previous_aspects or []

    for chunk in doc.noun_chunks:
        if "and" in chunk.text.lower() and "also" not in chunk.text.lower():
            sub_chunks = [c.strip() for c in chunk.text.split(" and ")]
            for sub_chunk in sub_chunks:
                sub_doc = nlp(sub_chunk)
                aspect_tokens = [
                    token.text for token in sub_doc
                    if token.text.lower() not in STOPWORDS
                    and token.pos_ in ["NOUN", "PROPN"]
                    and token.dep_ not in ["det", "poss", "prep", "pron"]
                ]
                aspect = " ".join(aspect_tokens).strip()
                if aspect:
                    aspects.append(aspect)
        else:
            aspect_tokens = [
                token.text for token in chunk
                if token.text.lower() not in STOPWORDS
                and token.pos_ in ["NOUN", "PROPN"]
                and token.dep_ not in ["det", "poss", "prep", "pron"]
            ]
            aspect = " ".join(aspect_tokens).strip()
            if aspect:
                aspects.append(aspect)

    if "it" in [token.text.lower() for token in doc] and previous_aspects and not aspects:
        aspects.append(previous_aspects[-1])

    if not aspects:
        for token in doc:
            if (token.text.lower() not in STOPWORDS and
                token.pos_ in ["NOUN", "PROPN"] and
                token.dep_ not in ["det", "poss", "prep", "pron"]):
                aspects.append(token.text)

    aspects = sorted(list(set(aspects)))
    return merge_similar_aspects(aspects)

def merge_similar_aspects(aspects, sentence_model, threshold=0.9):
    if len(aspects) <= 1:
        return aspects

    aspect_vectors = sentence_model.encode(aspects)
    merged_aspects = []
    used_indices = set()

    for i, aspect1 in enumerate(aspects):
        if i in used_indices:
            continue
        merged_aspects.append(aspect1)
        for j, aspect2 in enumerate(aspects):
            if i != j and j not in used_indices:
                similarity = cosine_similarity(
                    [aspect_vectors[i]], [aspect_vectors[j]]
                )[0][0]
                if similarity > threshold:
                    used_indices.add(j)
        used_indices.add(i)

    return merged_aspects

def classify_sentiment_absa(text, aspect_terms, absa_classifier):
    aspect_sentiments = []
    for aspect in aspect_terms:
        input_text = f"[CLS] {text} [SEP] {aspect} [SEP]"
        result = absa_classifier(input_text)
        sentiment = result[0]["label"].lower() if result else "neutral"
        confidence = round(result[0]["score"], 4) if result else 0.0
        aspect_sentiments.append((aspect, sentiment, confidence, text))
    return aspect_sentiments

def split_sentences(text):
    nlp = load_language_model(text)
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    final_sentences = []
    for sent in sentences:
        if " but " in sent:
            parts = sent.split(" but ")
            final_sentences.extend(parts)
        elif ",but " in sent:
            parts = sent.split(",but ")
            final_sentences.extend(parts)
        elif " and also " in sent:
            parts = sent.split(" and also ")
            final_sentences.extend(parts)
        else:
            final_sentences.append(sent)

    return [s.strip() for s in final_sentences if s.strip()]

def get_aspect_sentiments(text, sentence_model, absa_classifier):
    clauses = split_sentences(text)
    aspect_sentiment_dict = {}
    previous_aspects = []

    for clause in clauses:
        aspect_terms = predict_aspects(clause, previous_aspects)
        if aspect_terms:
            sentiments = classify_sentiment_absa(clause, aspect_terms, absa_classifier)
            for aspect, sentiment, confidence, clause_text in sentiments:
                if aspect not in aspect_sentiment_dict:
                    aspect_sentiment_dict[aspect] = []
                aspect_sentiment_dict[aspect].append((sentiment, confidence, clause_text))
            previous_aspects = aspect_terms

    aspect_sentiments = []
    for aspect, sentiment_list in aspect_sentiment_dict.items():
        for sentiment, confidence, clause_text in sentiment_list:
            aspect_sentiments.append((aspect, sentiment, confidence, clause_text))

    return aspect_sentiments

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

def plot_overall_sentiment(sentiments):
    sentiment_counts = pd.Series(sentiments).value_counts()
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title='Overall Sentiment Distribution',
        color=sentiment_counts.index,
        color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue', 'Error': 'gray'}
    )
    return fig

def plot_aspect_sentiments(aspect_sentiments):
    if not aspect_sentiments:
        return None
    df = pd.DataFrame(aspect_sentiments, columns=['Aspect', 'Sentiment', 'Confidence', 'Clause'])
    fig = px.bar(
        df,
        x='Aspect',
        y='Confidence',
        color='Sentiment',
        color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'blue'},
        title='Aspect-Based Sentiment Analysis',
        hover_data=['Clause'],
        text=df['Sentiment']
    )
    fig.update_traces(textposition='auto')
    fig.update_layout(barmode='group')
    return fig

# Streamlit interface
def main():
    st.title("Sentiment Analysis Application")
    
    # Load models
    model, tokenizer, bert_model, pca, analyzer, device, sentence_model, absa_classifier = load_models()
    
    if model is None:
        return

    # Create tabs
    tab1, tab2 = st.tabs(["Single Text Analysis", "CSV File Analysis"])

    # Tab 1: Single Text Analysis
    with tab1:
        st.write("Enter a sentence to analyze its sentiment, word-level insights, and aspect-based sentiments")
        user_input = st.text_area("Enter your text here:", height=150)
        
        if st.button("Analyze Sentiment", key="single_analyze"):
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
                    bar_fig = plot_sentiment_bar(word_sentiments)
                    if bar_fig:
                        st.plotly_chart(bar_fig)
                    else:
                        st.info("No significant word-level sentiments detected for bar chart.")

                    wc_fig = generate_wordcloud(word_sentiments)
                    if wc_fig:
                        st.pyplot(wc_fig)
                    else:
                        st.info("No significant word-level sentiments detected for word cloud.")

                    # Aspect-based sentiment analysis
                    aspect_sentiments = get_aspect_sentiments(user_input, sentence_model, absa_classifier)
                    if aspect_sentiments:
                        st.write("Aspect-Based Sentiments:")
                        for aspect, sentiment, confidence, clause in aspect_sentiments:
                            st.write(f"- **{aspect}**: {sentiment} (Confidence: {confidence}) [from: '{clause}']")
                        aspect_fig = plot_aspect_sentiments(aspect_sentiments)
                        st.plotly_chart(aspect_fig)
                    else:
                        st.info("No aspects detected in the text.")

            else:
                st.warning("Please enter some text to analyze!")

    # Tab 2: CSV File Analysis
    with tab2:
        st.write("Upload a CSV file with reviews to analyze sentiments and aspects")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded file:")
            st.dataframe(df.head())

            review_column = st.selectbox("Select the column containing reviews:", df.columns)
            
            if st.button("Analyze CSV", key="csv_analyze"):
                with st.spinner("Analyzing reviews..."):
                    progress_bar = st.progress(0)
                    total_rows = len(df)
                    sentiments = []
                    all_aspect_sentiments = []

                    # Process each review
                    for i, review in enumerate(df[review_column]):
                        sentiment = predict_sentence(review, model, tokenizer, bert_model, pca, analyzer, device)
                        sentiments.append(sentiment)
                        aspect_sentiments = get_aspect_sentiments(str(review), sentence_model, absa_classifier)
                        all_aspect_sentiments.extend(aspect_sentiments)
                        progress = (i + 1) / total_rows
                        progress_bar.progress(progress)

                    # Add predictions to dataframe (for display)
                    df['Predicted_Sentiment'] = sentiments
                    if all_aspect_sentiments:
                        aspect_df = pd.DataFrame(all_aspect_sentiments, columns=['Aspect', 'Sentiment', 'Confidence', 'Clause'])
                        df = df.join(aspect_df.groupby(df.index // len(df)).agg(lambda x: '; '.join(map(str, x))))

                    # Display results
                    st.write("Analysis complete! Here's the updated dataframe with all details:")
                    st.dataframe(df)

                    # Prepare simplified CSV with only reviewText and Sentiment
                    simplified_df = pd.DataFrame({
                        'reviewText': df[review_column],
                        'Sentiment': df['Predicted_Sentiment']
                    })
                    csv = simplified_df.to_csv(index=False)
                    st.download_button(
                        label="Download results as CSV",
                        data=csv,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv"
                    )

                    # Plot overall sentiment distribution
                    overall_fig = plot_overall_sentiment(df['Predicted_Sentiment'])
                    st.plotly_chart(overall_fig)

                    # Plot aspect sentiments
                    if all_aspect_sentiments:
                        aspect_fig = plot_aspect_sentiments(all_aspect_sentiments)
                        st.plotly_chart(aspect_fig)

if __name__ == "__main__":
    main()
