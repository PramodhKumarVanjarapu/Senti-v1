import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, pipeline
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
import os
import gc

# Modify NLTK download to avoid potential issues
def safe_nltk_download():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
    except Exception as e:
        st.warning(f"Error downloading NLTK data: {e}")

# Attempt to download at script start
safe_nltk_download()

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

# Stopwords for aspect extraction
STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "it", "this", "that", "of", "to",
    "for", "on", "with", "as", "by", "at", "in", "and", "but", "or"
}

def load_language_model(text):
    try:
        lang = detect(text)
        model_name = f"{lang}_core_web_sm" if lang != "en" else "en_core_web_sm"
        return spacy.load(model_name)
    except Exception:
        return spacy.load("en_core_web_sm")

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if required model files exist
    required_files = ['model.pth', 'pca_model.pkl']
    for file in required_files:
        if not os.path.exists(file):
            st.error(f"{file} not found. Please ensure all model files are in the correct directory.")
            return None, None, None, None, None, None, None, None
    
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
    
    # Simplified model loading with fallback
    analyzer = SentimentIntensityAnalyzer()
    sentence_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    
    # Remove the ABSA classifier and replace with a simpler fallback
    absa_classifier = None
    try:
        absa_classifier = pipeline("text-classification", model="yangheng/deberta-v3-base-absa-v1.1")
    except Exception as e:
        st.warning(f"Could not load ABSA classifier: {e}. Aspect-based sentiment analysis will be limited.")
    
    return model, tokenizer, bert_model, pca, analyzer, device, sentence_model, absa_classifier

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
        # Add input validation
        if not sentence or not isinstance(sentence, str):
            return "Error"
        
        # Existing prediction logic
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
        
        # Clean up resources
        torch.cuda.empty_cache()
        gc.collect()
        
        return sentiment
    except Exception as e:
        st.error(f"Prediction error for '{sentence}': {e}")
        return "Error"

def get_word_sentiments(text, analyzer):
    try:
        words = word_tokenize(text.lower())
        word_sentiments = {}
        for word in words:
            score = analyzer.polarity_scores(word)['compound']
            if score != 0:
                word_sentiments[word] = score
        return word_sentiments
    except Exception as e:
        st.warning(f"Error in word sentiment analysis: {e}")
        return {}

def predict_aspects(text, previous_aspects=None, nlp=None):
    try:
        # Use provided NLP model or try to load one
        if nlp is None:
            nlp = load_language_model(text)
        
        doc = nlp(text)
    except Exception as e:
        st.warning(f"Error processing text: {e}")
        return previous_aspects or []
    
    aspects = []
    previous_aspects = previous_aspects or []
    
    try:
        for chunk in doc.noun_chunks:
            aspect_tokens = [
                token.text for token in chunk
                if token.text.lower() not in STOPWORDS
                and token.pos_ in ["NOUN", "PROPN"]
                and token.dep_ not in ["det", "poss", "prep", "pron"]
            ]
            aspect = " ".join(aspect_tokens).strip()
            if aspect:
                aspects.append(aspect)
        
        # Fallback for single tokens if no chunks found
        if not aspects:
            for token in doc:
                if (token.text.lower() not in STOPWORDS and
                    token.pos_ in ["NOUN", "PROPN"] and
                    token.dep_ not in ["det", "poss", "prep", "pron"]):
                    aspects.append(token.text)
        
        # Add fallback if no aspects found and previous aspects exist
        if "it" in [token.text.lower() for token in doc] and previous_aspects and not aspects:
            aspects.append(previous_aspects[-1])
        
        aspects = sorted(list(set(aspects)))
        return merge_similar_aspects(aspects)
    except Exception as e:
        st.warning(f"Error extracting aspects: {e}")
        return previous_aspects or []

def merge_similar_aspects(aspects, sentence_model=None, threshold=0.9):
    if len(aspects) <= 1:
        return aspects
    
    # Add a default sentence model if not provided
    if sentence_model is None:
        sentence_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    
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
    if absa_classifier is None:
        # Fallback method if no ABSA classifier is available
        return [(aspect, "neutral", 0.5, text) for aspect in aspect_terms]
    
    aspect_sentiments = []
    for aspect in aspect_terms:
        try:
            input_text = f"{text}. Aspect: {aspect}"
            result = absa_classifier(input_text)
            sentiment = result[0]["label"].lower() if result else "neutral"
            confidence = round(result[0]["score"], 4) if result else 0.0
            aspect_sentiments.append((aspect, sentiment, confidence, text))
        except Exception as e:
            st.warning(f"Error classifying aspect '{aspect}': {e}")
            aspect_sentiments.append((aspect, "neutral", 0.5, text))
    
    return aspect_sentiments

def split_sentences(text):
    try:
        nlp = load_language_model(text)
        doc = nlp(text)
    except Exception as e:
        st.warning(f"Error loading language model: {e}")
        return [text]
    
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

def main():
    st.title("Sentiment Analysis Application")
    
    # Load models with error handling
    models = load_models()
    if models is None or models[0] is None:
        st.error("Failed to load models. Please check your model files and dependencies.")
        return
    
    model, tokenizer, bert_model, pca, analyzer, device, sentence_model, absa_classifier = models
    
    tab1, tab2 = st.tabs(["Single Text Analysis", "CSV File Analysis"])
    
    with tab1:
        user_input = st.text_area("Enter your text here:", height=150)
        if st.button("Analyze Sentiment", key="single_analyze"):
            if user_input:
                try:
                    # Overall Sentiment
                    sentiment = predict_sentence(user_input, model, tokenizer, bert_model, pca, analyzer, device)
                    if sentiment == "Positive":
                        st.success(f"Overall Sentiment: {sentiment} 😊")
                    elif sentiment == "Negative":
                        st.error(f"Overall Sentiment: {sentiment} 😔")
                    else:
                        st.info(f"Overall Sentiment: {sentiment} 😐")
                    
                    # Word-level Sentiment
                    word_sentiments = get_word_sentiments(user_input, analyzer)
                    if word_sentiments:
                        st.subheader("Word-Level Sentiment")
                        word_bar_fig = plot_sentiment_bar(word_sentiments)
                        if word_bar_fig:
                            st.plotly_chart(word_bar_fig)
                        
                        # Word Cloud
                        word_cloud_fig = generate_wordcloud(word_sentiments)
                        if word_cloud_fig:
                            st.pyplot(word_cloud_fig)
                    
                    # Aspect-based Sentiment
                    aspect_sentiments = get_aspect_sentiments(user_input, sentence_model, absa_classifier)
                    if aspect_sentiments:
                        st.subheader("Aspect-Based Sentiment")
                        aspect_fig = plot_aspect_sentiments(aspect_sentiments)
                        if aspect_fig:
                            st.plotly_chart(aspect_fig)
                    
                except Exception as e:
                    st.error(f"Error analyzing sentiment: {e}")
    
    with tab2:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            review_column = st.selectbox("Select the column containing reviews:", df.columns)
            
            if st.button("Analyze CSV", key="csv_analyze"):
                sentiments = []
                all_aspect_sentiments = []
                
                # Progress bar
                progress_bar = st.progress(0)
                
                for i, review in enumerate(df[review_column]):
                    # Update progress bar
                    progress_bar.progress((i + 1) / len(df))
                    
                    sentiment = predict_sentence(review, model, tokenizer, bert_model, pca, analyzer, device)
                    sentiments.append(sentiment)
                    
                    aspect_sentiments = get_aspect_sentiments(str(review), sentence_model, absa_classifier)
                    all_aspect_sentiments.extend(aspect_sentiments)
                
                # Clear progress bar
                progress_bar.empty()
                
                # Add sentiments to DataFrame
                df['Predicted_Sentiment'] = sentiments
                
                # If aspect sentiments exist, add to DataFrame
                if all_aspect_sentiments:
                    aspect_df = pd.DataFrame(all_aspect_sentiments, columns=['Aspect', 'Sentiment', 'Confidence', 'Clause'])
                    df = pd.concat([df, aspect_df], axis=1)
                
                # Prepare CSV for download
                csv = df.to_csv(index=False, encoding='utf-8')
                st.download_button(
                    label="Download results as CSV",
                    data=csv,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv"
                )
                
                # Overall Sentiment Distribution
                overall_fig = plot_overall_sentiment(df['Predicted_Sentiment'])
                st.plotly_chart(overall_fig)
                
                # Aspect Sentiment Plot (if applicable)
                if all_aspect_sentiments:
                    aspect_fig = plot_aspect_sentiments(all_aspect_sentiments)
                    st.plotly_chart(aspect_fig)

if __name__ == "__main__":
    main()
