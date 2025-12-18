"""
TOKOPEDIA SENTIMENT ANALYSIS - PREMIUM STREAMLIT APP
100% Sesuai dengan Notebook AnalisisREV.ipynb

Dataset: 36,383 reviews
Models: Logistic Regression, Multinomial Naive Bayes, Random Forest
Sentimen: IndoBERT (bukan rating)
Features: TF-IDF (10,000) + Lexicon (4)

Expected Results:
- Logistic Regression: Acc=0.7493, F1=0.5653
- Multinomial NB: Acc=0.8405, F1=0.5170
- Random Forest: Acc=0.7988, F1=0.5238
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score
)
from collections import Counter
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Tokopedia Sentiment Analysis",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - DARK THEME
# ============================================================================
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

/* Dark Background */
.main { background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%); padding: 2rem; }
.stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%); }

/* Sidebar Dark */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1e293b 0%, #334155 100%); }
[data-testid="stSidebar"] * { color: white !important; }

/* All Text White */
h1, h2, h3, h4, h5, h6, p, span, div, label { color: white !important; }

/* Metrics */
[data-testid="stMetricValue"] { color: white !important; font-size: 2rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: rgba(255, 255, 255, 0.7) !important; font-weight: 600 !important; }

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    color: white !important;
    border: none;
    padding: 0.75rem 2rem;
    border-radius: 10px;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: rgba(30, 41, 59, 0.8);
    border-radius: 12px;
    padding: 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    color: rgba(255, 255, 255, 0.7) !important;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    color: white !important;
}

/* Text Input */
.stTextInput>div>div>input, .stTextArea>div>div>textarea {
    background-color: rgba(30, 41, 59, 0.5) !important;
    color: white !important;
    border: 2px solid rgba(255, 255, 255, 0.2);
    border-radius: 10px;
}
.stTextInput label, .stTextArea label { color: white !important; font-weight: 600 !important; }

/* Selectbox */
.stSelectbox > div > div {
    background-color: rgba(30, 41, 59, 0.5) !important;
    border: 2px solid rgba(255, 255, 255, 0.2);
}
.stSelectbox label { color: white !important; font-weight: 600 !important; }

/* Multiselect */
.stMultiSelect label { color: white !important; font-weight: 600 !important; }
.stMultiSelect > div > div {
    background-color: rgba(30, 41, 59, 0.5) !important;
    border: 2px solid rgba(255, 255, 255, 0.2);
}

/* Slider, Number Input, Checkbox */
.stSlider label, .stNumberInput label, .stCheckbox label {
    color: white !important;
    font-weight: 600 !important;
}
.stNumberInput input {
    background-color: rgba(30, 41, 59, 0.5) !important;
    color: white !important;
    border: 2px solid rgba(255, 255, 255, 0.2);
}

/* Progress Bar */
.stProgress > div > div > div {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
}

/* Alert Messages */
.stSuccess {
    background-color: rgba(16, 185, 129, 0.2) !important;
    color: #10b981 !important;
    border: 1px solid #10b981;
}
.stSuccess * { color: #10b981 !important; }

.stError {
    background-color: rgba(239, 68, 68, 0.2) !important;
    color: #ef4444 !important;
    border: 1px solid #ef4444;
}
.stError * { color: #ef4444 !important; }

.stWarning {
    background-color: rgba(251, 191, 36, 0.2) !important;
    color: #fbbf24 !important;
    border: 1px solid #fbbf24;
}
.stWarning * { color: #fbbf24 !important; }

.stInfo {
    background-color: rgba(59, 130, 246, 0.2) !important;
    color: #3b82f6 !important;
    border: 1px solid #3b82f6;
}
.stInfo * { color: #3b82f6 !important; }

/* Expander */
.streamlit-expanderHeader {
    background-color: rgba(30, 41, 59, 0.8) !important;
    color: white !important;
}

/* Markdown */
.stMarkdown { color: white !important; }

/* Main Header Custom */
.main-header {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    padding: 40px;
    border-radius: 20px;
    color: white !important;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}

.main-header h1 {
    font-size: 3em !important;
    font-weight: 800 !important;
    margin: 0 !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3) !important;
    color: white !important;
}

.main-header p {
    color: white !important;
    font-size: 1.2em !important;
    opacity: 0.95 !important;
}
</style>
""", unsafe_allow_html=True)
# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

@st.cache_resource
def load_stopwords_id():
    try:
        return set(stopwords.words('indonesian'))
    except:
        nltk.download('stopwords')
        nltk.download('punkt')
        return set(stopwords.words('indonesian'))

@st.cache_resource
def load_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

SLANG_DICT = {
    "gk":"tidak","ga":"tidak","nggak":"tidak","ngga":"tidak","bgt":"banget","bngt":"banget",
    "kmrn":"kemarin","bsk":"besok","brg":"barang","dr":"dari","udh":"sudah","sdh":"sudah",
    "blm":"belum","tp":"tapi","sm":"sama","mantul":"mantap","rekomen":"recommended"
}

POSITIVE_WORDS = {
    "bagus","baik","mantap","puas","cepat","recommended","murah","oke","keren","rapi","top",
    "mantab","suka","worth","terbaik","mantul","wangi","sesuai","praktis","worthit","good",
    "nice","asli","original","ori","premium","berkualitas","awet","fast respon","ramah",
    "helpful","responsif","memuaskan","luar biasa","super","excellent","perfect","amazing",
    "pengiriman cepat","packing rapi","packing aman","kemasan rapi","harga bagus","value for money",
    "terjangkau","hemat","jos","josss","ciamik","sip","the best","recommended seller"
}

NEGATIVE_WORDS = {
    "jelek","buruk","rusak","mengecewakan","lambat","cacat","robek","parah","menyesal",
    "tidak bagus","tidak sesuai","tidak puas","zonk","tipu","bohong","kurang baik","ngecewain",
    "tidak sesuai deskripsi","tidak ori","palsu","abal","kualitas buruk","barang bekas",
    "pengiriman lambat","telat","packing jelek","kemasan rusak","retak","pecah","kotor","bau",
    "lecet","tidak direkomendasikan","tidak worth it","overprice","mahal","kemahalan",
    "tidak nyaman","kecewa","tidak berfungsi","tidak nyala","mati total","hang","error",
    "lemot","lag","freeze","bermasalah","pelayanan buruk","seller tidak responsif","slow respon",
    "salah kirim","barang salah","salah ukuran","tidak sesuai pesanan","barang palsu","kw"
}

NEGATION_WORDS = {"tidak","kurang","belum","bukan"}

def normalize_slang(text):
    return " ".join([SLANG_DICT.get(w, w) for w in text.split()])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9.,!? ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_and_remove_stopwords(text, stopwords_set):
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    tokens = [w for w in tokens if len(w) > 1 or w in NEGATION_WORDS]
    tokens = [w for w in tokens if w not in stopwords_set]
    return " ".join(tokens)

def handle_negation(text):
    tokens = text.split()
    words = []
    skip_next = False
    for i in range(len(tokens)-1):
        if tokens[i] in NEGATION_WORDS:
            words.append(tokens[i] + "_" + tokens[i+1])
            skip_next = True
        else:
            if not skip_next:
                words.append(tokens[i])
            skip_next = False
    if not skip_next and len(tokens) > 0:
        words.append(tokens[-1])
    return " ".join(words)

def preprocess_text(text, stopwords_set):
    text = clean_text(text)
    text = normalize_slang(text)
    text = tokenize_and_remove_stopwords(text, stopwords_set)
    text = handle_negation(text)
    return text

def sentiment_lexicon_features(text):
    """FIXED: Return dictionary instead of Series"""
    tokens = text.split()
    L = max(1, len(tokens))
    pos_count = sum(1 for w in tokens if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in tokens if w in NEGATIVE_WORDS)
    negation_count = sum(1 for w in tokens if w in NEGATION_WORDS)
    sentiment_score = pos_count - neg_count
    return {
        'pos_count': pos_count,
        'neg_count': neg_count,
        'negation_count': negation_count,
        'sentiment_score': sentiment_score
    }

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_data():
    try:
        df_clean = pd.read_csv('dataset_bersih.csv')
        return df_clean
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_sentiment_distribution(df):
    sentiment_counts = df['sentiment'].value_counts()
    colors = {'Positif': '#10b981', 'Netral': '#3b82f6', 'Negatif': '#ef4444'}
    fig = go.Figure(data=[go.Bar(
        x=sentiment_counts.index, y=sentiment_counts.values,
        marker=dict(color=[colors.get(x, '#999') for x in sentiment_counts.index], line=dict(color='white', width=2)),
        text=sentiment_counts.values, textposition='auto'
    )])
    fig.update_layout(
        title={'text': '📊 Distribusi Sentimen', 'x': 0.5}, 
        xaxis_title='Sentimen', yaxis_title='Jumlah Review', 
        template='plotly_dark', height=500,
        font=dict(color='white'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(30, 41, 59, 0.5)'
    )
    return fig

def create_confusion_matrix_plot(cm, labels):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    annotations = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            annotations.append(f"{cm[i, j]}<br>({cm_normalized[i, j]*100:.1f}%)")
    annotations = np.array(annotations).reshape(cm.shape)
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels, colorscale='Blues',
        text=annotations, texttemplate='%{text}', textfont={"size": 14, "color": "white"}
    ))
    fig.update_layout(
        title={'text': '🎯 Confusion Matrix', 'x': 0.5}, 
        xaxis_title='Predicted', yaxis_title='Actual', 
        template='plotly_dark', height=600,
        font=dict(color='white'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(30, 41, 59, 0.5)'
    )
    return fig

def create_model_comparison(results):
    models = list(results.keys())
    fig = go.Figure()
    accuracies = [results[model]['test_accuracy'] for model in models]
    fig.add_trace(go.Bar(
        name='Test Accuracy', x=models, y=accuracies, 
        text=[f'{v:.4f}' for v in accuracies], textposition='auto', 
        marker_color='#3b82f6'
    ))
    f1_scores = [results[model]['test_f1'] for model in models]
    fig.add_trace(go.Bar(
        name='Test F1-macro', x=models, y=f1_scores,
        text=[f'{v:.4f}' for v in f1_scores], textposition='auto', 
        marker_color='#8b5cf6'
    ))
    fig.update_layout(
        title={'text': '🏆 Perbandingan Performa Model', 'x': 0.5},
        xaxis_title='Model', yaxis_title='Score', barmode='group',
        template='plotly_dark', height=500, yaxis=dict(range=[0, 1]),
        font=dict(color='white'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(30, 41, 59, 0.5)'
    )
    return fig

def create_overfitting_comparison(results):
    models = list(results.keys())
    fig = go.Figure()
    train_f1 = [results[model]['train_f1'] for model in models]
    fig.add_trace(go.Bar(
        name='Train F1', x=models, y=train_f1,
        text=[f'{v:.4f}' for v in train_f1], textposition='auto', 
        marker_color='#10b981'
    ))
    test_f1 = [results[model]['test_f1'] for model in models]
    fig.add_trace(go.Bar(
        name='Test F1', x=models, y=test_f1,
        text=[f'{v:.4f}' for v in test_f1], textposition='auto', 
        marker_color='#ef4444'
    ))
    fig.update_layout(
        title={'text': '🔍 Overfitting Check', 'x': 0.5},
        xaxis_title='Model', yaxis_title='F1-Score', barmode='group',
        template='plotly_dark', height=500, yaxis=dict(range=[0, 1]),
        font=dict(color='white'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(30, 41, 59, 0.5)'
    )
    return fig

def create_wordcloud(text, title, colormap='viridis'):
    if not text or len(text.strip()) == 0:
        return None
    cleaned_text = " ".join([w for w in text.split() if len(w) > 1])
    if not cleaned_text:
        return None
    wordcloud = WordCloud(
        width=1600, height=800, background_color='#1e293b', 
        colormap=colormap, max_words=100
    ).generate(cleaned_text)
    fig, ax = plt.subplots(figsize=(20, 10), facecolor='#1e293b')
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_facecolor('#1e293b')
    ax.set_title(title, fontsize=24, fontweight='bold', pad=20, color='white')
    plt.tight_layout(pad=0)
    return fig

def create_top_words_chart(df, sentiment, n=20):
    text = ' '.join(df[df['sentiment'] == sentiment]['text'].values)
    words = [w for w in text.split() if len(w) > 1]
    word_counts = Counter(words).most_common(n)
    if not word_counts:
        return None
    words_list = [w[0] for w in word_counts]
    counts_list = [w[1] for w in word_counts]
    colors_map = {'Positif': 'Greens', 'Netral': 'Blues', 'Negatif': 'Reds'}
    fig = go.Figure(data=[go.Bar(
        x=counts_list, y=words_list, orientation='h',
        marker=dict(color=counts_list, colorscale=colors_map.get(sentiment, 'Viridis')),
        text=counts_list, textposition='auto'
    )])
    fig.update_layout(
        title={'text': f'🔤 Top {n} Kata - {sentiment}', 'x': 0.5},
        xaxis_title='Frekuensi', yaxis_title='Kata',
        template='plotly_dark', height=600, yaxis=dict(autorange="reversed"),
        font=dict(color='white'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(30, 41, 59, 0.5)'
    )
    return fig
# ============================================================================
# MAIN APP
# ============================================================================

def main():
    df_clean = load_data()
    
    if df_clean is None:
        st.error("❌ File 'dataset_bersih.csv' tidak ditemukan!")
        return
    
    # SIDEBAR
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: white; font-size: 2.5em;'>🛍️</h1>
        <h2 style='color: white;'>Tokopedia</h2>
        <p style='color: rgba(255,255,255,0.8);'>Sentiment Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    menu = st.sidebar.radio(
        "📍 Navigation", 
        ["🏠 Dashboard", "📊 EDA", "🤖 Model Training", 
         "🔮 Prediction", "📈 Evaluation", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Dataset Info")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Reviews", f"{len(df_clean):,}")
    with col2:
        st.metric("Features", len(df_clean.columns))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Sentimen")
    sentiment_counts = df_clean['sentiment'].value_counts()
    for sentiment in ['Positif', 'Netral', 'Negatif']:
        count = sentiment_counts.get(sentiment, 0)
        pct = (count / len(df_clean) * 100) if len(df_clean) > 0 else 0
        emoji = "😊" if sentiment == "Positif" else ("😐" if sentiment == "Netral" else "😞")
        st.sidebar.metric(f"{emoji} {sentiment}", f"{count:,}", f"{pct:.1f}%")
    
    # ========================================================================
    # DASHBOARD
    # ========================================================================
    if menu == "🏠 Dashboard":
        st.markdown("""
        <div class='main-header'>
            <h1>🛍️ Tokopedia Sentiment Analysis</h1>
            <p>Analisis Sentimen Ulasan Produk dengan NLP & Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Reviews", f"{len(df_clean):,}")
        with col2:
            positive_pct = (df_clean['sentiment'] == 'Positif').sum() / len(df_clean) * 100
            st.metric("😊 Positive", f"{positive_pct:.1f}%")
        with col3:
            neutral_pct = (df_clean['sentiment'] == 'Netral').sum() / len(df_clean) * 100
            st.metric("😐 Neutral", f"{neutral_pct:.1f}%")
        with col4:
            negative_pct = (df_clean['sentiment'] == 'Negatif').sum() / len(df_clean) * 100
            st.metric("😞 Negative", f"{negative_pct:.1f}%")
        
        fig_sentiment = create_sentiment_distribution(df_clean)
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
        st.markdown("### 📋 Sample Data")
        st.dataframe(
            df_clean[['text_asli', 'text', 'sentiment', 'text_length']].head(10), 
            use_container_width=True, height=400
        )
    
    # ========================================================================
    # EDA
    # ========================================================================
    elif menu == "📊 EDA":
        st.markdown("""
        <div class='main-header'>
            <h1>📊 Exploratory Data Analysis</h1>
            <p>Analisis Mendalam Dataset</p>
        </div>
        """, unsafe_allow_html=True)
        
        tabs = st.tabs(["📈 Statistik", "☁️ Word Cloud", "🔤 Top Words"])
        
        with tabs[0]:
            st.markdown("### 📊 Statistik Deskriptif")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Panjang Teks")
                st.dataframe(df_clean.groupby('sentiment')['text_length'].describe())
            with col2:
                st.markdown("#### Sentiment Score")
                st.dataframe(df_clean.groupby('sentiment')['sentiment_score'].describe())
        
        with tabs[1]:
            st.markdown("### ☁️ Word Cloud")
            sentiment_choice = st.selectbox(
                "Pilih Sentimen:", 
                ["Semua", "Positif", "Netral", "Negatif"]
            )
            if sentiment_choice == "Semua":
                text_data = ' '.join(df_clean['text'].values)
                colormap = 'viridis'
            else:
                text_data = ' '.join(
                    df_clean[df_clean['sentiment'] == sentiment_choice]['text'].values
                )
                colormap = 'Greens' if sentiment_choice == 'Positif' else (
                    'Blues' if sentiment_choice == 'Netral' else 'Reds'
                )
            
            if text_data.strip():
                fig_wc = create_wordcloud(text_data, f"Word Cloud - {sentiment_choice}", colormap)
                if fig_wc:
                    st.pyplot(fig_wc)
        
        with tabs[2]:
            st.markdown("### 🔤 Top Words")
            col1, col2 = st.columns([1, 3])
            with col1:
                sentiment_choice = st.selectbox(
                    "Sentimen:", 
                    ["Positif", "Netral", "Negatif"], 
                    key="top"
                )
                n_words = st.slider("Jumlah:", 10, 50, 20)
            with col2:
                fig_top = create_top_words_chart(df_clean, sentiment_choice, n_words)
                if fig_top:
                    st.plotly_chart(fig_top, use_container_width=True)
    # ========================================================================
    # MODEL TRAINING
    # ========================================================================
    elif menu == "🤖 Model Training":
        st.markdown("""
        <div class='main-header'>
            <h1>🤖 Model Training</h1>
            <p>Train Model Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("""
        **Pipeline (Sesuai Notebook):**
        1. Train-Test Split (80:20)
        2. TF-IDF (10,000 features, ngram 1-3)
        3. Lexicon features (4 fitur)
        4. Training: LR, MNB (GridSearch), RF (SVD)
        5. Evaluasi & Overfitting Check
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
            random_state = st.number_input("Random State", 0, 100, 42)
        with col2:
            max_features = st.number_input("Max TF-IDF", 1000, 15000, 10000, 1000)
            use_grid = st.checkbox("GridSearchCV (MNB)", value=True)
        with col3:
            models_to_train = st.multiselect(
                "Pilih Model:",
                ["Logistic Regression", "Multinomial Naive Bayes", "Random Forest"],
                default=["Logistic Regression", "Multinomial Naive Bayes", "Random Forest"]
            )
        
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            if not models_to_train:
                st.error("❌ Pilih minimal satu model!")
            else:
                with st.spinner("Training..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Prepare data
                        status_text.text("📊 Preparing data...")
                        progress_bar.progress(10)
                        
                        X = df_clean['text']
                        y = df_clean['sentiment']
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=random_state, stratify=y
                        )
                        
                        # TF-IDF
                        status_text.text("🔤 TF-IDF...")
                        progress_bar.progress(25)
                        
                        vectorizer = TfidfVectorizer(
                            max_features=max_features, ngram_range=(1,3), min_df=2
                        )
                        X_train_tfidf = vectorizer.fit_transform(X_train)
                        X_test_tfidf = vectorizer.transform(X_test)
                        
                        # Lexicon features
                        status_text.text("➕ Lexicon features...")
                        progress_bar.progress(35)
                        
                        train_extra = df_clean.loc[
                            X_train.index, 
                            ['pos_count','neg_count','negation_count','sentiment_score']
                        ].to_numpy()
                        test_extra = df_clean.loc[
                            X_test.index, 
                            ['pos_count','neg_count','negation_count','sentiment_score']
                        ].to_numpy()
                        
                        X_train_final = hstack([X_train_tfidf, csr_matrix(train_extra)])
                        X_test_final = hstack([X_test_tfidf, csr_matrix(test_extra)])
                        
                        # Training
                        results = {}
                        trained_models = {}
                        
                        for idx, model_name in enumerate(models_to_train):
                            status_text.text(f"🤖 Training {model_name}...")
                            progress_bar.progress(35 + int((idx+1) * 45 / len(models_to_train)))
                            
                            if model_name == "Logistic Regression":
                                model = LogisticRegression(
                                    random_state=random_state, max_iter=3000, 
                                    C=2.0, class_weight="balanced", solver="lbfgs"
                                )
                                model.fit(X_train_final, y_train)
                                y_train_pred = model.predict(X_train_final)
                                y_test_pred = model.predict(X_test_final)
                                trained_models[model_name] = {
                                    'model': model, 
                                    'X_train': X_train_final, 
                                    'X_test': X_test_final
                                }
                            
                            elif model_name == "Multinomial Naive Bayes":
                                if use_grid:
                                    from sklearn.model_selection import GridSearchCV
                                    grid = GridSearchCV(
                                        MultinomialNB(), 
                                        {"alpha": [0.1, 0.3, 0.5, 1.0]}, 
                                        cv=3, scoring="f1_macro", n_jobs=-1
                                    )
                                    grid.fit(X_train_final, y_train)
                                    model = grid.best_estimator_
                                else:
                                    model = MultinomialNB(alpha=0.1)
                                    model.fit(X_train_final, y_train)
                                y_train_pred = model.predict(X_train_final)
                                y_test_pred = model.predict(X_test_final)
                                trained_models[model_name] = {
                                    'model': model, 
                                    'X_train': X_train_final, 
                                    'X_test': X_test_final
                                }
                            
                            else:  # Random Forest
                                svd = TruncatedSVD(n_components=300, random_state=random_state)
                                X_train_rf = svd.fit_transform(X_train_final)
                                X_test_rf = svd.transform(X_test_final)
                                model = RandomForestClassifier(
                                    n_estimators=250, max_depth=22, 
                                    class_weight="balanced", 
                                    random_state=random_state, n_jobs=4
                                )
                                model.fit(X_train_rf, y_train)
                                y_train_pred = model.predict(X_train_rf)
                                y_test_pred = model.predict(X_test_rf)
                                trained_models[model_name] = {
                                    'model': model, 'svd': svd, 
                                    'X_train': X_train_rf, 'X_test': X_test_rf
                                }
                            
                            # Metrics
                            results[model_name] = {
                                'train_accuracy': accuracy_score(y_train, y_train_pred),
                                'test_accuracy': accuracy_score(y_test, y_test_pred),
                                'train_f1': f1_score(y_train, y_train_pred, average='macro'),
                                'test_f1': f1_score(y_test, y_test_pred, average='macro'),
                                'f1_gap': f1_score(y_train, y_train_pred, average='macro') - 
                                         f1_score(y_test, y_test_pred, average='macro'),
                                'y_pred': y_test_pred
                            }
                        
                        # Save to session
                        st.session_state['trained_models'] = trained_models
                        st.session_state['vectorizer'] = vectorizer
                        st.session_state['results'] = results
                        st.session_state['y_train'] = y_train
                        st.session_state['y_test'] = y_test
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Training completed!")
                        
                        st.success("🎉 Training berhasil!")
                        
                        # Results
                        st.markdown("### 📊 Results")
                        results_df = pd.DataFrame({
                            'Model': list(results.keys()),
                            'Test Acc': [f"{results[m]['test_accuracy']:.4f}" for m in results.keys()],
                            'Test F1': [f"{results[m]['test_f1']:.4f}" for m in results.keys()],
                            'Train F1': [f"{results[m]['train_f1']:.4f}" for m in results.keys()],
                            'F1 Gap': [f"{results[m]['f1_gap']:.4f}" for m in results.keys()]
                        })
                        st.dataframe(results_df, use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            fig_comp = create_model_comparison(results)
                            st.plotly_chart(fig_comp, use_container_width=True)
                        with col2:
                            fig_over = create_overfitting_comparison(results)
                            st.plotly_chart(fig_over, use_container_width=True)
                        
                        # Expected results
                        st.info("""
                        **🎯 Expected Results (dari Notebook):**
                        - **Logistic Regression:** Acc: 0.7493, F1: 0.5653
                        - **Multinomial NB:** Acc: 0.8405, F1: 0.5170
                        - **Random Forest:** Acc: 0.7988, F1: 0.5238
                        """)
                    
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        if 'trained_models' in st.session_state:
            st.markdown("---")
            st.success(f"✅ {len(st.session_state['trained_models'])} model(s) tersimpan")
    # ========================================================================
    # PREDICTION PAGE
    # ========================================================================
    elif menu == "🔮 Prediction":
        st.markdown("""
        <div class='main-header'>
            <h1>🔮 Sentiment Prediction</h1>
            <p>Prediksi Sentimen Review Baru dengan Semua Model</p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'trained_models' not in st.session_state:
            st.warning("⚠️ Belum ada model yang di-train. Silakan train model terlebih dahulu di menu **Model Training**.")
        else:
            st.markdown("### ✍️ Input Review")
            
            user_input = st.text_area(
                "Masukkan review produk:",
                height=150,
                placeholder="Contoh: Barang bagus, pengiriman cepat, harga terjangkau, recommended seller!"
            )
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                predict_button = st.button("🔮 Predict All Models", type="primary", use_container_width=True)
            
            if predict_button:
                if user_input.strip():
                    with st.spinner("🔄 Analyzing with all models..."):
                        # Preprocessing
                        stopwords_set = load_stopwords_id()
                        processed_text = preprocess_text(user_input, stopwords_set)
                        
                        # TF-IDF
                        vectorizer = st.session_state['vectorizer']
                        text_tfidf = vectorizer.transform([processed_text])
                        
                        # Lexicon features
                        features_dict = sentiment_lexicon_features(processed_text)
                        
                        pos_count = features_dict['pos_count']
                        neg_count = features_dict['neg_count']
                        negation_count = features_dict['negation_count']
                        sentiment_score = features_dict['sentiment_score']
                        
                        extra_features = np.array([[pos_count, neg_count, negation_count, sentiment_score]])
                        
                        # Combine
                        final_vec = hstack([text_tfidf, csr_matrix(extra_features)])
                        
                        # Display preprocessing results
                        st.markdown("---")
                        st.markdown("### 📝 Preprocessing Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Original Text")
                            st.info(user_input)
                        
                        with col2:
                            st.markdown("#### Processed Text")
                            st.info(processed_text if processed_text else "N/A")
                        
                        # Lexicon features
                        st.markdown("#### 📊 Lexicon Features")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("➕ Positive Words", int(pos_count))
                        with col2:
                            st.metric("➖ Negative Words", int(neg_count))
                        with col3:
                            st.metric("🚫 Negation Words", int(negation_count))
                        with col4:
                            st.metric("📊 Sentiment Score", f"{sentiment_score:.3f}")
                        
                        # Predict with all models
                        st.markdown("---")
                        st.markdown("### 🤖 Predictions from All Models")
                        
                        predictions = {}
                        probabilities = {}
                        
                        for model_name, model_data in st.session_state['trained_models'].items():
                            model = model_data['model']
                            
                            # Transform input based on model type
                            if 'svd' in model_data:  # Random Forest
                                input_vec = model_data['svd'].transform(final_vec)
                            else:
                                input_vec = final_vec
                            
                            # Predict
                            prediction = model.predict(input_vec)[0]
                            predictions[model_name] = prediction
                            
                            # Get probability if available
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(input_vec)[0]
                                probabilities[model_name] = {
                                    label: prob for label, prob in zip(model.classes_, proba)
                                }
                        
                        # Display individual model predictions
                        st.markdown("#### 📊 Individual Model Predictions")
                        
                        cols = st.columns(len(predictions))
                        
                        for idx, (model_name, prediction) in enumerate(predictions.items()):
                            with cols[idx]:
                                st.markdown(f"**{model_name}**")
                                
                                if prediction == "Positif":
                                    st.success(f"😊 {prediction}")
                                elif prediction == "Netral":
                                    st.info(f"😐 {prediction}")
                                else:
                                    st.error(f"😞 {prediction}")
                                
                                # Show confidence
                                if model_name in probabilities:
                                    max_prob = max(probabilities[model_name].values())
                                    st.metric("Confidence", f"{max_prob*100:.2f}%")
                        
                        # Summary - Voting
                        st.markdown("---")
                        st.markdown("### 🗳️ Consensus Prediction (Majority Voting)")
                        
                        vote_counts = Counter(predictions.values())
                        final_prediction = vote_counts.most_common(1)[0][0]
                        
                        col1, col2, col3 = st.columns([1, 2, 1])
                        
                        with col2:
                            if final_prediction == "Positif":
                                st.markdown("""
                                <div style='background: linear-gradient(135deg, #10b981 0%, #34d399 100%); 
                                            padding: 30px; border-radius: 15px; text-align: center;'>
                                    <h1 style='color: white; margin: 0;'>😊 POSITIF</h1>
                                    <p style='color: white; font-size: 1.2em; margin: 10px 0;'>
                                        Mayoritas model memprediksi sentimen POSITIF
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            elif final_prediction == "Netral":
                                st.markdown("""
                                <div style='background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%); 
                                            padding: 30px; border-radius: 15px; text-align: center;'>
                                    <h1 style='color: white; margin: 0;'>😐 NETRAL</h1>
                                    <p style='color: white; font-size: 1.2em; margin: 10px 0;'>
                                        Mayoritas model memprediksi sentimen NETRAL
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div style='background: linear-gradient(135deg, #ef4444 0%, #f87171 100%); 
                                            padding: 30px; border-radius: 15px; text-align: center;'>
                                    <h1 style='color: white; margin: 0;'>😞 NEGATIF</h1>
                                    <p style='color: white; font-size: 1.2em; margin: 10px 0;'>
                                        Mayoritas model memprediksi sentimen NEGATIF
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Voting details
                        st.markdown("#### 🗳️ Voting Details")
                        vote_df = pd.DataFrame({
                            'Sentimen': list(vote_counts.keys()),
                            'Jumlah Vote': list(vote_counts.values()),
                            'Persentase': [f"{v/len(predictions)*100:.1f}%" for v in vote_counts.values()]
                        })
                        st.dataframe(vote_df, use_container_width=True)
                        
                        # Probability distribution for each model
                        if probabilities:
                            st.markdown("---")
                            st.markdown("### 📊 Probability Distribution per Model")
                            
                            for model_name, probs in probabilities.items():
                                st.markdown(f"#### {model_name}")
                                
                                labels = list(probs.keys())
                                values = [probs[l] * 100 for l in labels]
                                colors_list = [
                                    '#10b981' if l == 'Positif' else 
                                    '#3b82f6' if l == 'Netral' else '#ef4444' 
                                    for l in labels
                                ]
                                
                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=labels,
                                        y=values,
                                        marker=dict(color=colors_list),
                                        text=[f'{v:.2f}%' for v in values],
                                        textposition='auto',
                                    )
                                ])
                                
                                fig.update_layout(
                                    xaxis_title='Sentiment',
                                    yaxis_title='Probability (%)',
                                    template='plotly_dark',
                                    height=300,
                                    showlegend=False,
                                    font=dict(color='white'),
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(30, 41, 59, 0.5)'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("❌ Masukkan review terlebih dahulu!")
    
    # ========================================================================
    # EVALUATION
    # ========================================================================
    elif menu == "📈 Evaluation":
        st.markdown("""
        <div class='main-header'>
            <h1>📈 Model Evaluation</h1>
            <p>Evaluasi Detail Performa Model</p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'results' not in st.session_state:
            st.warning("⚠️ Train model terlebih dahulu!")
        else:
            results = st.session_state['results']
            y_test = st.session_state['y_test']
            
            model_choice = st.selectbox("Pilih Model:", list(results.keys()))
            y_pred = results[model_choice]['y_pred']
            
            st.markdown("### 📊 Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            test_acc = results[model_choice]['test_accuracy']
            test_f1 = results[model_choice]['test_f1']
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            with col1:
                st.metric("Test Accuracy", f"{test_acc:.4f}")
            with col2:
                st.metric("Test F1-macro", f"{test_f1:.4f}")
            with col3:
                st.metric("Precision", f"{precision:.4f}")
            with col4:
                st.metric("Recall", f"{recall:.4f}")
            
            st.markdown("### 🔍 Overfitting Check")
            col1, col2, col3 = st.columns(3)
            
            train_acc = results[model_choice]['train_accuracy']
            train_f1 = results[model_choice]['train_f1']
            f1_gap = results[model_choice]['f1_gap']
            
            with col1:
                st.metric("Train Acc", f"{train_acc:.4f}")
                st.metric("Test Acc", f"{test_acc:.4f}")
            with col2:
                st.metric("Train F1", f"{train_f1:.4f}")
                st.metric("Test F1", f"{test_f1:.4f}")
            with col3:
                st.metric("F1 Gap", f"{f1_gap:.4f}")
                if f1_gap > 0.2:
                    st.error("⚠️ Overfitting!")
                elif f1_gap > 0.1:
                    st.warning("⚠️ Slight overfitting")
                else:
                    st.success("✅ Good generalization")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🎯 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                labels = sorted(y_test.unique())
                fig_cm = create_confusion_matrix_plot(cm, labels)
                st.plotly_chart(fig_cm, use_container_width=True)
            
            with col2:
                st.markdown("### 📋 Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
            
            if len(results) > 1:
                st.markdown("### 🏆 Model Comparison")
                col1, col2 = st.columns(2)
                with col1:
                    fig_comp = create_model_comparison(results)
                    st.plotly_chart(fig_comp, use_container_width=True)
                with col2:
                    fig_over = create_overfitting_comparison(results)
                    st.plotly_chart(fig_over, use_container_width=True)
    
    # ========================================================================
    # ABOUT
    # ========================================================================
    else:
        st.markdown("""
        <div class='main-header'>
            <h1>ℹ️ About This Project</h1>
            <p>Tokopedia Sentiment Analysis Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 Project Overview
        
        Aplikasi ini menganalisis sentimen ulasan produk Tokopedia menggunakan:
        - **Natural Language Processing (NLP)**
        - **Machine Learning Models**
        - **IndoBERT** untuk labeling sentimen
        
        ### 📊 Dataset
        - **Total Reviews:** 36,383 (setelah preprocessing)
        - **Train/Test Split:** 29,106 / 7,277 (80:20)
        - **Sentimen:** Positif, Netral, Negatif (dari IndoBERT)
        - **Features:** TF-IDF (10,000) + Lexicon (4)
        
        ### 🤖 Models
        
        **1. Logistic Regression**
        - Accuracy: 0.7493
        - F1-macro: 0.5653
        - Fast and interpretable
        
        **2. Multinomial Naive Bayes**
        - Accuracy: 0.8405 (highest!)
        - F1-macro: 0.5170
        - GridSearchCV tuning
        
        **3. Random Forest**
        - Accuracy: 0.7988
        - F1-macro: 0.5238
        - SVD dimensionality reduction
        
        ### 🔄 Pipeline
        
        1. **Data Loading** - Load dataset CSV
        2. **Preprocessing** - Cleaning, tokenization, stopwords, stemming, negation handling
        3. **Feature Engineering** - TF-IDF + Lexicon features
        4. **Model Training** - LR, MNB, RF dengan hyperparameter tuning
        5. **Evaluation** - Accuracy, F1, Confusion Matrix, Overfitting Check
        6. **Prediction** - Real-time sentiment prediction
        
        ### 🛠️ Tech Stack
        
        - **Frontend:** Streamlit
        - **NLP:** NLTK, Sastrawi
        - **ML:** Scikit-learn
        - **Visualization:** Plotly, Matplotlib, WordCloud
        - **Data:** Pandas, NumPy
        
        ### 📝 Key Features
        
        ✅ Interactive Dashboard  
        ✅ Comprehensive EDA  
        ✅ Model Training dengan 3 algoritma  
        ✅ Real-time Prediction  
        ✅ Detailed Model Evaluation  
        ✅ Overfitting Detection  
        ✅ Word Cloud & Top Words Analysis  
        ✅ Premium UI/UX Design  
        
        ### 👥 Team
        
        Made with ❤️ by Bayu & Yudha & Alfonsus
        
        ### 📞 Contact
        
        For questions or feedback, please contact us.
        
        ---
        
        © 2025 Tokopedia Sentiment Analysis. All Rights Reserved.
        """)

if __name__ == "__main__":
    main()



