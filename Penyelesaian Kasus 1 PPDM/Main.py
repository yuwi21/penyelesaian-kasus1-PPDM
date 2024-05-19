import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

st.title("Klasifikasi Berita dengan Multinomial Naive Bayes")
st.header("Kelompok 2 Kasus Klasifikasi Text")

# Fungsi preprocessing
def olah_text(text):
    text = text.lower()
    text = ' '.join(re.sub(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', ' ', text).split())
    text = re.sub(r'\d+', '', text)
    punct = str.maketrans('', '', string.punctuation + string.digits)
    text = text.translate(punct)
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    filtered_words = [w for w in tokens if w not in Sastrawi_StopWords_id]
    
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w) for w in filtered_words]
    text = " ".join(lemma_words)
    return text

# Load stopwords and stemmer
stemmer = StemmerFactory().create_stemmer()
stopword_factory = StopWordRemoverFactory()
Sastrawi_StopWords_id = stopword_factory.get_stop_words()

# Mengupload file
uploaded_file = st.file_uploader("Unggah file dataset (format .txt)", type="txt")

if uploaded_file:
    # Membaca file yang diupload
    df = pd.read_csv(uploaded_file, delimiter='\t', header=None)
    df.columns = ['kategori', 'sumber', 'berita']
    
    # Menampilkan original dataframe
    st.subheader("Dataframe Original")
    st.write(df.head(11))

    # Mmenghilangkan Kategori yang tidak diinginkan
    df = df[df['kategori'].isin(['showbiz', 'tajuk utama']) == False]

    # Drop 'sumber' column
    df.drop('sumber', axis=1, inplace=True)

    # Preprocess text
    df['Hasil'] = df['berita'].apply(olah_text)
    
    # Menampilkan preprocessed dataframe
    st.subheader("Dataframe Setelah Preprocessing")
    st.write(df.head(11))
    
    # TF-IDF Vectorization
    tf_idf = TfidfVectorizer(ngram_range=(1, 1))
    X_tf_idf = tf_idf.fit_transform(df['Hasil']).toarray()
    y = df['kategori']
    
    # Seleksi Fitur
    chi2_features = SelectKBest(chi2, k=10000)
    X_kbest_features = chi2_features.fit_transform(X_tf_idf, y)

    # Split data training dan set data testing 
    X_train, X_test, y_train, y_test = train_test_split(X_kbest_features, y, test_size=0.2, random_state=42)
    
    # Hyperparameter tuning menggunakan GridSearchCV
    param_grid = {
        'alpha': [0.01, 0.1, 0.5, 1.0, 2.0],
        'fit_prior': [True, False]
    }
    grid_search = GridSearchCV(estimator=MultinomialNB(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    # Mendapatkan hyperparameter terbaik
    best_params = grid_search.best_params_
    st.write(f"Hyperparameter terbaik: {best_params}")
    
    # Melatih ulang model dengan hyperparameter terbaik
    best_model = grid_search.best_estimator_
    
    # Prediksi Data Tes
    y_pred = best_model.predict(X_test)
    
    # Menampilkan classification report
    st.subheader("Hasil Klasifikasi")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write(report_df)
    
    # Menginput teks yang ingin diklasifikasikan 
    st.subheader("Klasifikasikan Berita Anda Sendiri")
    user_input = st.text_area("Masukkan teks berita di sini")
    
    if st.button("Klasifikasikan"):
        user_preprocessed = olah_text(user_input)
        user_tf_idf = tf_idf.transform([user_preprocessed]).toarray()
        user_kbest = chi2_features.transform(user_tf_idf)
        user_pred = best_model.predict(user_kbest)
        st.write(f"Kategori Berita: {user_pred[0]}")