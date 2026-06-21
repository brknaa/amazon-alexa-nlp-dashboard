import streamlit as st
import sqlite3
import pandas as pd
import joblib
import os
import datetime

# --- 1. AYARLAR VE YÜKLEMELER ---
st.set_page_config(page_title="NLP Duygu Analizi", layout="wide")
st.title("Amazon Alexa Yorumları - Duygu Analizi Paneli")

# Modelleri önbelleğe alarak yükleyen fonksiyon (Uygulamayı hızlandırır)
@st.cache_resource
def load_models():
    base_path = os.getcwd()
    tfidf_loaded = joblib.load(os.path.join(base_path, 'models', 'tfidf.pkl'))
    model_loaded = joblib.load(os.path.join(base_path, 'models', 'log_model.pkl'))
    return tfidf_loaded, model_loaded

try:
    tfidf, model = load_models()
except Exception as e:
    st.error("Modeller bulunamadı! Lütfen Jupyter Notebook'ta modelleri kaydettiğinizden emin olun.")
    st.stop()

# Veritabanı bağlantı fonksiyonu
db_path = os.path.join(os.getcwd(), 'data', 'sentiment_db.sqlite')
def get_db_connection():
    return sqlite3.connect(db_path)

# --- 2. KULLANICI GİRİŞİ VE TAHMİN BÖLÜMÜ ---
st.subheader("Yeni Yorum Analizi")
user_input = st.text_area("Analiz edilecek İngilizce yorumu buraya yazın:", height=100)

if st.button("Duyguyu Analiz Et"):
    if user_input.strip() != "":
        # Tahmin olasılıklarını al
        vectorized_text = tfidf.transform([user_input])
        probabilities = model.predict_proba(vectorized_text)[0]
        
        prob_neg = probabilities[0] # Negatif olma olasılığı
        prob_pos = probabilities[1] # Pozitif olma olasılığı
        
        # Eşik değeri (Threshold) mantığı ile NÖTR sınıfını oluşturma
        if 0.35 <= prob_pos <= 0.65:
            sentiment_label = "Nötr (2)"
            prediction_val = 2 # Veritabanında Nötr için 2 kodunu kullanıyoruz
            confidence = max(prob_pos, prob_neg) 
            st.warning(f"Sonuç: {sentiment_label} | Güven Skoru: %{confidence*100:.1f} (Model Kararsız)")
        elif prob_pos > 0.60:
            sentiment_label = "Pozitif (1)"
            prediction_val = 1
            confidence = prob_pos
            st.success(f"Sonuç: {sentiment_label} | Güven Skoru: %{confidence*100:.1f}")
        else:
            sentiment_label = "Negatif (0)"
            prediction_val = 0
            confidence = prob_neg
            st.error(f"Sonuç: {sentiment_label} | Güven Skoru: %{confidence*100:.1f}")
        
        # Sonucu veritabanına kaydet
        conn = get_db_connection()
        cursor = conn.cursor()
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO predictions (review_text, predicted_sentiment, confidence_score, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (user_input, prediction_val, float(confidence), current_time))
        conn.commit()
        conn.close()
        
    else:
        st.warning("Lütfen analiz için bir metin girin.")

# --- 3. ANALİTİK VE GÖRSELLEŞTİRME BÖLÜMÜ ---
st.divider()
st.subheader("Geçmiş Analiz İstatistikleri ve Görselleştirme")

try:
    conn = get_db_connection()
    df_db = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()
    
    if not df_db.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Son Tahmin Kayıtları**")
            # Sadece son 5 kaydı göster ve ID'ye göre ters sırala (en yeni en üstte)
            st.dataframe(df_db[['timestamp', 'review_text', 'predicted_sentiment', 'confidence_score']].tail().sort_index(ascending=False))
        
        with col2:
            st.write("**Veritabanındaki Duygu Dağılımı**")
            # 1 ve 0'ları metne çevirip grafiğini çiz
            sentiment_counts = df_db['predicted_sentiment'].value_counts().rename(index={1: 'Pozitif', 0: 'Negatif'})
            st.bar_chart(sentiment_counts)
    else:
        st.info("Veritabanında henüz kayıt bulunmuyor.")
except Exception as e:
    st.error("Veritabanı okunurken bir hata oluştu.")