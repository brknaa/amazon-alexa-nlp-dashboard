from googletrans import Translator
import streamlit as st
import sqlite3
import pandas as pd
import joblib
import os
import datetime
import plotly.express as px 

st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. AYARLAR VE YÜKLEMELER ---
translator = Translator() 
st.set_page_config(page_title="NLP Duygu Analizi", layout="wide")
st.title("Amazon Alexa Yorumları - Duygu Analizi Paneli 🚀")

# Modelleri önbelleğe alarak yükleyen fonksiyon
@st.cache_resource
def load_models():
    base_path = os.getcwd()
    tfidf_loaded = joblib.load(os.path.join(base_path, 'models', 'tfidf.pkl'))
    model_loaded = joblib.load(os.path.join(base_path, 'models', 'log_model.pkl'))
    return tfidf_loaded, model_loaded

try:
    tfidf, model = load_models()
except Exception as e:
    st.error("Modeller bulunamadı!")
    st.stop()

# Veritabanı bağlantı fonksiyonu
db_path = os.path.join(os.getcwd(), 'data', 'sentiment_db.sqlite')
def get_db_connection():
    return sqlite3.connect(db_path)

# --- 2. KULLANICI GİRİŞİ VE TAHMİN BÖLÜMÜ ---
st.subheader("Yeni Yorum Analizi 📝")

# DİL SEÇİMİNİ BURAYA EKLEDİK
dil = st.sidebar.selectbox("Giriş Dili:", ["İngilizce", "Türkçe"])
user_input = st.text_area("Analiz edilecek yorumu buraya yazın:", height=100)

if st.button("Duyguyu Analiz Et"):
    if user_input.strip() != "":
        
        # ÇEVİRİ MANTIĞINI BURAYA EKLEDİK
        metin = user_input
        if dil == "Türkçe":
            try:
                metin = translator.translate(user_input, src='tr', dest='en').text
                st.info(f"Algılanan İngilizce çeviri: {metin}")
            except Exception as e:
                st.error("🌐 Çeviri servisine bağlanılamadı! Lütfen internet bağlantınızı kontrol edin veya yorumu doğrudan İngilizce girin.")
                st.stop() # Hata alırsak kodu burada durdurur, uygulamanın çökmesini engeller.

        # Tahmin olasılıklarını al (Artık 'metin' değişkenini kullanıyoruz)
        vectorized_text = tfidf.transform([metin])
        probabilities = model.predict_proba(vectorized_text)[0]
        
        prob_neg = probabilities[0] 
        prob_pos = probabilities[1] 
        
        # Eşik değeri mantığı
        if 0.35 <= prob_pos <= 0.65:
            sentiment_label = "Nötr (2)"
            prediction_val = 2 
            confidence = max(prob_pos, prob_neg) 
            st.warning(f"Sonuç: {sentiment_label} | Güven Skoru: %{confidence*100:.1f} (Model Kararsız)")
        elif prob_pos > 0.65: # Eşik tutarlılığı için 0.65'e güncelledim
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

# Analitik Kartları
conn = get_db_connection()
df_db = pd.read_sql_query("SELECT * FROM predictions", conn)
conn.close()

# Nötr eklendi ve sütun 4'e çıkarıldı
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Toplam Analiz", len(df_db))
col_b.metric("Pozitif Yorum", len(df_db[df_db['predicted_sentiment'] == 1]))
col_c.metric("Negatif Yorum", len(df_db[df_db['predicted_sentiment'] == 0]))
col_d.metric("Nötr Yorum", len(df_db[df_db['predicted_sentiment'] == 2]))

st.divider()
st.subheader("Geçmiş Analiz İstatistikleri ve Görselleştirme")

try:
    if not df_db.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Son Tahmin Kayıtları**")
            st.dataframe(df_db[['timestamp', 'review_text', 'predicted_sentiment', 'confidence_score']].tail().sort_index(ascending=False))
        with col2:
            st.write("**Veritabanındaki Duygu Dağılımı**")
            sentiment_counts = df_db['predicted_sentiment'].value_counts().rename(index={1: 'Pozitif', 0: 'Negatif', 2: 'Nötr'})
            # Madde 4: Plotly grafiği
            fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="Duygu Dağılımı", hole=0.3)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Veritabanında henüz kayıt bulunmuyor.")
except Exception as e:
    st.error("Veritabanı okunurken bir hata oluştu.")