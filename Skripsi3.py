import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("datakecil.csv")
df = df.drop_duplicates().dropna()
df = df.head(5)  # 🔥 BATASI DATA

# Bersihkan kolom artists dari tanda [ ] '
df['artists'] = df['artists'].astype(str).str.replace(r"[\[\]']", "", regex=True)

# =========================
# CLEANING TEXT
# =========================
def clean_text(text):
    text = str(text).lower() # ubah ke huruf kecil
    text = re.sub(r'[^a-zA-Z\s]', '', text) # hapus simbol/angka
    text = re.sub(r'\s+', ' ', text).strip()    # hapus spasi berlebih 
    return text

df['metadata'] = df['name'] # ambil metadata dari judul lagu
df['metadata_clean'] = df['metadata'].apply(clean_text) # bersihin metadata

# =========================
# TOKENISASI
# =========================
df['tokens'] = df['metadata_clean'].str.split() # pecah jadi token / kata-kata

# 🔥 HAPUS TOKEN KOSONG (PENTING BANGET)
df = df[df['tokens'].apply(len) > 0]    
df = df.reset_index(drop=True)  # reset indek

# =========================
# TF MANUAL
# =========================
vocab = sorted(set([word for tokens in df['tokens'] for word in tokens]))   # ambil semua kata unik 

tf_matrix = []  #untuk menyimpan hasil tf
for tokens in df['tokens']:     # loop setiap lagu 
    total_words = len(tokens)   # jumlah kata dalam lagu
    tf_dict = {}    # dictionary tf

    for word in vocab:  # loop semua kata di vocab
        if total_words == 0:    # jika kosong
            tf_dict[word] = 0
        else:
            tf_dict[word] = tokens.count(word) / total_words    # hitung kata

    tf_matrix.append(tf_dict)

tf_df = pd.DataFrame(tf_matrix)

# =========================
# IDF MANUAL
# =========================
N = len(df)
idf_dict = {}

for word in vocab:
    df_count = sum([1 for tokens in df['tokens'] if word in tokens])
    if df_count == 0:
        idf_dict[word] = 0
    else:
        idf_dict[word] = np.log(N / df_count)

idf_series = pd.Series(idf_dict)

# =========================
# TF-IDF MANUAL
# =========================
tfidf_df = tf_df * idf_series
tfidf_matrix = tfidf_df.values

# =========================
# COSINE MANUAL
# =========================
def cosine_manual(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.sqrt(np.sum(vec1**2))
    norm2 = np.sqrt(np.sum(vec2**2))
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Rekomendasi Musik", layout="wide")

st.title("🎶 Sistem Rekomendasi Musik")
st.markdown("Content-Based Filtering: **Numerik & Metadata **")

selected_song = st.selectbox("🎧 Pilih Lagu:", sorted(df['name'].unique()))

if st.button("Tampilkan Rekomendasi 🚀"):

    idx = df[df['name'] == selected_song].index[0]

    # =====================
    # 🔢 NUMERIK
    # =====================
    numerik = df.select_dtypes(include=['float64'])

    cor_matrix = numerik.corr().abs()
    upper = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))

    to_drop = [col for col in upper.columns if any(upper[col] > 0.85)]
    fitur_terpilih = numerik.drop(columns=to_drop)

    scaler = MinMaxScaler()
    fitur_normal = scaler.fit_transform(fitur_terpilih)

    vektor_input = fitur_normal[idx].reshape(1, -1)
    sim_num = cosine_similarity(vektor_input, fitur_normal)[0]

    df['similarity_num'] = sim_num

    rekom_num = df[df.index != idx] \
        .sort_values(by='similarity_num', ascending=False) \
        .head(5)

    # =====================
    # 📝 METADATA MANUAL
    # =====================
    sim_text = []
    for i in range(len(tfidf_matrix)):
        sim = cosine_manual(tfidf_matrix[idx], tfidf_matrix[i])
        sim_text.append(sim)

    sim_text = np.array(sim_text)
    df['similarity_text'] = sim_text

    rekom_text = df[df.index != idx] \
        .sort_values(by='similarity_text', ascending=False) \
        .head(5)

    # =====================
    # OUTPUT
    # =====================
    st.subheader(f"🎧 Rekomendasi untuk: {selected_song}")

    # NUMERIK
    st.markdown("### 🔢 Rekomendasi Numerik")
    tabel_num = rekom_num[['name', 'artists', 'similarity_num']].copy()
    tabel_num.insert(0, 'Ranking', range(1, len(tabel_num)+1))
    st.dataframe(tabel_num)

    # METADATA
    st.markdown("### 📝 Rekomendasi Metadata ")
    tabel_text = rekom_text[['name', 'artists', 'similarity_text']].copy()
    tabel_text.insert(0, 'Ranking', range(1, len(tabel_text)+1))
    st.dataframe(tabel_text)

    # =====================
    # DETAIL NUMERIK
    # =====================
    st.markdown("---")
    st.subheader("🔢 Detail Numerik")

    st.markdown("### Korelasi (PCC)")
    fig, ax = plt.subplots()
    sns.heatmap(cor_matrix, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.markdown("### Normalisasi")
    df_norm = pd.DataFrame(fitur_normal, columns=fitur_terpilih.columns)
    st.dataframe(df_norm)

    st.markdown("### Cosine Similarity Numerik")
    sim_df = pd.DataFrame({
        'Judul Lagu': df['name'],
        'Similarity': sim_num
    }).sort_values(by='Similarity', ascending=False)
    st.dataframe(sim_df)

    # =====================
    # DETAIL METADATA
    # =====================
    st.markdown("---")
    st.subheader("📝 Detail Metadata ")

    st.markdown("### Cleaning")
    st.dataframe(df[['metadata', 'metadata_clean']])

    st.markdown("### Tokenisasi")
    st.dataframe(df[['metadata_clean', 'tokens']])

    st.markdown("### TF Matrix")
    st.dataframe(tf_df)

    st.markdown("### IDF")
    st.dataframe(idf_series)

    st.markdown("### TF-IDF Matrix")
    st.dataframe(tfidf_df)

    st.markdown("### Cosine Similarity Metadata")
    sim_text_df = pd.DataFrame({
        'Judul Lagu': df['name'],
        'Similarity': sim_text
    }).sort_values(by='Similarity', ascending=False)

    st.dataframe(sim_text_df)
