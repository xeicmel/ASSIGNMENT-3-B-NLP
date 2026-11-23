
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(
    page_title="TikTok Politeness Detector",
    page_icon="💬",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f1020, #19132c, #101827);
    color: white;
}

.block-container {
    padding-top: 3rem;
    max-width: 850px;
}

h1, h2, h3, p, label {
    color: white !important;
}

.hero {
    padding: 30px;
    border-radius: 22px;
    background: linear-gradient(135deg, #7c3aed, #db2777);
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 8px 30px rgba(124,58,237,0.25);
}

.hero h1 {
    font-size: 36px;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero p {
    font-size: 16px;
    color: #f3e8ff !important;
}

.info-card {
    background: #1f2337;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #353854;
    text-align: center;
}

.info-number {
    font-size: 28px;
    font-weight: bold;
    color: #c4b5fd;
}

.info-label {
    color: #cbd5e1;
    font-size: 14px;
}

.result-card {
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    margin-top: 20px;
}

div.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    background: linear-gradient(90deg, #8b5cf6, #ec4899);
    color: white;
    font-weight: bold;
    border: none;
    font-size: 17px;
}

div.stButton > button:hover {
    border: 1px solid white;
    color: white;
}

textarea {
    border-radius: 12px !important;
}

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 13px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# Load dataset
df = pd.read_csv("dataset_detecting_politeness (1).csv")

data = df[["Original TikTok Comment", "Label"]].copy()

X = data["Original TikTok Comment"]
y = data["Label"]

# Train model
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_tfidf, y)

# Header
st.markdown("""
<div class="hero">
    <h1>💬 TikTok Politeness Detector</h1>
    <p>AI-powered comment classification for Polite and Impolite language</p>
</div>
""", unsafe_allow_html=True)

# Information cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">549</div>
        <div class="info-label">Comments</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">2</div>
        <div class="info-label">Classification Labels</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">89.09%</div>
        <div class="info-label">Test Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.subheader("🔍 Analyse a Comment")
st.write("Enter a TikTok comment below to analyse its politeness.")

comment = st.text_area(
    "Your comment",
    placeholder="Type or paste a TikTok comment here...",
    height=150
)

if st.button("✨ Analyse Comment"):
    if comment.strip():

        comment_tfidf = vectorizer.transform([comment])
        prediction = model.predict(comment_tfidf)[0]
        probabilities = model.predict_proba(comment_tfidf)[0]

        confidence = max(probabilities) * 100

        if prediction == "Polite":
            st.markdown("""
            <div class="result-card" style="background:#123b32; border:1px solid #34d399;">
                <h2 style="color:#6ee7b7 !important;">😊 POLITE</h2>
                <p style="color:#d1fae5 !important;">This comment is classified as polite.</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="result-card" style="background:#4a2029; border:1px solid #fb7185;">
                <h2 style="color:#fda4af !important;">⚠️ IMPOLITE</h2>
                <p style="color:#ffe4e6 !important;">This comment is classified as impolite.</p>
            </div>
            """, unsafe_allow_html=True)

        st.metric("Model Confidence", f"{confidence:.2f}%")

        st.caption(
            "Model confidence represents the model's predicted probability, "
            "not a guarantee that the classification is correct."
        )

    else:
        st.warning("Please enter a comment first.")

# Footer
st.markdown("""
<div class="footer">
    NLP Project | Malaysian TikTok Comment Politeness Classification
    <br>
    TF-IDF + Multinomial Naive Bayes
</div>
""", unsafe_allow_html=True)
