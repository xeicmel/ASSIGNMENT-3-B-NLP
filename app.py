
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="TikTok Politeness Detector",
    page_icon="💬",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0b1020, #15132b, #101827);
    color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 900px;
}

/* General text */
h1, h2, h3, p, label {
    color: #f8fafc !important;
}

/* Hero */
.hero {
    padding: 35px 25px;
    border-radius: 24px;
    background: linear-gradient(135deg, #6d28d9, #9333ea, #db2777);
    text-align: center;
    margin-bottom: 28px;
    box-shadow: 0 10px 40px rgba(139, 92, 246, 0.25);
}

.hero h1 {
    font-size: 36px;
    font-weight: 800;
    color: white !important;
    margin-bottom: 10px;
}

.hero p {
    font-size: 16px;
    color: #f3e8ff !important;
}

/* Information cards */
.info-card {
    background: rgba(30, 41, 59, 0.85);
    padding: 22px 15px;
    border-radius: 18px;
    border: 1px solid rgba(148, 163, 184, 0.2);
    text-align: center;
    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.15);
}

.info-number {
    font-size: 30px;
    font-weight: 800;
    color: #c4b5fd !important;
}

.info-label {
    color: #cbd5e1 !important;
    font-size: 14px;
}

/* TEXT AREA - DARK BACKGROUND + WHITE TEXT */
.stTextArea textarea,
[data-testid="stTextArea"] textarea,
div[data-baseweb="textarea"] textarea {
    background-color: #1e293b !important;
    background: #1e293b !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    caret-color: #ffffff !important;
    border: 1px solid #a78bfa !important;
    border-radius: 14px !important;
    opacity: 1 !important;
}

/* Placeholder */
.stTextArea textarea::placeholder,
[data-testid="stTextArea"] textarea::placeholder {
    color: #94a3b8 !important;
    -webkit-text-fill-color: #94a3b8 !important;
    opacity: 1 !important;
}

/* Text area focus */
.stTextArea textarea:focus,
[data-testid="stTextArea"] textarea:focus,
div[data-baseweb="textarea"]:focus-within {
    border: 1px solid #c084fc !important;
    box-shadow: 0 0 10px rgba(192, 132, 252, 0.3) !important;
}

/* Buttons */
div.stButton > button {
    width: 100%;
    border-radius: 14px;
    height: 55px;
    background: linear-gradient(90deg, #7c3aed, #db2777);
    color: white !important;
    font-weight: 800;
    border: none;
    font-size: 17px;
    transition: 0.3s;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(168, 85, 247, 0.3);
    border: 1px solid white;
}

/* Divider */
hr {
    border-color: rgba(148, 163, 184, 0.3);
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8 !important;
    font-size: 13px;
    margin-top: 35px;
    padding: 15px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.65);
    padding: 15px;
    border-radius: 15px;
    border: 1px solid rgba(148, 163, 184, 0.15);
}

</style>
""", unsafe_allow_html=True)


# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("dataset_detecting_politeness (1).csv")

data = df[["Original TikTok Comment", "Label"]].copy()

X = data["Original TikTok Comment"]
y = data["Label"]


# =========================
# TRAIN MODEL
# =========================
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_tfidf, y)


# =========================
# HEADER
# =========================
st.markdown("""
<div class="hero">
    <h1>💬 TikTok Politeness Detector</h1>
    <p>Analyse the politeness of Malaysian TikTok comments using machine learning.</p>
</div>
""", unsafe_allow_html=True)


# =========================
# INFORMATION CARDS
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">549</div>
        <div class="info-label">Dataset Comments</div>
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
        <div class="info-label">Test Accuracy Benchmark</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("---")


# =========================
# COMMENT ANALYSIS
# =========================
st.subheader("🔍 Analyse a Comment")

st.write("Enter a TikTok comment below to analyse its politeness.")

comment = st.text_area(
    "Your comment",
    placeholder="Type or paste a TikTok comment here...",
    height=160,
    key="comment_input"
)


# =========================
# PREDICTION
# =========================
if st.button("✨ Analyse Comment"):

    if comment.strip():

        comment_tfidf = vectorizer.transform([comment])

        prediction = model.predict(comment_tfidf)[0]

        probabilities = model.predict_proba(comment_tfidf)[0]

        confidence = max(probabilities) * 100

        if prediction == "Polite":

            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #064e3b, #065f46);
                border: 1px solid #34d399;
                border-radius: 20px;
                padding: 25px;
                text-align: center;
                margin-top: 20px;
            ">
                <h2 style="color:#6ee7b7 !important;">
                    😊 POLITE
                </h2>
                <p style="color:#d1fae5 !important;">
                    This comment is classified as polite.
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #7f1d1d, #4c0519);
                border: 1px solid #fb7185;
                border-radius: 20px;
                padding: 25px;
                text-align: center;
                margin-top: 20px;
            ">
                <h2 style="color:#fda4af !important;">
                    ⚠️ IMPOLITE
                </h2>
                <p style="color:#ffe4e6 !important;">
                    This comment is classified as impolite.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.metric("🎯 Model Confidence", f"{confidence:.2f}%")

        st.caption(
            "Model confidence represents the model's predicted probability, "
            "not a guarantee that the classification is correct."
        )

    else:

        st.warning("Please enter a comment first.")


# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
    <p>NLP Project | Malaysian TikTok Comment Politeness Classification</p>
    <p>TF-IDF + Multinomial Naive Bayes</p>
</div>
""", unsafe_allow_html=True)
