import os
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
st.set_page_config(page_title="BizInsight AI", layout="wide")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from database import insert_feedback, fetch_feedback, clear_data
from openai import OpenAI
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from clustering.run_clustering import run_pipeline
import re
from clustering.vectorize import load_model
import matplotlib.pyplot as plt

# ---------- Chimera AI Client ----------

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable not set. Please create a .env file with your API key.")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

vader_analyzer = SentimentIntensityAnalyzer()

st.title("📊 BizInsight AI")
st.caption("AI-powered customer intelligence platform for business growth")

tabs = st.tabs(["📊 Dashboard", "🤖 AI Assistant", "📂 Data Upload", "⚙ Controls"])

# ---------- Core Functions ----------
# Vader sentiment analysis (replacing TextBlob for better performance on short reviews)
def get_sentiment(text):
    """Improved sentiment using VADER"""
    scores = vader_analyzer.polarity_scores(text)
    return scores['compound']

# Text cleaning for sentiment analysis (minimal cleaning to preserve sentiment)
def clean_text_for_sentiment(text):
    text = text.lower()
    # Remove ALL digits
    text = re.sub(r'\d+', '', text)
    # Remove # symbol
    text = re.sub(r'#', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def ask_ai(question, reviews):
    context = "\n".join(reviews[:40])

    prompt = f"""
    You are a professional business analyst.

    Customer feedback:
    {context}

    Analyze patterns, root problems and give improvement suggestions.

    Question:
    {question}
    """

    response = client.chat.completions.create(
        model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", # Fixed model name
        messages=[
            {"role": "system", "content": "You provide business intelligence insights."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    return response.choices[0].message.content

# ================= DATA UPLOAD =================

with tabs[2]:
    st.subheader("📂 Upload Customer Reviews")

    uploaded_file = st.file_uploader("Upload CSV with review column", type="csv")

    if uploaded_file:
        clear_data()  # Clear existing data before adding new

        df = pd.read_csv(uploaded_file)
        # Fixed dataframe width argument
        st.dataframe(df, width="stretch")

        # Clean text and get sentiment for each review
        df["cleaned_for_sentiment"] = df["review"].apply(clean_text_for_sentiment)
        df["sentiment"] = df["cleaned_for_sentiment"].apply(get_sentiment)

        for _, row in df.iterrows():
            insert_feedback(row["cleaned_for_sentiment"], row["sentiment"])

        st.success("Feedback successfully added!")  

# ================= LOAD STORED DATA =================

data = fetch_feedback()

if data:
    df = pd.DataFrame(data, columns=["review", "sentiment", "date"])
    df["date"] = pd.to_datetime(df["date"])

    positive = (df["sentiment"] > 0).sum()
    negative = (df["sentiment"] < 0).sum()

    trend = df.groupby(df["date"].dt.date)["sentiment"].mean()

    vectorizer = CountVectorizer(stop_words="english", max_features=10)
    X = vectorizer.fit_transform(df["review"])
    keywords = vectorizer.get_feature_names_out()

    # ================= DASHBOARD =================

    with tabs[0]:
        st.subheader("📈 Business Health Overview")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Reviews", len(df))
        c2.metric("Positive", positive)
        c3.metric("Negative", negative)

        st.markdown("---")
        
        # ========== SMART COMPLAINT CLUSTERING ==========
        st.subheader("🔍 Smart Complaint Clustering")

        embedding_model = load_model()
        
        if st.button("Find Complaint Clusters"):
            with st.spinner("Analyzing negative reviews... (This may take a moment to download the embedding model on first run)"):
                # Get only negative reviews (sentiment < 0)
                negative_reviews = df[df["sentiment"] < 0]["review"].tolist()
                
                if len(negative_reviews) < 10:
                    st.warning(f"Only {len(negative_reviews)} negative reviews found. Need at least 10 for meaningful clustering.")
                else:
                    result = run_pipeline(
                        negative_reviews, 
                        embedding_model, 
                        min_topic_size=25, # Adjusted for better cluster quality with small datasets 
                        similarity_threshold=0.4, # Tuned for better merging of similar clusters, can be adjusted based on dataset size and diversity
                        verbose=True # Keep verbose on to show progress
                    ) 
                    
                    if result["success"]:
                        st.success(f"✅ Found {result['n_clusters']} complaint clusters from {result['total_negative_reviews']} negative reviews")
                        
                        # Display clusters
                        for cluster in result["clusters"]:
                            with st.expander(f"📌 {cluster['name']} ({cluster['percentage']:.1f}%) - {cluster['count']} reviews"):
                                st.write("**Some Reviews:**")
                                for i, review in enumerate(cluster.get('example_reviews', [cluster['sample_review']])[:3]):
                                    st.write(f"  {i+1}. \"{review}\"")
                    else:
                        st.error(result["message"])

        st.markdown("---")
        col1, col2 = st.columns([2,1])

        with col1:
            st.subheader("Customer Satisfaction Trend")
            st.line_chart(trend)

        with col2:
            fig, ax = plt.subplots()
            ax.bar(["Positive", "Negative"], [positive, negative])
            st.pyplot(fig)

        st.markdown("---")

        st.subheader("Top Customer Issues")
        st.write(list(keywords))

    # ================= AI ASSISTANT =================

    with tabs[1]:
        st.subheader("🤖 AI Business Consultant")
        st.write("Ask questions about customer experience and improvement strategy.")

        user_q = st.text_input("Type your business question here")

        if user_q:
            with st.spinner("Analyzing feedback..."):
                st.success(ask_ai(user_q, df["review"].tolist()))


    # ================= CONTROLS =================

    with tabs[3]:
        st.subheader("⚙ System Controls")

        if st.button("🗑 Clear all stored feedback"):
            clear_data()
            st.success("All data removed successfully.")
            st.rerun() # Refresh the page to show empty state

        st.warning("This action cannot be undone.")

else:
    st.info("Upload feedback to start building insights.")