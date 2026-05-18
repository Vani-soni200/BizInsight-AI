# 📊 BizInsight AI

BizInsight AI is an AI-powered customer feedback analytics platform that helps businesses understand customer sentiment, identify key issues, track satisfaction trends, and receive intelligent improvement suggestions.

Built as a real-world business intelligence tool using Python, Streamlit, and advanced AI models.

---

## 🚀 Features

- Upload customer feedback CSV files  
- Automatic sentiment analysis  
- **Smart complaint clustering** – automatically groups negative reviews into business‑relevant categories (Payment, Delivery, Technical, Account, Product Quality, Customer Service, etc.)  
- Trend tracking over time  
- Top issue detection  
- AI-powered business assistant  
- Persistent data storage  
- Clean dashboard UI  

---

## 🧠 AI Capabilities

- Understands customer pain points  
- Identifies repeating issues  
- Suggests improvement actions  
- Provides business-style insights  
- **Unsupervised topic modelling** – finds hidden complaint patterns without manual labelling  

---

## 🛠 Tech Stack

- Python  
- Streamlit  
- Pandas  
- Matplotlib  
- Scikit-learn  
- **VADER** (sentiment analysis)  
- **BERTopic** (clustering & topic extraction)  
- **HDBSCAN** + **UMAP** (density‑based clustering)  
- **Sentence‑Transformers** (embedding model, fine‑tunable)  
- OpenRouter / nvidia AI
- SQLite  

---

## 🔍 Smart Complaint Clustering (Core Feature)

This module automatically groups customer complaints into meaningful clusters, names them (e.g., `"Payment Issues"`, `"Delivery Issues"`).

### How it works

1. **Preprocessing** – removes numbers, `#`, punctuation (keeps apostrophes). No stopword removal – preserves meaning.  
2. **Embedding** – converts reviews into vectors using a Sentence‑Transformer (`all-mpnet-base-v2` or fine‑tuned model).  
3. **Dimensionality reduction** – UMAP (5 components, cosine distance).  
4. **Clustering** – HDBSCAN (density‑based, automatically marks noise as outliers).  
5. **Topic extraction** – BERTopic extracts c‑TF‑IDF words.  
6. **Category mapping** – each cluster is compared to 11 predefined category descriptions (Payment, Delivery, Technical, Account, Product Quality, Customer Service, Shipping Damage, Subscription, Checkout, Return/Refund). If similarity ≥ threshold, the cluster gets a standard name; otherwise it receives a dynamic name generated from the two most frequent content words + suffix (`Issues` / `Error` / `Delay`).  
7. **Merge duplicates** – clusters with the same name are combined.

### How to use it in the dashboard

1. Upload a CSV with a `review` column.  
2. Go to the **Dashboard** tab and click **“Find Complaint Clusters”**.  
3. Wait for the analysis (first run loads the embedding model).  
4. Expandable clusters – each shows name, number of reviews, some complaints.

## 📂 Project Structure

bizinsight-ai/
├── app.py
├── database.py
├── clustering/
│   ├── run_clustering.py
│   ├── preprocess.py
│   ├── vectorize.py
├── models/finetuned_complaint_model_final
├── data / reviews.csv
├── tests / 
    ├── product_reviews_1000.csv
    ├── test1.csv
    ├── test2.csv

---

## 📥 How to Run Locally

### Install dependencies

pip install -r requirements.txt

streamlit run app.py


---

### Requirements.txt
streamlit
pandas
matplotlib
scikit-learn
vaderSentiment
bertopic
hdbscan
umap-learn
sentence-transformers
python-dotenv

## 📄 CSV Format

Your CSV file must contain a column named `review`.


## 📈 Example Use Cases

- E-commerce customer experience analysis  
- Service quality monitoring  
- Product feedback insights  
- Business performance improvement  

---

## 🏆 Why BizInsight AI?

Manually analyzing customer feedback is time-consuming and error-prone.  
BizInsight AI converts raw reviews into actionable business intelligence using AI.

---

## 📌 Future Enhancements

- Multi-business login system  
- Automated report generation (PDF)   
- Trend alert system  
- Website integration chatbot  

---

## 👨‍💻 Author

Built by **Prateek Singh**  
BTech Student | AI & Software Development Enthusiast

---

⭐ If you like this project, consider giving it a star!
