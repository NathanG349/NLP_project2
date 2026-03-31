import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Set up the page layout
st.set_page_config(page_title="Insurance Review AI", layout="wide")
st.title("Comprehensive Insurance Review Analyzer")
st.write("Navigate through the tabs below to explore different AI models")

# 1. Load the saved ML models (Prediction & Explanation)
@st.cache_resource
def load_ml_models():
    vec = joblib.load('tfidf_vectorizer.pkl')
    mod = joblib.load('logistic_model.pkl')
    return vec, mod

# 2. Load the Hugging Face pipelines (Summary, QA, RAG)
@st.cache_resource
def load_hf_pipelines():
    # We use lightweight models so they run smoothly on your local CPU
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    return summarizer, qa_model

# 3. Load the dataset (Information Retrieval & RAG)
@st.cache_data
def load_data():
    try:
        # We load a sample to keep the memory usage low
        df = pd.read_csv('cleaned_reviews_backup.csv')
        df = df.dropna(subset=['avis_en'])
        return df.head(2000)
    except Exception:
        return pd.DataFrame()

# Initialize everything
try:
    vectorizer, model = load_ml_models()
    hf_summarizer, hf_qa = load_hf_pipelines()
    df_reviews = load_data()
except Exception as e:
    st.error(f"System Error: {e}")
# Create the layout tabs
tab1, tab2, tab3 = st.tabs(["Prediction & Explanation", "Summary & QA", "Information Retrieval & RAG"])

# TAB 1: PREDICTION (2 points) & EXPLANATION (3 points)
with tab1:
    st.header("Predict & Explain Star Ratings")
    user_input = st.text_area("Customer Review:", height=100, key="pred_input")
    
    if st.button("Predict Star Rating"):
        if user_input.strip() == "":
            st.warning("Please enter a review first!")
        else:
            clean_text = re.sub(r'[^\w\s]', '', user_input.lower())
            vectorized_text = vectorizer.transform([clean_text])
            prediction = model.predict(vectorized_text)[0]
            
            st.subheader("Prediction Result")
            st.write(f"This review looks like a **{int(prediction)} Star** rating!")
            st.write("⭐" * int(prediction))
            
            # EXPLANATION Logic
            st.subheader("Why did the model choose this?")
            st.write("Here are the words from your review that influenced the mathematical model the most:")
            
            feature_names = vectorizer.get_feature_names_out()
            dense_vector = vectorized_text.todense().tolist()[0]
            
            # Map words to their TF-IDF scores
            word_scores = [(feature_names[i], score) for i, score in enumerate(dense_vector) if score > 0]
            word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
            
            if word_scores:
                for word, score in word_scores[:5]:
                    st.write(f"- **{word}** (Importance score: {score:.2f})")
            else:
                st.write("No strong keywords were recognized in the vocabulary.")

# TAB 2: SUMMARY (2 points) & QA (3 points)
with tab2:
    st.header("Review Summarization & Question Answering")
    st.write("Paste a very long review below to summarize it, or ask a specific question about it.")
    
    long_review = st.text_area("Long Review Text:", height=150, key="sum_input")
    question = st.text_input("Ask a question about this review (e.g., 'What was the problem?'):")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Summarize Review"):
            if len(long_review.split()) < 20:
                st.warning("Please enter a longer review to generate a summary.")
            else:
                with st.spinner("Generating summary"):
                    summary = hf_summarizer(long_review, max_length=40, min_length=10, do_sample=False)
                    st.success("Summary generated!")
                    st.write(summary[0]['summary_text'])
                    
    with col2:
        if st.button("Answer Question"):
            if not long_review or not question:
                st.warning("Please provide both a review and a question.")
            else:
                with st.spinner("Finding the answer"):
                    answer = hf_qa(question=question, context=long_review)
                    st.success("Answer found!")
                    st.write(f"**Answer:** {answer['answer']}")
                    st.write(f"*(Confidence score: {answer['score']:.2f})*")

# TAB 3: INFORMATION RETRIEVAL (3 points) & RAG (3 points)
with tab3:
    st.header("Information Retrieval & RAG")
    st.write("Search the database of existing reviews and ask AI to synthesize the findings.")
    
    search_query = st.text_input("Search the database (e.g., 'car windshield glass replacement'):")
    rag_question = st.text_input("Ask the AI a question based on these results (e.g., 'How long does replacement take?'):")
    
    if st.button("Search Database & Generate Answer"):
        if df_reviews.empty:
            st.error("Database not loaded. Please ensure cleaned_reviews_backup.csv is in the folder.")
        elif not search_query:
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching and Analyzing"):
                # INFORMATION RETRIEVAL Logic (Using TF-IDF and Cosine Similarity)
                query_vec = vectorizer.transform([search_query])
                doc_vecs = vectorizer.transform(df_reviews['avis_en'].fillna(""))
                
                similarities = cosine_similarity(query_vec, doc_vecs).flatten()
                top_indices = similarities.argsort()[-3:][::-1]
                
                st.subheader("Top 3 Relevant Reviews Found (Information Retrieval)")
                retrieved_context = ""
                for idx in top_indices:
                    text = df_reviews.iloc[idx]['avis_en']
                    st.write(f"- {text}")
                    retrieved_context += text + " "
                
                # RAG Logic (Combining Retrieved context with Generative QA)
                if rag_question:
                    st.subheader("RAG AI Answer")
                    rag_answer = hf_qa(question=rag_question, context=retrieved_context)
                    st.write(f"**Generated Answer:** {rag_answer['answer']}")
                    st.write("This answer was generated exclusively using the retrieved database context.")