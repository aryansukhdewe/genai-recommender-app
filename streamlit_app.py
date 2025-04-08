
import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load the model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Upload catalog
st.title("SHL GenAI Assessment Recommender")
uploaded_file = st.file_uploader("Upload SHL Catalog JSON", type="json")

def process_catalog(catalog):
    summaries = [item["Summary"] for item in catalog]
    embeddings = model.encode(summaries, convert_to_tensor=True)
    return summaries, embeddings

if uploaded_file:
    catalog = json.load(uploaded_file)
    summaries, embeddings = process_catalog(catalog)

    query = st.text_area("Enter job description or query:")

    if st.button("Get Recommendations"):
        query_embedding = model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, embeddings)[0]
        top_k = np.argsort(-scores)[:3]

        st.subheader("Top 3 Recommendations")
        for i in top_k:
            rec = catalog[i]
            st.markdown(f"**{rec['Assessment Name']}**")
            st.markdown(f"- Summary: {rec['Summary']}")
            st.markdown(f"- Duration: {rec.get('Duration', 'Unknown')}")
            st.markdown(f"- URL: {rec.get('URL', 'N/A')}")
            st.markdown("---")
