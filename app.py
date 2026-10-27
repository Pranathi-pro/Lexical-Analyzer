import re
import streamlit as st
from classifier import ContextAwareTokenClassifier

st.set_page_config(page_title="Intelligent Lexical Analyzer", page_icon="🔎", layout="wide")
st.title("🔎 Intelligent Lexical Analyzer")
st.caption("Machine Learning for Context-Aware Token Classification")

@st.cache_resource
def load_classifier():
    return ContextAwareTokenClassifier()

model = load_classifier()
source = st.text_area("Enter source code", value="int total = price + 25;\nif (total > 100) {\n    print(total);\n}", height=180)

if st.button("Analyze Tokens", type="primary"):
    tokens = model.analyze(source)
    if not tokens:
        st.info("No tokens found. Enter some source code.")
    else:
        st.subheader("Token Classification")
        st.dataframe(tokens, use_container_width=True, hide_index=True)
        st.subheader("Summary")
        counts = {}
        for row in tokens:
            counts[row["Predicted Class"]] = counts.get(row["Predicted Class"], 0) + 1
        st.write(counts)
        st.success(f"Analyzed {len(tokens)} tokens.")
st.markdown("---")
st.caption("Educational prototype: the model is trained on a small illustrative dataset. It demonstrates contextual classification, not production-grade compiler analysis.")
