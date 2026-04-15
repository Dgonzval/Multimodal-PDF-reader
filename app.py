"""
app_v3.py
----------
Streamlit frontend for the production-grade multimodal PDF RAG pipeline.

Pipeline (per query):
  1. User uploads PDF.
  2. On first run → build_multimodal_retriever():
       • extract text chunks   → embed in Chroma
       • extract images        → GPT-5 summarizes each image → embed summary in Chroma
       • store raw content (text bytes / image data-URLs) in InMemoryByteStore
  3. User types a question.
  4. retrieve_and_answer() → MultiVectorRetriever fetches relevant raw docs
     (may be text chunks, raw images, or both) → single GPT-5 multimodal call.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from auxiliary_functions import build_multimodal_retriever, retrieve_and_answer

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal PDF Analyzer — V3",
    page_icon="📄",
    layout="wide",
)

with st.sidebar:
    st.title("📄 PDF Analyzer V3")
    st.markdown(
        """
        ## Architecture
        **Multi-Vector Retriever (production pattern)**
        - GPT-5 summarises every image during ingestion
        - Text chunks AND image summaries are embedded in Chroma
        - Retrieval fetches the *original* content linked to top summaries
        - A single multimodal GPT-5 call synthesises the final answer

        ---
        > Originals: `app.py`, `auxiliary_functions.py`  
        > Intermediate: `app_v2.py`, `auxiliary_functions_v2.py`
        """
    )

# ── Session state ─────────────────────────────────────────────────────────────
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


def main():
    st.header("🤖 AI Assistant — Multimodal PDF RAG")

    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        store_name = pdf.name[:-4]

        # Only re-index if the PDF changed
        if st.session_state.pdf_name != pdf.name:
            pdf_path = f"temp_{pdf.name}"
            with open(pdf_path, "wb") as f:
                f.write(pdf.getbuffer())

            with st.spinner(
                "🔍 Ingesting PDF: extracting text + images, summarising images with GPT-5, building index… (this may take a minute)"
            ):
                st.session_state.retriever = build_multimodal_retriever(
                    pdf_path=pdf_path,
                    pdf_file_obj=pdf,
                    store_name=store_name,
                )
                st.session_state.pdf_name = pdf.name

            st.success("✅ PDF ingested! Text chunks and images are indexed.")

        query = st.text_input("Ask a question about your PDF:")

        if query and st.session_state.retriever:
            with st.spinner("🧠 Retrieving context and generating answer…"):
                answer = retrieve_and_answer(query, st.session_state.retriever)
            st.markdown("### Answer")
            st.write(answer)

    else:
        st.session_state.retriever = None
        st.session_state.pdf_name = None

        st.markdown("#### No PDF loaded — General Chat mode")
        prompt = st.text_input("Ask any question:")

        if prompt:
            with st.spinner("Thinking…"):
                llm = ChatOpenAI(model="gpt-5", temperature=0)
                response = llm.invoke(prompt)
            st.write(response.content)


if __name__ == "__main__":
    main()
