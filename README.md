# Multimodal PDF Reader

A production-grade RAG (Retrieval-Augmented Generation) application that lets you chat with any PDF — understanding both its **text and images** — powered by GPT-4o and LangChain's Multi-Vector Retriever pattern.

---

## How it works

```
PDF Upload
   │
   ├─── Text Extraction  ──► Chunked & embedded in Chroma
   │
   └─── Image Extraction ──► GPT-4o summarizes each image
                                       │
                                       └──► Summary embedded in Chroma
                                            (raw image stored in DocStore)
                                                       │
                                            User Query ▼
                                   Similarity search (text + images)
                                                       │
                                   Fetch original content from DocStore
                                                       │
                                   Single multimodal GPT-4o call
                                   (text chunks + raw images together)
                                                       │
                                               Final Answer ✓
```

### Key architectural decisions

| Decision | Why |
|---|---|
| **Multi-Vector Retriever** | Embed *summaries* for retrieval, store *originals* for synthesis — best of both worlds |
| **GPT-4o image summaries** | Images become searchable without ever converting pixels to embedding vectors directly |
| **Chroma** persistence | Vector index survives page reloads; only raw content (InMemoryByteStore) needs re-indexing |
| **Separate text + image searches** | Prevents text results from drowning out relevant images at retrieval time |

---

## Stack

- **Frontend:** Streamlit
- **LLM / Vision:** GPT-4o (OpenAI)
- **Embeddings:** OpenAI `text-embedding-ada-002`
- **Vector Store:** Chroma (persisted locally)
- **RAG Framework:** LangChain — `MultiVectorRetriever`
- **PDF parsing:** PyMuPDF (images) + PyPDF2 (text)

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Dgonzval/Multimodal-PDF-reader.git
cd Multimodal-PDF-reader
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your API key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## Usage

1. Open the app in your browser (`http://localhost:8501`)
2. Upload any PDF using the file uploader
3. Wait for ingestion — the app extracts text, finds images, and asks GPT-4o to describe each one
4. Type any question about the PDF and get a grounded, multimodal answer

> **No PDF loaded?** The app falls back to a general chat mode powered by GPT-4o.

---

## Project structure

```
├── app.py                  # Streamlit UI
├── auxiliary_functions.py  # RAG pipeline (ingestion + retrieval + synthesis)
├── prompt.py               # Prompt templates
├── requirements.txt        # Python dependencies
└── .env                    # API keys (not committed)
```

---

## Notes

- The `InMemoryByteStore` (raw document store) is rebuilt on every session — a production deployment would swap this for Redis or PostgreSQL.
- Images smaller than 50×50 px or positioned in page headers are filtered out to skip decorative elements.
- Chroma collections are versioned (`_v4` suffix) to force re-indexing when the pipeline changes.
