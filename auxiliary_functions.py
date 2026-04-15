"""
auxiliary_functions_v3.py
--------------------------
Production-grade multimodal PDF RAG helpers.

Architecture (Multi-Vector Retriever pattern):
  INGESTION:
    1. Extract text chunks (RecursiveCharacterTextSplitter).
    2. Extract images from the PDF as base64 bytes.
    3. Ask GPT-4o to summarize every image → produces a text summary per image.
    4. Embed BOTH text-chunk summaries AND image summaries → store in Chroma.
    5. Store the *original* content (raw text chunk / raw base64 image) in an
       InMemoryByteStore, keyed by a UUID that links back to the embedding.

  RETRIEVAL:
    6. LangChain MultiVectorRetriever: similarity-search the vector store
       (summaries), then auto-fetches the linked raw content from the docstore.

  SYNTHESIS:
    7. Build a single multimodal GPT-4o message that contains:
         - all retrieved text chunks as plain text
         - all retrieved images as base64 image_url entries
       → one call that reasons over both modalities simultaneously.
"""

from __future__ import annotations

import base64
import io
import os
import re
import uuid
from typing import Any, List, Tuple

import pymupdf  # PyMuPDF
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.stores import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from PyPDF2 import PdfReader

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
raw_client = OpenAI(api_key=openai_api_key)

# ── Constants ────────────────────────────────────────────────────────────────
ID_KEY = "doc_id"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4  # number of elements to retrieve

# ── Image extraction ─────────────────────────────────────────────────────────

def find_closest_text_block(text_blocks: List[Any], x0: float, y0: float, x1: float, y1: float) -> str:
    """Finds the text block that is geographically closest to the image bounding box."""
    min_distance = float('inf')
    closest_text = ""
    image_center_x = (x0 + x1) / 2
    image_center_y = (y0 + y1) / 2

    for block in text_blocks:
        # PyMuPDF block format: (x0, y0, x1, y1, "lines in block", block_no, block_type)
        if len(block) < 5:
            continue
        bx0, by0, bx1, by1, text = block[:5]
        if not isinstance(text, str):
            continue

        block_center_x = (bx0 + bx1) / 2
        block_center_y = (by0 + by1) / 2

        # Calculate Euclidean distance between centers
        distance = ((image_center_x - block_center_x) ** 2 + (image_center_y - block_center_y) ** 2) ** 0.5

        # Filter blocks that are vertically aligned with the image
        if by0 < y1 and by1 > y0:
            if distance < min_distance:
                min_distance = distance
                closest_text = text

    return closest_text


def extract_images_from_pdf(pdf_path: str) -> List[Tuple[int, str, str, str]]:
    """
    Returns a list of (page_number, image_ext, base64_string, closest_text) for every
    reasonably-sized image found in the PDF.
    """
    doc = pymupdf.open(pdf_path)
    images = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text_blocks = page.get_text("blocks")

        for img in page.get_images(full=True):
            xref, name = img[0], img[7]
            base_image = doc.extract_image(xref)
            ext = base_image["ext"]
            bbox = page.get_image_bbox(name)
            x0, y0, x1, y1 = bbox

            # Skip decorative / very small images
            if (x1 - x0) < 50 or (y1 - y0) < 50 or y0 < 100:
                continue

            closest_text = find_closest_text_block(text_blocks, x0, y0, x1, y1)
            b64 = base64.b64encode(base_image["image"]).decode()
            images.append((page_num + 1, ext, b64, closest_text.strip()))

    doc.close()
    return images


def extract_text_from_pdf(pdf: Any) -> str:
    """Extract all text from an uploaded Streamlit PDF file object or path."""
    reader = PdfReader(pdf)
    return "".join(page.extract_text() or "" for page in reader.pages)


# ── Summarisation (key step that replaces "closest text block") ──────────────

def summarize_image(b64_image: str, image_ext: str, page: int, closest_text: str) -> str:
    """
    Ask GPT-5 to produce a rich textual description of an image.
    This summary is what gets embedded – not the raw pixel data.
    """
    data_url = f"data:image/{image_ext};base64,{b64_image}"
    
    prompt_text = (
        f"(This image is on page {page} of a PDF document.)\n"
        "Provide a detailed, factual description of this image. "
        "Include any numbers, labels, legends, axes, trends, or "
        "key observations visible in the graphic. "
        "Your description will be used as the index for retrieval "
        "so be precise and comprehensive."
    )
    
    if closest_text:
        prompt_text += (
            f"\n\nContext from the document right next to this image:\n"
            f"\"\"\"{closest_text}\"\"\"\n"
            f"Please incorporate context from this text into your description "
            f"if it explains the image."
        )

    response = raw_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        max_completion_tokens=512,
    )
    return response.choices[0].message.content.strip()


# ── Multi-Vector retriever builder ────────────────────────────────────────────

def build_multimodal_retriever(
    pdf_path: str,
    pdf_file_obj: Any,
    store_name: str,
    persist_dir: str = "Data",
) -> MultiVectorRetriever:
    """
    Builds (or reloads) a LangChain MultiVectorRetriever that stores:
      • text-chunk summaries  → embed in Chroma
      • image summaries       → embed in Chroma
    linked to their original content in an InMemoryByteStore.

    Returns the retriever ready for similarity-retrieval.
    """
    # ── Sanitize collection name for Chroma ──────────────────────────────────
    print(f"\n[BACKEND LOG] Building/Loading MultiVectorRetriever for '{store_name}'...")
    # Chroma requires: 3-512 chars, [a-zA-Z0-9._-], start/end with [a-zA-Z0-9]
    sanitized_name = re.sub(r"[^a-zA-Z0-9._-]", "_", store_name)
    # Ensure it doesn't start with a non-alphanumeric (though _v4 suffix handles end)
    if not sanitized_name[0].isalnum():
        sanitized_name = "pdf_" + sanitized_name

    embedding_model = OpenAIEmbeddings()
    # Using _v4 to force a re-index now that images include closest text
    chroma_dir = f"{persist_dir}/{sanitized_name}_v4"
    byte_store = InMemoryByteStore()

    vector_store = Chroma(
        collection_name=f"{sanitized_name}_v4",
        embedding_function=embedding_model,
        persist_directory=chroma_dir,
    )

    retriever = MultiVectorRetriever(
        vectorstore=vector_store,
        byte_store=byte_store,
        id_key=ID_KEY,
    )

    # ── If already indexed, just return retriever (text chunks won't persist
    #    in InMemoryByteStore across sessions, so we always re-index for now) ──
    # production systems would use a persistent docstore (Redis, Postgres etc.)

    # ── 1. Text chunks ───────────────────────────────────────────────────────
    print(f"[BACKEND LOG] Extracting and splitting text...")
    raw_text = extract_text_from_pdf(pdf_file_obj)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    text_chunks = splitter.split_text(raw_text)

    chunk_ids = [str(uuid.uuid4()) for _ in text_chunks]
    chunk_docs = [
        Document(page_content=chunk, metadata={ID_KEY: chunk_ids[i]})
        for i, chunk in enumerate(text_chunks)
    ]

    print(f"[BACKEND LOG] Inserting {len(chunk_docs)} text chunks into VectorStore and DocStore...")
    # For text we embed the chunk itself (it is already a good "summary")
    retriever.vectorstore.add_documents(chunk_docs)
    retriever.docstore.mset(
        list(zip(chunk_ids, [Document(page_content=c) for c in text_chunks]))
    )

    # ── 2. Images ────────────────────────────────────────────────────────────
    print(f"[BACKEND LOG] Extracting images from PDF...")
    images = extract_images_from_pdf(pdf_path)
    print(f"[BACKEND LOG] Found {len(images)} valid images. Summarizing with Vision LLM...")
    
    for page_num, ext, b64, closest_text in images:
        img_id = str(uuid.uuid4())
        summary = summarize_image(b64, ext, page_num, closest_text)

        # Embed the GPT-4o summary for retrieval
        summary_doc = Document(
            page_content=summary,
            metadata={ID_KEY: img_id, "type": "image", "page": page_num},
        )
        retriever.vectorstore.add_documents([summary_doc])

        # Store the raw base64 data as a Document (will be fetched at answer-time)
        img_payload = f"data:image/{ext};base64,{b64}"
        retriever.docstore.mset([(img_id, Document(page_content=img_payload))])

    return retriever


# ── Retrieval + Synthesis ─────────────────────────────────────────────────────

def retrieve_and_answer(
    query: str,
    retriever: MultiVectorRetriever,
) -> str:
    """
    1. Retrieve top-k raw documents (text or image) linked to the query.
       Specifically searches for both text and images to avoid text dominating the results.
    2. Build a single multimodal GPT-4o/5 message and return the answer.
    """
    print(f"\n[BACKEND LOG] --- NEW QUERY ---")
    print(f"[BACKEND LOG] Query: '{query}'")
    print(f"[BACKEND LOG] Executing Chroma similarity searches (text k=10, image k=3)...")
    
    vector_store = retriever.vectorstore
    
    # Search for general content (mostly text). Increased k to 10 for better context.
    general_matches = vector_store.similarity_search(query, k=10)
    
    # Explicitly search for images. Increased k to 3.
    try:
        image_matches = vector_store.similarity_search(query, k=3, filter={"type": "image"})
    except Exception:
        # Fallback if no images were indexed or filter syntax fails
        image_matches = []

    # Get unique doc_ids
    doc_ids = []
    for doc in general_matches + image_matches:
        if ID_KEY in doc.metadata and doc.metadata[ID_KEY] not in doc_ids:
            doc_ids.append(doc.metadata[ID_KEY])

    # Fetch original content from docstore
    print(f"[BACKEND LOG] Matches found: {len(general_matches)} general + {len(image_matches)} image = {len(doc_ids)} unique documents.")
    print(f"[BACKEND LOG] Fetching original document content from InMemoryByteStore...")
    
    raw_docs = retriever.docstore.mget(doc_ids)
    raw_docs = [doc for doc in raw_docs if doc is not None]

    text_parts: List[str] = []
    image_parts: List[dict] = []

    for doc in raw_docs:
        if isinstance(doc, Document):
            decoded = doc.page_content
        elif isinstance(doc, bytes):
            # Fallback for old caches
            decoded = doc.decode()
        else:
            decoded = str(doc)

        if decoded.startswith("data:image"):
            # It's a raw image payload
            image_parts.append(
                {"type": "image_url", "image_url": {"url": decoded}}
            )
        else:
            text_parts.append(decoded)

    # Build the multimodal message content
    content: List[dict] = []

    if text_parts:
        combined_text = "\n\n---\n\n".join(text_parts)
        content.append(
            {
                "type": "text",
                "text": (
                    f"Below are relevant passages retrieved from the document:\n\n"
                    f"{combined_text}\n\n"
                    f"Using these passages (and any images below), answer the following question "
                    f"clearly and concisely:\n\n**{query}**"
                ),
            }
        )
    else:
        content.append(
            {
                "type": "text",
                "text": f"Using the images below, answer the following question:\n\n**{query}**",
            }
        )

    content.extend(image_parts)

    print(f"[BACKEND LOG] Sending {len(text_parts)} text blocks and {len(image_parts)} images to LLM...")
    print(f"[BACKEND LOG] Waiting for LLM response (this might take a few seconds)...")

    llm = ChatOpenAI(model="gpt-4o", temperature=0, max_completion_tokens=1024)
    response = llm.invoke([{"role": "user", "content": content}])
    
    print(f"[BACKEND LOG] Response received successfully!")
    print(f"[BACKEND LOG] Raw LLM Output: {repr(response.content[:100])}... (truncated)\n")
    
    # If response.content is a list (sometimes happens with complex outputs), convert to string
    if isinstance(response.content, list):
        final_answer = " ".join([str(x) for x in response.content])
    else:
        final_answer = str(response.content)
        
    return final_answer.strip()
