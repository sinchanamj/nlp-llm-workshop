# ============================================================
# MULTI-DOCUMENT RAG CHATBOT
# Streamlit + FAISS + Sentence Transformers + Groq
#
# Supports:
# PDF / DOCX / TXT
# ============================================================

import streamlit as st
import faiss
import numpy as np
import pickle
import os
import re
import tempfile

from sentence_transformers import SentenceTransformer
from groq import Groq

from pypdf import PdfReader
from docx import Document


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.kpi-box {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #ddd;
    background-color: #f8f9fa;
    text-align: center;
    margin-bottom: 10px;
}

.kpi-number {
    font-size: 28px;
    font-weight: bold;
}

.kpi-label {
    font-size: 14px;
    color: #666;
}

.source-box {
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #ddd;
    margin-bottom: 12px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. TITLE
# ============================================================

st.title("🤖 RAG Document Assistant")

st.write(
    "Upload documents and ask questions. "
    "The system retrieves relevant document sections "
    "and uses Groq to generate a grounded answer."
)


# ============================================================
# 4. LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


embedding_model = load_embedding_model()


# ============================================================
# 5. SESSION STATE
# ============================================================

# Store uploaded document chunks
if "chunks" not in st.session_state:
    st.session_state.chunks = []


# Store FAISS index
if "index" not in st.session_state:
    st.session_state.index = None


# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Store uploaded file names
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


# ============================================================
# 6. TEXT CLEANING
# ============================================================

def clean_text(text):

    # Remove excessive spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 7. PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(file):

    reader = PdfReader(file)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        1
    ):

        text = page.extract_text()

        if text:

            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


# ============================================================
# 8. DOCX TEXT EXTRACTION
# ============================================================

def extract_docx_text(file):

    doc = Document(file)

    paragraphs = []

    for paragraph_number, paragraph in enumerate(
        doc.paragraphs,
        1
    ):

        text = paragraph.text.strip()

        if text:

            paragraphs.append({
                "page": None,
                "paragraph": paragraph_number,
                "text": text
            })

    return paragraphs


# ============================================================
# 9. TXT TEXT EXTRACTION
# ============================================================

def extract_txt_text(file):

    text = file.read().decode(
        "utf-8",
        errors="ignore"
    )

    return [{
        "page": None,
        "text": text
    }]


# ============================================================
# 10. CHUNKING
# ============================================================

def create_chunks(
    text,
    chunk_size=500,
    overlap=100
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():

            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ============================================================
# 11. PROCESS UPLOADED DOCUMENT
# ============================================================

def process_document(uploaded_file):

    filename = uploaded_file.name

    extension = os.path.splitext(
        filename
    )[1].lower()


    extracted_sections = []


    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if extension == ".pdf":

        extracted_sections = extract_pdf_text(
            uploaded_file
        )


    # --------------------------------------------------------
    # DOCX
    # --------------------------------------------------------

    elif extension == ".docx":

        extracted_sections = extract_docx_text(
            uploaded_file
        )


    # --------------------------------------------------------
    # TXT
    # --------------------------------------------------------

    elif extension == ".txt":

        extracted_sections = extract_txt_text(
            uploaded_file
        )


    else:

        return []


    document_chunks = []


    # --------------------------------------------------------
    # Process every extracted section
    # --------------------------------------------------------

    for section in extracted_sections:

        text = clean_text(
            section["text"]
        )

        if not text:
            continue


        chunks = create_chunks(
            text,
            chunk_size=500,
            overlap=100
        )


        for chunk_number, chunk in enumerate(
            chunks
        ):

            document_chunks.append({

                "chunk_id":
                    len(st.session_state.chunks)
                    + len(document_chunks),

                "filename":
                    filename,

                "page":
                    section.get("page"),

                "paragraph":
                    section.get("paragraph"),

                "chunk_number":
                    chunk_number,

                "text":
                    chunk

            })


    return document_chunks


# ============================================================
# 12. CREATE / UPDATE FAISS INDEX
# ============================================================

def add_chunks_to_index(
    new_chunks
):

    if not new_chunks:

        return


    # --------------------------------------------------------
    # Extract text
    # --------------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in new_chunks
    ]


    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True
    ).astype("float32")


    # --------------------------------------------------------
    # Create FAISS index if necessary
    # --------------------------------------------------------

    dimension = embeddings.shape[1]


    if st.session_state.index is None:

        st.session_state.index = faiss.IndexFlatL2(
            dimension
        )


    # --------------------------------------------------------
    # Add vectors
    # --------------------------------------------------------

    st.session_state.index.add(
        embeddings
    )


    # --------------------------------------------------------
    # Store chunks
    # --------------------------------------------------------

    st.session_state.chunks.extend(
        new_chunks
    )


# ============================================================
# 13. SIDEBAR
# ============================================================

st.sidebar.header("📄 Documents")


uploaded_files = st.sidebar.file_uploader(
    "Upload documents",
    type=[
        "pdf",
        "docx",
        "txt"
    ],
    accept_multiple_files=True
)


# ============================================================
# 14. PROCESS DOCUMENTS
# ============================================================

if uploaded_files:

    for uploaded_file in uploaded_files:

        # Avoid processing same file repeatedly
        if uploaded_file.name not in st.session_state.uploaded_files:

            with st.spinner(
                f"Processing {uploaded_file.name}..."
            ):

                new_chunks = process_document(
                    uploaded_file
                )


                add_chunks_to_index(
                    new_chunks
                )


                st.session_state.uploaded_files.append(
                    uploaded_file.name
                )


            st.sidebar.success(
                f"Loaded: {uploaded_file.name}"
            )


# ============================================================
# 15. DOCUMENT INFORMATION
# ============================================================

st.sidebar.divider()

st.sidebar.subheader(
    "📚 Loaded Documents"
)


if st.session_state.uploaded_files:

    for filename in st.session_state.uploaded_files:

        st.sidebar.write(
            f"📄 {filename}"
        )

else:

    st.sidebar.info(
        "No documents uploaded yet."
    )


# ============================================================
# 16. GROQ API KEY
# ============================================================

st.sidebar.divider()

st.sidebar.header(
    "⚙️ Groq Configuration"
)


groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password"
)


MODEL = "openai/gpt-oss-20b"


# ============================================================
# 17. RETRIEVE RELEVANT DOCUMENTS
# ============================================================

def retrieve_documents(
    query,
    top_k=5
):

    if st.session_state.index is None:

        return []


    # --------------------------------------------------------
    # Convert query into embedding
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")


    # --------------------------------------------------------
    # FAISS search
    # --------------------------------------------------------

    distances, indices = (
        st.session_state.index.search(
            query_embedding,
            top_k
        )
    )


    results = []


    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx < 0:

            continue


        if idx >= len(
            st.session_state.chunks
        ):

            continue


        chunk = st.session_state.chunks[
            idx
        ]


        results.append({

            "chunk_id": int(idx),

            "distance":
                float(distance),

            "filename":
                chunk["filename"],

            "page":
                chunk.get("page"),

            "paragraph":
                chunk.get("paragraph"),

            "text":
                chunk["text"]

        })


    return results


# ============================================================
# 18. GENERATE RAG ANSWER
# ============================================================

def generate_rag_answer(
    question,
    retrieved_results
):


    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = ""


    for i, result in enumerate(
        retrieved_results,
        1
    ):

        context += f"""

SOURCE {i}

Document:
{result['filename']}

Page:
{result['page']}

Paragraph:
{result['paragraph']}

Chunk ID:
{result['chunk_id']}

Content:
{result['text']}

--------------------------------------------------
"""


    # ========================================================
    # PROMPT
    # ========================================================

    prompt = f"""

You are a document-based AI assistant.

Answer the user's question using ONLY the
provided document context.

IMPORTANT RULES:

1. Do not invent information.
2. Do not use outside knowledge.
3. Base the answer only on the retrieved documents.
4. If the answer is not present in the documents,
   clearly say that it was not found.
5. Keep the answer easy to understand.
6. Identify the source documents used.
7. After the answer, provide a short summary.
8. Do not claim that a paragraph was used unless
   it is present in the provided context.


DOCUMENT CONTEXT:

{context}


USER QUESTION:

{question}


Provide the response in this structure:

ANSWER:
[Detailed answer]

SUMMARY:
[2-3 sentence summary]

"""


    # ========================================================
    # GROQ
    # ========================================================

    client = Groq(
        api_key=groq_api_key
    )


    response = client.chat.completions.create(

        model=MODEL,

        messages=[

            {
                "role": "system",

                "content":
                "You are a grounded document "
                "question-answering assistant."
            },

            {
                "role": "user",

                "content": prompt
            }

        ],

        temperature=0.2
    )


    return response.choices[
        0
    ].message.content


# ============================================================
# 19. DISPLAY PREVIOUS CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# 20. CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask something about your documents..."
)


# ============================================================
# 21. PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # Check documents
    # --------------------------------------------------------

    if st.session_state.index is None:

        st.warning(
            "Please upload at least one document first."
        )

        st.stop()


    # --------------------------------------------------------
    # Check API key
    # --------------------------------------------------------

    if not groq_api_key:

        st.warning(
            "Please enter your Groq API key in the sidebar."
        )

        st.stop()


    # --------------------------------------------------------
    # Save user message
    # --------------------------------------------------------

    st.session_state.messages.append({

        "role": "user",

        "content": question

    })


    # --------------------------------------------------------
    # Display user question
    # --------------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # ========================================================
    # RETRIEVAL
    # ========================================================

    with st.spinner(
        "🔎 Searching documents..."
    ):

        retrieved_results = retrieve_documents(
            question,
            top_k=5
        )


    # ========================================================
    # GENERATION
    # ========================================================

    with st.spinner(
        "🤖 Generating answer..."
    ):

        answer = generate_rag_answer(
            question,
            retrieved_results
        )


    # --------------------------------------------------------
    # Save assistant response
    # --------------------------------------------------------

    st.session_state.messages.append({

        "role": "assistant",

        "content": answer

    })


    # ========================================================
    # DISPLAY ANSWER
    # ========================================================

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            answer
        )


    # ========================================================
    # KPI SECTION
    # ========================================================

    st.subheader(
        "📊 Retrieval Analytics"
    )


    # --------------------------------------------------------
    # Calculate KPI values
    # --------------------------------------------------------

    retrieved_count = len(
        retrieved_results
    )


    source_files = len(
        set(
            result["filename"]
            for result in retrieved_results
        )
    )


    if retrieved_results:

        best_distance = min(
            result["distance"]
            for result in retrieved_results
        )

    else:

        best_distance = 0


    # --------------------------------------------------------
    # KPI boxes
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "🔎 Retrieved Chunks",
            retrieved_count
        )


    with col2:

        st.metric(
            "📄 Source Documents",
            source_files
        )


    with col3:

        st.metric(
            "🎯 Best FAISS Distance",
            f"{best_distance:.4f}"
        )


    with col4:

        st.metric(
            "💬 Chat Messages",
            len(st.session_state.messages)
        )


    # ========================================================
    # RETRIEVED PARAGRAPHS
    # ========================================================

    st.subheader(
        "📚 Retrieved Paragraphs / Chunks"
    )


    st.caption(
        "These are the document sections retrieved "
        "by FAISS and provided to the LLM as context."
    )


    for rank, result in enumerate(
        retrieved_results,
        1
    ):

        with st.expander(
            f"Source {rank} — {result['filename']}"
        ):

            st.write(
                f"**Chunk ID:** "
                f"{result['chunk_id']}"
            )

            st.write(
                f"**FAISS Distance:** "
                f"{result['distance']:.4f}"
            )


            if result["page"]:

                st.write(
                    f"**Page:** "
                    f"{result['page']}"
                )


            if result["paragraph"]:

                st.write(
                    f"**Paragraph:** "
                    f"{result['paragraph']}"
                )


            st.markdown(
                "**Retrieved Text:**"
            )

            st.write(
                result["text"]
            )


# ============================================================
# 22. CHAT HISTORY SIDEBAR
# ============================================================

st.sidebar.divider()

st.sidebar.header(
    "💬 Chat History"
)


if st.session_state.messages:

    for i, message in enumerate(
        st.session_state.messages,
        1
    ):

        if message["role"] == "user":

            st.sidebar.write(
                f"{i}. 👤 "
                f"{message['content'][:70]}"
            )

else:

    st.sidebar.info(
        "No questions asked yet."
    )


# ============================================================
# 23. CLEAR CHAT
# ============================================================

if st.sidebar.button(
    "🗑️ Clear Chat History"
):

    st.session_state.messages = []

    st.rerun()