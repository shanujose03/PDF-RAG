import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
import cohere
import os
from PyPDF2 import PdfReader

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

# ---------------- STYLES ---------------- #
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: linear-gradient(to bottom right, #f5f7fa, #c3cfe2);
    z-index: -1;
}

.chat-bubble {
    background-color: rgba(255, 255, 255, 0.95);
    border-radius: 12px;
    padding: 15px;
    margin-top: 10px;
    font-size: 16px;
    color: #000000;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.15);
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ---------------- COHERE CLIENT ---------------- #
# SET YOUR KEY IN ENV VARIABLE: COHERE_API_KEY
import os
co = cohere.Client(os.getenv("CO_API_KEY"))

from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

# Streamlit Cloud friendly: always download to /tmp (ephemeral storage)
model_path = snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    cache_dir="/tmp/model_cache"  # Streamlit Cloud temp folder
)

embedder = SentenceTransformer(model_path)

# ---------------- PDF UTILITIES ---------------- #
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def create_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def search_document(query, index, documents, top_k=3):
    query_vector = embedder.encode([query])
    distances, indices = index.search(query_vector, top_k)
    return [documents[i] for i in indices[0]]


def generate_answer(query, index, documents):
    context = "\n\n".join(search_document(query, index, documents))

    prompt = f"""
You are a helpful AI assistant.
Answer the user's question using ONLY the information provided in the context below.
Do NOT use outside knowledge.
If the answer cannot be found in the context, respond with:
"Answer not found in the document."

Context:
{context}

Question:
{query}

Answer:
"""

    response = co.chat(
        model="command-xlarge-nightly",
        message=prompt,
        max_tokens=300,
        temperature=0.3
    )

    return response.text.strip()

# ---------------- STREAMLIT UI ---------------- #
st.title("📄 PDF Question Answering (RAG System)")

uploaded_pdf = st.file_uploader("Upload any PDF document", type="pdf")

if uploaded_pdf:
    if "index" not in st.session_state:
        with st.spinner("Reading PDF and creating vector embeddings..."):
            text = extract_text_from_pdf(uploaded_pdf)
            chunks = chunk_text(text)
            index = create_faiss_index(chunks)

            st.session_state.index = index
            st.session_state.documents = chunks

        st.success("PDF processed successfully. You can now ask questions.")

    query = st.text_input("Ask a question from the document")

    if query:
        with st.spinner("Generating answer..."):
            answer = generate_answer(
                query,
                st.session_state.index,
                st.session_state.documents
            )

        st.subheader("🤖 Answer")
        st.markdown(f"<div class='chat-bubble'>{answer}</div>", unsafe_allow_html=True)



