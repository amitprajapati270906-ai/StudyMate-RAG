import re
import faiss
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# =========================================================
# EMBEDDING MODEL
# =========================================================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

document_chunks = []


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_text_from_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    pages = []

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            pages.append(page_text)

    return "\n\n".join(pages)


# =========================================================
# CLEAN PDF TEXT
# =========================================================

def clean_pdf_text(text):

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # PDF bullet characters
    text = text.replace("", "\n")
    text = text.replace("", "\n")
    text = text.replace("", "\n")
    text = text.replace("▪", "\n")
    text = text.replace("●", "\n")
    text = text.replace("•", "\n")

    # Remove page-number-only lines
    text = re.sub(
        r"(?m)^\s*\d{1,3}\s*$",
        "",
        text
    )

    # Remove page number before headings
    text = re.sub(
        r"(?m)^\s*\d{1,3}\s+(?=[A-Z][A-Za-z ]{2,80}$)",
        "",
        text
    )

    # Fix words broken across lines
    # Example:
    # computa-
    # tional
    #
    # becomes:
    # computational

    text = re.sub(
        r"(\w)-\n(\w)",
        r"\1\2",
        text
    )

    # Convert normal line breaks inside sentences into spaces.
    #
    # Example:
    # computational
    # procedure that takes
    #
    # becomes:
    # computational procedure that takes

    text = re.sub(
        r"(?<![.!?:])\n(?=[a-z])",
        " ",
        text
    )

    # If previous line ends with comma/closing bracket,
    # join with next line.
    text = re.sub(
        r"([,:;])\n",
        r"\1 ",
        text
    )

    # Multiple spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# =========================================================
# SENTENCE SPLITTING
# =========================================================

def split_into_sentences(text):

    text = clean_pdf_text(text)

    if not text:
        return []

    # First separate paragraphs
    paragraphs = re.split(
        r"\n{2,}",
        text
    )

    sentences = []

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # Split after ., !, ?
        parts = re.split(
            r"(?<=[.!?])\s+(?=[A-Z])",
            paragraph
        )

        for part in parts:

            part = part.strip()

            if len(part) >= 20:
                sentences.append(part)

    return sentences


# =========================================================
# CREATE SEMANTIC CHUNKS
# =========================================================

def create_chunks(
    text,
    chunk_size=700,
    overlap=120
):
    """
    Creates chunks without cutting sentences unnecessarily.
    """

    text = clean_pdf_text(text)

    if not text:
        return []

    sentences = split_into_sentences(text)

    if not sentences:
        return []

    chunks = []

    current_chunk = ""
    current_sentences = []

    for sentence in sentences:

        # If adding the sentence stays within chunk size
        if (
            len(current_chunk) + len(sentence) + 1
            <= chunk_size
        ):

            current_sentences.append(sentence)

            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

        else:

            if current_chunk.strip():
                chunks.append(
                    current_chunk.strip()
                )

            # Keep a small overlap using last sentence
            overlap_sentences = []

            if current_sentences:
                overlap_text = current_sentences[-1]

                if len(overlap_text) <= overlap:
                    overlap_sentences.append(
                        overlap_text
                    )

            current_sentences = overlap_sentences

            if current_sentences:
                current_chunk = current_sentences[0]

                # If current sentence itself is too long
                if (
                    len(current_chunk)
                    + len(sentence)
                    + 1
                    <= chunk_size
                ):
                    current_chunk += " " + sentence
                    current_sentences.append(sentence)
                else:
                    current_chunk = sentence
                    current_sentences = [sentence]

            else:
                current_chunk = sentence
                current_sentences = [sentence]

    # Add final chunk
    if current_chunk.strip():
        chunks.append(
            current_chunk.strip()
        )

    return chunks


# =========================================================
# CREATE FAISS VECTOR DATABASE
# =========================================================

def create_vector_database(chunks):

    global document_chunks

    document_chunks = chunks

    if not chunks:
        raise ValueError(
            "No document chunks available."
        )

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return index


# =========================================================
# SEARCH SIMILAR CHUNKS
# =========================================================

def search_similar_chunks(
    index,
    query,
    top_k=3
):

    if index is None:
        return []

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False
    )

    query_embedding = np.asarray(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for i in indices[0]:

        if (
            i != -1
            and i < len(document_chunks)
        ):
            results.append(
                document_chunks[i]
            )

    return results