import numpy as np
import os
import re
import uuid
import time
import threading
from pathlib import Path
from typing import List

import requests
from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from gtts import gTTS


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

UPLOAD_DIR = BASE_DIR / "uploads"
AUDIO_DIR = BASE_DIR / "audio"

UPLOAD_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

load_dotenv(ROOT_DIR / ".env")

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    ""
).strip()

GEMINI_MODEL = "gemini-3.6-flash"

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="StudyMate AI",
    description="RAG Based Document Assistant",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
        "http://localhost:5177",

        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        "http://127.0.0.1:5176",
        "http://127.0.0.1:5177",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# GLOBAL RAG VARIABLES
# =========================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

vector_store = None
document_chunks: List[str] = []


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_pdf_text(text: str) -> str:

    if not text:
        return ""

    # Remove null characters
    text = text.replace("\x00", " ")

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Replace common PDF bullet characters
    text = re.sub(
        r"[▪●•]",
        "\n",
        text
    )

    # Remove page-number-only lines
    text = re.sub(
        r"(?m)^\s*\d{1,3}\s*$",
        "",
        text
    )

    # Fix words broken across lines
    # Example:
    # computational-
    # procedure
    #
    # becomes:
    # computational procedure

    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text
    )

    # Replace multiple spaces/tabs
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Normalize excessive newlines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    # Clean spaces around newlines
    text = re.sub(
        r"[ \t]*\n[ \t]*",
        "\n",
        text
    )

    return text.strip()


# =========================================================
# SENTENCE SPLITTER
# =========================================================

def split_into_sentences(
    text: str
) -> List[str]:

    text = clean_pdf_text(text)

    if not text:
        return []

    # Convert single line breaks into spaces
    # when they occur inside normal sentences.

    text = re.sub(
        r"(?<![.!?:])\n(?=[a-z])",
        " ",
        text
    )

    text = re.sub(
        r"([,:;])\n",
        r"\1 ",
        text
    )

    # Separate sentences
    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z])",
        text
    )

    result = []

    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence) >= 20:
            result.append(sentence)

    return result


# =========================================================
# DOCUMENT CHUNKING
# =========================================================

def create_chunks(
    text: str
) -> List[str]:

    text = clean_pdf_text(text)

    if not text:
        return []

    sentences = split_into_sentences(text)

    if not sentences:
        return []

    chunks = []

    current_chunk = []
    current_length = 0

    max_chars = 900

    for sentence in sentences:

        sentence_length = len(sentence)

        if (
            current_length + sentence_length + 1
            > max_chars
            and current_chunk
        ):

            chunks.append(
                " ".join(current_chunk).strip()
            )

            current_chunk = []
            current_length = 0

        current_chunk.append(sentence)

        current_length += sentence_length + 1

    if current_chunk:

        chunks.append(
            " ".join(current_chunk).strip()
        )

    return chunks


# =========================================================
# DUPLICATE REMOVAL
# =========================================================

def remove_duplicate_points(
    points: List[str]
) -> List[str]:

    result = []
    seen = set()

    for point in points:

        cleaned = re.sub(
            r"\s+",
            " ",
            point
        ).strip()

        key = cleaned.lower()

        if (
            cleaned
            and key not in seen
        ):

            seen.add(key)
            result.append(cleaned)

    return result


# =========================================================
# ALGORITHM-SPECIFIC EXTRACTION
# =========================================================

def extract_algorithm_points(
    text: str
) -> List[str]:

    points = []

    text = clean_pdf_text(text)

    # -----------------------------------------------------
    # Algorithm definitions
    # -----------------------------------------------------

    definition_patterns = [

        r"An algorithm is any well-defined computational procedure.*?\.",

        r"An algorithm is thus a sequence of computational steps.*?\.",

        r"An algorithm is a set of steps.*?\."
    ]

    for pattern in definition_patterns:

        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        for match in matches:

            match = re.sub(
                r"\s+",
                " ",
                match
            ).strip()

            if len(match) > 40:
                points.append(match)

    # -----------------------------------------------------
    # Characteristics of Algorithm
    # -----------------------------------------------------

    characteristics_match = re.search(
        r"Characteristics of an algorithm"
        r"(.*?)"
        r"(?=Analysis of algorithms|"
        r"Running time|"
        r"Pseudo Code|"
        r"$)",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if characteristics_match:

        section = characteristics_match.group(1)

        characteristic_patterns = [

            r"Input\s*:\s*.*?"
            r"(?=\s+(?:Output|Finiteness|"
            r"Definiteness|Effectiveness)\s*:|$)",

            r"Output\s*:\s*.*?"
            r"(?=\s+(?:Input|Finiteness|"
            r"Definiteness|Effectiveness)\s*:|$)",

            r"Finiteness\s*:\s*.*?"
            r"(?=\s+(?:Input|Output|"
            r"Definiteness|Effectiveness)\s*:|$)",

            r"Definiteness\s*:\s*.*?"
            r"(?=\s+(?:Input|Output|"
            r"Finiteness|Effectiveness)\s*:|$)",

            r"Effectiveness\s*:\s*.*?"
            r"(?=\s+(?:Input|Output|"
            r"Finiteness|Definiteness)\s*:|$)"
        ]

        for pattern in characteristic_patterns:

            matches = re.findall(
                pattern,
                section,
                flags=re.IGNORECASE | re.DOTALL
            )

            for match in matches:

                cleaned = re.sub(
                    r"\s+",
                    " ",
                    match
                ).strip()

                if len(cleaned) > 10:
                    points.append(cleaned)

    return remove_duplicate_points(points)


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(
    text: str
) -> str:

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =========================================================
# LOCAL RAG SEARCH
# =========================================================

def search_relevant_chunks(
    query: str,
    top_k: int = 3
):

    global vector_store
    global document_chunks

    if (
        vector_store is None
        or not document_chunks
    ):
        return []

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding
    ).astype("float32")

    distances, indices = vector_store.search(
        query_embedding,
        min(
            top_k,
            len(document_chunks)
        )
    )

    results = []

    for distance, index in zip(
        distances[0],
        indices[0]
    ):

        if index < 0:
            continue

        results.append({
            "text": document_chunks[index],
            "score": float(distance)
        })

    return results


# =========================================================
# CLEAN ANSWER FORMAT
# =========================================================

def clean_answer_format(
    answer: str
) -> str:

    if not answer:
        return ""

    # Normalize line endings
    answer = answer.replace("\r\n", "\n")
    answer = answer.replace("\r", "\n")

    # Remove Markdown code fences
    answer = re.sub(
        r"```(?:markdown|md|text)?",
        "",
        answer,
        flags=re.IGNORECASE
    )

    answer = answer.replace(
        "```",
        ""
    )

    # Remove source/mode labels if Gemini accidentally adds them
    # These will remain separate in the API response.

    answer = re.sub(
        r"(?im)^\s*(?:📚\s*)?sources?\s*:\s*.*$",
        "",
        answer
    )

    answer = re.sub(
        r"(?im)^\s*(?:🔎\s*)?mode\s*:\s*.*$",
        "",
        answer
    )

    # -----------------------------------------------------
    # Normalize escaped Markdown
    # -----------------------------------------------------

    answer = answer.replace(
        r"\*",
        "*"
    )

    answer = answer.replace(
        r"\_",
        "_"
    )

    answer = answer.replace(
        r"\#",
        "#"
    )

    answer = answer.replace(
        r"\`",
        "`"
    )

    # -----------------------------------------------------
    # Convert PDF bullet symbols into proper Markdown
    #
    # IMPORTANT:
    # Put bullet on a NEW LINE.
    # This prevents examples and sentences from getting
    # mixed together.
    # -----------------------------------------------------

    answer = re.sub(
        r"[▪●•]\s*",
        "\n- ",
        answer
    )

    # Normalize existing bullet lines
    answer = re.sub(
        r"(?m)^[ \t]*[-]\s*",
        "- ",
        answer
    )

    # Normalize numbered lists
    answer = re.sub(
        r"(?m)^[ \t]*(\d+)[.)]\s+",
        r"\1. ",
        answer
    )

    # -----------------------------------------------------
    # Keep "For example:" connected to its explanation.
    # -----------------------------------------------------

    answer = re.sub(
        r"\n\s*(For example\s*:?)\s*\n+",
        r"\n\n\1 ",
        answer,
        flags=re.IGNORECASE
    )

    # If "For example:" occurs after a bullet,
    # keep it as part of the same bullet.
    answer = re.sub(
        r"(?m)^(- .+?)\n+(For example\s*:?)\s*",
        r"\1 \2 ",
        answer,
        flags=re.IGNORECASE
    )

    # -----------------------------------------------------
    # Normalize spaces while preserving newlines
    # -----------------------------------------------------

    answer = re.sub(
        r"[ \t]+",
        " ",
        answer
    )

    # Remove spaces before punctuation
    answer = re.sub(
        r"\s+([,.;:!?])",
        r"\1",
        answer
    )

    # Clean spaces after newlines
    answer = re.sub(
        r"[ \t]*\n[ \t]*",
        "\n",
        answer
    )

    # Keep Markdown headings clean
    answer = re.sub(
        r"(?m)^#{1,6}\s*",
        lambda m: m.group(0).strip() + " ",
        answer
    )

    # Remove excessive blank lines
    answer = re.sub(
        r"\n{3,}",
        "\n\n",
        answer
    )

    return answer.strip()


# =========================================================
# LOCAL RAG FALLBACK
# =========================================================

def local_rag_answer(
    query: str,
    chunks
) -> str:

    if not chunks:

        return (
            "## Answer\n\n"
            "Uploaded document mein "
            "relevant information nahi mili."
        )

    query_words = set(
        re.findall(
            r"\b[a-zA-Z]{3,}\b",
            query.lower()
        )
    )

    scored_sentences = []

    for item in chunks:

        text = normalize_text(
            item["text"]
        )

        # Remove PDF bullet symbols
        text = re.sub(
            r"[▪●•]",
            " ",
            text
        )

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if len(sentence) < 30:
                continue

            sentence_words = set(
                re.findall(
                    r"\b[a-zA-Z]{3,}\b",
                    sentence.lower()
                )
            )

            score = len(
                query_words.intersection(
                    sentence_words
                )
            )

            if score > 0:

                scored_sentences.append(
                    (score, sentence)
                )

    # Highest matching sentences first
    scored_sentences.sort(
        key=lambda x: x[0],
        reverse=True
    )

    selected = []
    seen = set()

    for _, sentence in scored_sentences:

        key = sentence.lower()

        if key in seen:
            continue

        seen.add(key)

        selected.append(sentence)

        if len(selected) >= 6:
            break

    if not selected:

        return (
            "## Answer\n\n"
            "Uploaded document mein "
            "is question ka clear answer nahi mila."
        )

    # -----------------------------------------------------
    # Try to include a relevant example from the document
    # -----------------------------------------------------

    example_sentences = []

    for item in chunks:

        text = normalize_text(
            item["text"]
        )

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if (
                len(sentence) >= 30
                and (
                    "for example" in sentence.lower()
                    or "example:" in sentence.lower()
                    or "e.g." in sentence.lower()
                )
            ):
                example_sentences.append(sentence)

    # Add relevant example if it is not already selected
    if example_sentences:

        for example in example_sentences:

            if example.lower() not in {
                x.lower() for x in selected
            }:

                selected.append(example)
                break

    # -----------------------------------------------------
    # Clean selected points
    # -----------------------------------------------------

    cleaned_points = []

    for sentence in selected:

        sentence = re.sub(
            r"^[\s•▪●\-]+",
            "",
            sentence
        ).strip()

        if sentence:
            cleaned_points.append(sentence)

    # -----------------------------------------------------
    # Markdown answer
    # -----------------------------------------------------

    answer_lines = [
        "## Answer from your uploaded document",
        ""
    ]

    for point in cleaned_points:

        # Keep example inside answer
        if (
            "for example" in point.lower()
            or "example:" in point.lower()
        ):

            answer_lines.append(
                f"**For example:** {point}"
            )

        else:

            answer_lines.append(
                f"- {point}"
            )

    return "\n".join(answer_lines)


# =========================================================
# GEMINI AI
# =========================================================

def ask_gemini(
    question: str,
    context: str
):

    if not GEMINI_API_KEY:

        print(
            "⚠ GEMINI_API_KEY not found"
        )

        return None

    prompt = f"""
You are StudyMate AI, a document-based study assistant.

Answer the student's question ONLY using the provided document context.

IMPORTANT RULES:

1. Do not use information that is not present in the document.

2. Give a clear, simple and student-friendly explanation.

3. Organize the answer using clean Markdown.

4. Use headings with ## or ### when useful.

5. Use bullet points with "-" only.

6. Use numbered lists with "1.", "2.", "3." when steps or lists are needed.

7. Use **bold** for important terms.

8. Do not use symbols such as •, , ,  or ▪.

9. If the document contains an example relevant to the question,
include that example INSIDE THE ANSWER.

10. When an example is available, write it naturally using
"For example:" and keep the example directly connected to
the statement it explains.

11. Do NOT move "For example:" into a separate source section.

12. Do NOT put examples only in the sources.

13. Keep "For example:" and its explanation together.

14. Each bullet point should contain a complete thought.

15. Keep numbered-list content on the same line as its number.

16. Do not include phrases such as:
"Source:"
"Sources:"
"Mode:"
"Answer from your uploaded document"

17. Do not include source information or mode information inside
the answer.

18. Keep the answer focused on the student's question.

19. Do not add unrelated information.

20. Do not repeat the same point unnecessarily.

21. If the document does not contain enough information, clearly say:

"The uploaded document does not provide enough information to answer this question."

Question:

{question}

Document Context:

{context}

Now provide the final answer in clean Markdown.

Make sure the answer reads naturally, like a good teacher
explaining the topic to a student.
"""

    payload = {

        "contents": [

            {

                "parts": [

                    {

                        "text": prompt

                    }

                ]

            }

        ],

        "generationConfig": {

            "temperature": 0.2,

            "maxOutputTokens": 1200

        }

    }

    headers = {

        "Content-Type":
            "application/json",

        "x-goog-api-key":
            GEMINI_API_KEY
    }

    try:

        response = requests.post(
            GEMINI_URL,
            headers=headers,
            json=payload,
            timeout=15
        )

        if response.status_code != 200:

            print(
                "Gemini error:",
                response.status_code,
                response.text[:500]
            )

            return None

        data = response.json()

        candidates = data.get(
            "candidates",
            []
        )

        if not candidates:
            return None

        parts = candidates[0].get(
            "content",
            {}
        ).get(
            "parts",
            []
        )

        if not parts:
            return None

        answer = parts[0].get(
            "text",
            ""
        ).strip()

        if not answer:
            return None

        answer = clean_answer_format(
            answer
        )

        print(
            "✓ Gemini AI response generated"
        )

        return answer

    except requests.exceptions.Timeout:

        print(
            "⚠ Gemini request timed out"
        )

        return None

    except Exception as e:

        print(
            "⚠ Gemini request failed:",
            str(e)
        )

        return None


# =========================================================
# BACKGROUND AUDIO GENERATION
# =========================================================

def generate_audio_file(
    text: str,
    filename: str
):

    try:

        # Remove Markdown
        speech_text = re.sub(
            r"[\*_#`]",
            "",
            text
        )

        # Remove bullet symbols
        speech_text = re.sub(
            r"[•▪●]",
            "",
            speech_text
        )

        # Remove Markdown list markers
        speech_text = re.sub(
            r"(?m)^\s*[-*]\s*",
            "",
            speech_text
        )

        speech_text = re.sub(
            r"(?m)^\s*\d+\.\s*",
            "",
            speech_text
        )

        # Remove excessive whitespace
        speech_text = re.sub(
            r"\s+",
            " ",
            speech_text
        ).strip()

        filepath = AUDIO_DIR / filename

        try:

            tts = gTTS(
                text=speech_text,
                lang="en",
                slow=False,
                timeout=10
            )

        except TypeError:

            tts = gTTS(
                text=speech_text,
                lang="en",
                slow=False
            )

        tts.save(
            str(filepath)
        )

        print(
            f"✓ Audio generated: {filename}"
        )

    except Exception as e:

        print(
            "⚠ TTS failed:",
            str(e)
        )


# =========================================================
# AUDIO ENDPOINT
# =========================================================

@app.get("/audio/{filename}")
def get_audio(
    filename: str
):

    filepath = AUDIO_DIR / filename

    # Wait maximum 15 seconds
    for _ in range(30):

        if filepath.exists():

            return FileResponse(
                path=str(filepath),
                media_type="audio/mpeg"
            )

        time.sleep(0.5)

    raise HTTPException(
        status_code=404,
        detail="Audio is not available."
    )


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {

        "message":
            "StudyMate AI Backend is Running!",

        "status":
            "online",

        "model":
            GEMINI_MODEL
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "gemini_configured":
            bool(GEMINI_API_KEY),

        "document_loaded":
            bool(document_chunks),

        "chunks":
            len(document_chunks)
    }


# =========================================================
# PDF UPLOAD
# =========================================================

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):

    global vector_store
    global document_chunks

    if not file.filename.lower().endswith(
        ".pdf"
    ):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    try:

        file_path = (
            UPLOAD_DIR /
            file.filename
        )

        file_content = await file.read()

        with open(
            file_path,
            "wb"
        ) as f:

            f.write(
                file_content
            )

        # -------------------------------------------------
        # Extract PDF text
        # -------------------------------------------------

        reader = PdfReader(
            str(file_path)
        )

        full_text = ""

        for page in reader.pages:

            page_text = (
                page.extract_text()
                or ""
            )

            full_text += (
                page_text + "\n"
            )

        if not full_text.strip():

            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF."
            )

        # -------------------------------------------------
        # Clean text
        # -------------------------------------------------

        cleaned_text = clean_pdf_text(
            full_text
        )

        # -------------------------------------------------
        # Create chunks
        # -------------------------------------------------

        chunks = create_chunks(
            cleaned_text
        )

        if not chunks:

            raise HTTPException(
                status_code=400,
                detail="No usable text found in PDF."
            )

        document_chunks = chunks

        # -------------------------------------------------
        # Create embeddings
        # -------------------------------------------------

        embeddings = embedding_model.encode(
            document_chunks,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        embeddings = np.asarray(
            embeddings
        ).astype("float32")

        dimension = embeddings.shape[1]

        vector_store = faiss.IndexFlatIP(
            dimension
        )

        vector_store.add(
            embeddings
        )

        return {

            "message":
                "PDF uploaded successfully",

            "filename":
                file.filename,

            "pages":
                len(reader.pages),

            "chunks":
                len(document_chunks)
        }

    except HTTPException:

        raise

    except Exception as e:

        print(
            "Upload error:",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )


# =========================================================
# ASK QUESTION
# =========================================================

@app.post("/ask")
def ask_question(
    query: str
):

    if not query.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    if not document_chunks:

        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF first."
        )

    # -----------------------------------------------------
    # Retrieve relevant chunks
    # -----------------------------------------------------

    relevant_chunks = search_relevant_chunks(
        query,
        top_k=3
    )

    if not relevant_chunks:

        raise HTTPException(
            status_code=404,
            detail="No relevant information found."
        )

    # -----------------------------------------------------
    # Algorithm query detection
    # -----------------------------------------------------

    algorithm_keywords = [
        "algorithm",
        "characteristics of algorithm",
        "definition of algorithm"
    ]

    is_algorithm_query = any(
        keyword in query.lower()
        for keyword in algorithm_keywords
    )

    # -----------------------------------------------------
    # Build context
    # -----------------------------------------------------

    context = "\n\n".join(
        item["text"]
        for item in relevant_chunks
    )

    # -----------------------------------------------------
    # Try Gemini first
    # -----------------------------------------------------

    gemini_answer = ask_gemini(
        query,
        context
    )

    mode = "Gemini AI"

    if gemini_answer:

        answer = gemini_answer

    else:

        # -------------------------------------------------
        # Local RAG fallback
        # -------------------------------------------------

        if is_algorithm_query:

            points = extract_algorithm_points(
                "\n".join(
                    item["text"]
                    for item in relevant_chunks
                )
            )

            if points:

                answer = (
                    "## Answer from your uploaded document\n\n"
                    "### Key Points\n\n"
                    + "\n".join(
                        f"- {point}"
                        for point in points[:6]
                    )
                )

            else:

                answer = local_rag_answer(
                    query,
                    relevant_chunks
                )

        else:

            answer = local_rag_answer(
                query,
                relevant_chunks
            )

        mode = "Local RAG fallback"

    # -----------------------------------------------------
    # Final answer cleanup
    # -----------------------------------------------------

    answer = clean_answer_format(
        answer
    )

    # -----------------------------------------------------
    # Generate audio in background
    # -----------------------------------------------------

    audio_filename = (
        f"{uuid.uuid4().hex}.mp3"
    )

    audio_thread = threading.Thread(
        target=generate_audio_file,
        args=(
            answer,
            audio_filename
        ),
        daemon=True
    )

    audio_thread.start()

    audio_url = (
        f"/audio/{audio_filename}"
    )

    # -----------------------------------------------------
    # Sources
    # -----------------------------------------------------

    sources = [
        normalize_text(item["text"])
        for item in relevant_chunks
    ]

    # Remove duplicate sources
    sources = remove_duplicate_points(
        sources
    )

    # -----------------------------------------------------
    # FINAL API RESPONSE
    #
    # Exact order:
    #
    # question
    # answer
    # audio_url
    # sources
    # mode
    # -----------------------------------------------------

    return {

        "question":
            query,

        "answer":
            answer,

        "audio_url":
            audio_url,

        "sources":
            sources,

        "mode":
            mode
    }