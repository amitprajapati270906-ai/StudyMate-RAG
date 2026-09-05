# StudyMate AI – RAG Based Document Assistant

StudyMate AI is an AI-powered document assistant that allows users to upload PDF study material and ask questions based on the uploaded document.

The system uses **Retrieval-Augmented Generation (RAG)** to find relevant information from the document and generate a clear, student-friendly answer using Gemini AI.

---

## 🚀 Features

- 📄 Upload PDF study material
- 🔍 Extract and process document text
- ✂️ Split documents into smaller chunks
- 🧠 Generate semantic embeddings
- ⚡ Fast similarity search using FAISS
- 🤖 Generate answers using Gemini AI
- 📚 Display relevant source information
- 🔊 Listen to generated answers using Text-to-Speech
- 📋 Copy answers easily
- 📝 Markdown formatted answers with headings, bullets and examples
- 🔐 API key stored securely using environment variables

---

## 🧠 How It Works

The application follows a Retrieval-Augmented Generation (RAG) pipeline:

```text
PDF / Notes
     ↓
Text Extraction
     ↓
Text Chunking
     ↓
Embeddings
     ↓
FAISS Vector Database
     ↓
Relevant Chunks Retrieval
     ↓
Gemini AI
     ↓
Student-Friendly Answer