import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

const API_URL = "http://127.0.0.1:8010";

function App() {
  const [file, setFile] = useState(null);
  const [uploaded, setUploaded] = useState(false);
  const [uploading, setUploading] = useState(false);

  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);

  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);

  const [audioUrl, setAudioUrl] = useState("");
  const [playing, setPlaying] = useState(false);
  const [copied, setCopied] = useState(false);

  const [error, setError] = useState("");

  const [voices, setVoices] = useState([]);
  const speechRef = useRef(null);

  // ========================================
  // LOAD AVAILABLE VOICES
  // ========================================

  useEffect(() => {
    const loadVoices = () => {
      const availableVoices =
        window.speechSynthesis.getVoices();

      setVoices(availableVoices);
    };

    loadVoices();

    window.speechSynthesis.onvoiceschanged =
      loadVoices;

    return () => {
      window.speechSynthesis.onvoiceschanged = null;
    };
  }, []);

  // ========================================
  // CLEAN AI ANSWER
  // ========================================

  const cleanAnswer = (text) => {
    if (!text) return "";

    return text
      .replace(/\\\*/g, "*")
      .replace(/\\_/g, "_")
      .replace(/\\#/g, "#")
      .replace(/\\`/g, "`")
      .replace(/\\"/g, '"')
      .trim();
  };

  // ========================================
  // STOP VOICE
  // ========================================

  const stopVoice = () => {
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }

    speechRef.current = null;
    setPlaying(false);
  };

  // ========================================
  // SELECT PDF
  // ========================================

  const handleFile = (e) => {
    const selected = e.target.files[0];

    if (!selected) return;

    if (
      !selected.name
        .toLowerCase()
        .endsWith(".pdf")
    ) {
      setError("Please select a PDF file.");
      return;
    }

    stopVoice();

    setFile(selected);
    setUploaded(false);
    setAnswer("");
    setSources([]);
    setAudioUrl("");
    setCopied(false);
    setError("");
  };

  // ========================================
  // UPLOAD PDF
  // ========================================

  const uploadPDF = async () => {
    if (!file) {
      setError("Please select a PDF first.");
      return;
    }

    setUploading(true);
    setError("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(
        `${API_URL}/upload`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "PDF upload failed."
        );
      }

      setUploaded(true);
      setError("");
    } catch (err) {
      console.error("Upload error:", err);

      setUploaded(false);

      setError(
        err.message || "PDF upload failed."
      );
    } finally {
      setUploading(false);
    }
  };

  // ========================================
  // ASK AI
  // ========================================

  const askAI = async () => {
    if (!uploaded) {
      setError(
        "Please upload and process your PDF first."
      );
      return;
    }

    if (!question.trim()) {
      setError("Please enter your question.");
      return;
    }

    stopVoice();

    setAsking(true);
    setError("");
    setAnswer("");
    setSources([]);
    setAudioUrl("");
    setCopied(false);

    try {
      const response = await fetch(
        `${API_URL}/ask?query=${encodeURIComponent(
          question.trim()
        )}`,
        {
          method: "POST",
        }
      );

      const data = await response.json();

      if (response.status === 429) {
        setError(
          "Gemini AI quota exceeded. Please try again later."
        );
        return;
      }

      if (!response.ok) {
        throw new Error(
          data.detail || "Failed to get answer."
        );
      }

      const cleanedAnswer = cleanAnswer(
        data.answer || ""
      );

      setAnswer(cleanedAnswer);

      setSources(data.sources || []);

      setAudioUrl("voice-ready");
    } catch (err) {
      console.error("Ask AI error:", err);

      setError(
        err.message ||
          "Something went wrong. Please try again."
      );
    } finally {
      setAsking(false);
    }
  };

  // ========================================
  // COPY ANSWER
  // ========================================

  const copyAnswer = async () => {
    if (!answer) {
      setError("Answer is not available.");
      return;
    }

    try {
      await navigator.clipboard.writeText(answer);

      setCopied(true);
      setError("");

      setTimeout(() => {
        setCopied(false);
      }, 2000);
    } catch (err) {
      console.error("Copy error:", err);

      setError("Unable to copy answer.");
    }
  };

  // ========================================
  // PLAY / STOP VOICE
  // ========================================

  const playAudio = () => {
    if (!answer) {
      setError("Answer is not available.");
      return;
    }

    if (!window.speechSynthesis) {
      setError(
        "Text-to-speech is not supported in this browser."
      );
      return;
    }

    try {
      // Stop current voice
      if (playing) {
        window.speechSynthesis.cancel();

        setPlaying(false);
        speechRef.current = null;

        return;
      }

      window.speechSynthesis.cancel();
      setError("");

      const speech =
        new SpeechSynthesisUtterance(answer);

      // ========================================
      // PREFER MICROSOFT DAVID
      // ========================================

      const davidVoice = voices.find((voice) =>
        voice.name
          .toLowerCase()
          .includes("microsoft david")
      );

      if (davidVoice) {
        speech.voice = davidVoice;
        speech.lang = davidVoice.lang;
      } else {
        // Fallback English voice
        const englishVoice = voices.find(
          (voice) =>
            voice.lang
              .toLowerCase()
              .startsWith("en")
        );

        if (englishVoice) {
          speech.voice = englishVoice;
          speech.lang = englishVoice.lang;
        } else {
          speech.lang = "en-US";
        }
      }

      speech.rate = 0.95;
      speech.pitch = 1;
      speech.volume = 1;

      speech.onstart = () => {
        setPlaying(true);
        setError("");
      };

      speech.onend = () => {
        setPlaying(false);
        speechRef.current = null;
      };

      speech.onerror = (event) => {
        console.error(
          "Speech error:",
          event
        );

        setPlaying(false);
        speechRef.current = null;

        setError(
          "Unable to play voice. Please try again."
        );
      };

      speechRef.current = speech;

      window.speechSynthesis.speak(speech);
    } catch (err) {
      console.error(
        "Voice playback error:",
        err
      );

      setPlaying(false);
      speechRef.current = null;

      setError(
        "Voice could not be played. Please try again."
      );
    }
  };

  // ========================================
  // UI
  // ========================================

  return (
    <div className="app">

      {/* ================= NAVBAR ================= */}

      <nav className="navbar">

        <div className="brand">

          <div className="brand-icon">
            📚
          </div>

          <div>
            <h1>StudyMate AI</h1>

            <p>
              Smart Document Assistant
            </p>
          </div>

        </div>

        <div className="ai-status">
          <span></span>
          AI Ready
        </div>

      </nav>

      {/* ================= MAIN ================= */}

      <main className="container">

        {/* ================= HERO ================= */}

        <section className="hero">

          <div className="pill">
            ✨ RAG Powered • Gemini AI • Voice
          </div>

          <h2>
            Learn smarter with
            <br />

            <span>
              your own study material.
            </span>
          </h2>

          <p>
            Upload your notes or syllabus,
            ask questions and get intelligent
            answers directly from your documents.
          </p>

        </section>

        {/* ================= UPLOAD CARD ================= */}

        <section className="card">

          <div className="card-heading">

            <div className="heading-icon">
              📄
            </div>

            <div>

              <h3>
                Upload your document
              </h3>

              <p>
                Upload a PDF to start studying
                with StudyMate.
              </p>

            </div>

          </div>

          <label className="upload-area">

            <input
              type="file"
              accept=".pdf,application/pdf"
              onChange={handleFile}
            />

            <div className="cloud">
              ☁️
            </div>

            {file ? (
              <>
                <strong>
                  {file.name}
                </strong>

                <small>
                  PDF selected
                </small>
              </>
            ) : (
              <>
                <strong>
                  Choose your PDF
                </strong>

                <small>
                  Click here to browse your files
                </small>
              </>
            )}

          </label>

          <button
            className="main-button"
            onClick={uploadPDF}
            disabled={!file || uploading}
          >
            {uploading
              ? "Processing document..."
              : "Upload & Process PDF"}
          </button>

          {uploaded && (
            <div className="success-message">
              ✓ Document processed successfully.
              You can ask questions now.
            </div>
          )}

        </section>

        {/* ================= ASK CARD ================= */}

        <section className="card">

          <div className="card-heading">

            <div className="heading-icon">
              💬
            </div>

            <div>

              <h3>
                Ask StudyMate
              </h3>

              <p>
                Ask anything related to your
                uploaded document.
              </p>

            </div>

          </div>

          <div className="question-wrapper">

            <textarea
              value={question}
              onChange={(e) =>
                setQuestion(e.target.value)
              }
              placeholder="Example: What is the main topic of this document?"
              disabled={asking}
            />

            <button
              className="ask-button"
              onClick={askAI}
              disabled={asking}
            >
              {asking
                ? "Thinking..."
                : "Ask AI →"}
            </button>

          </div>

          {asking && (
            <div className="loading">

              <div className="spinner"></div>

              <span>
                Searching your document and
                generating an answer...
              </span>

            </div>
          )}

          {error && (
            <div className="error-message">
              ⚠️ {error}
            </div>
          )}

        </section>

        {/* ================= ANSWER CARD ================= */}

        {answer && (
          <section className="answer-card">

            {/* ================= ANSWER HEADER ================= */}

            <div className="answer-top">

              <div>

                <span className="label">
                  AI RESPONSE
                </span>

                <h3>
                  🤖 StudyMate Answer
                </h3>

              </div>

            </div>

            {/* ================= ANSWER ================= */}

            <div className="answer-box">

              <ReactMarkdown>
                {answer}
              </ReactMarkdown>

            </div>

            {/* ================= ANSWER ACTIONS ================= */}

            <div className="answer-actions">

              <button
                className="copy-button"
                onClick={copyAnswer}
              >
                {copied
                  ? "✓ Copied!"
                  : "📋 Copy Answer"}
              </button>

              {audioUrl && (
                <button
                  className="voice-button"
                  onClick={playAudio}
                >
                  {playing
                    ? "⏹ Stop Voice"
                    : "🔊 Listen to Answer"}
                </button>
              )}

            </div>

            {/* ================= SOURCES ================= */}

            {sources.length > 0 && (
              <div className="sources">

                <div className="sources-header">

                  <div>
                    <h4>
                      📚 Sources
                    </h4>

                    <p>
                      Information retrieved from your document
                    </p>
                  </div>

                </div>

                <div className="sources-list">

                  {sources.map(
                    (source, index) => (

                      <div
                        className="source"
                        key={index}
                      >

                        <div className="source-number">
                          {index + 1}
                        </div>

                        <div className="source-content">

                          <strong>
                            Document Source {index + 1}
                          </strong>

                          <p>
                            {source}
                          </p>

                        </div>

                      </div>

                    )
                  )}

                </div>

              </div>
            )}

          </section>
        )}

      </main>

      {/* ================= FOOTER ================= */}

      <footer>

        <strong>
          StudyMate AI
        </strong>

        <span>
          {" "}• RAG Based Document Assistant
        </span>

      </footer>

    </div>
  );
}

export default App;