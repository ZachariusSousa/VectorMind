import { useState } from "react";

const API_BASE = "http://localhost:8000";

function App() {
  // Ingest form state
  const [ingestPath, setIngestPath] = useState("backend");
  const [ingestCollection, setIngestCollection] = useState("rag-self");
  const [ingestStatus, setIngestStatus] = useState("");

  // Query form state
  const [question, setQuestion] = useState("");
  const [queryCollection, setQueryCollection] = useState("rag-self");
  const [answer, setAnswer] = useState("");
  const [isAsking, setIsAsking] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);

  const handleIngest = async (e) => {
    e.preventDefault();
    setIngestStatus("");
    setIsIngesting(true);
    try {
      const res = await fetch(`${API_BASE}/api/ingest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          path: ingestPath,
          collection: ingestCollection,
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Ingestion failed");
      }
      setIngestStatus(data.message || "Ingestion complete.");
    } catch (err) {
      setIngestStatus(`Error: ${err.message}`);
    } finally {
      setIsIngesting(false);
    }
  };

  const handleAsk = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    setIsAsking(true);
    setAnswer("");
    try {
      const res = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          collection: queryCollection,
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Query failed");
      }
      setAnswer(data.answer);
    } catch (err) {
      setAnswer(`Error: ${err.message}`);
    } finally {
      setIsAsking(false);
    }
  };

  return (
    <div style={{ maxWidth: "900px", margin: "0 auto", padding: "1.5rem", fontFamily: "system-ui, sans-serif" }}>
      <h1 style={{ marginBottom: "0.5rem" }}>VectorMind Local RAG</h1>
      <p style={{ marginBottom: "1.5rem", opacity: 0.8, fontSize: "0.9rem" }}>
        Privacy-focused local code search. All indexing and queries stay on your machine.
      </p>

      {/* Ingest section */}
      <section
        style={{
          border: "1px solid #444",
          borderRadius: "0.75rem",
          padding: "1rem 1.25rem",
          marginBottom: "1.5rem",
        }}
      >
        <h2 style={{ marginTop: 0 }}>Ingest a directory</h2>
        <form onSubmit={handleIngest}>
          <div style={{ marginBottom: "0.75rem" }}>
            <label style={{ display: "block", marginBottom: "0.25rem" }}>
              Root path to index
            </label>
            <input
              type="text"
              value={ingestPath}
              onChange={(e) => setIngestPath(e.target.value)}
              style={{ width: "100%", padding: "0.4rem", borderRadius: "0.5rem", border: "1px solid #555" }}
            />
          </div>

          <div style={{ marginBottom: "0.75rem" }}>
            <label style={{ display: "block", marginBottom: "0.25rem" }}>
              Collection name
            </label>
            <input
              type="text"
              value={ingestCollection}
              onChange={(e) => setIngestCollection(e.target.value)}
              style={{ width: "100%", padding: "0.4rem", borderRadius: "0.5rem", border: "1px solid #555" }}
            />
          </div>

          <button
            type="submit"
            disabled={isIngesting}
            style={{
              padding: "0.45rem 0.9rem",
              borderRadius: "999px",
              border: "none",
              cursor: "pointer",
              fontWeight: 600,
              opacity: isIngesting ? 0.7 : 1,
            }}
          >
            {isIngesting ? "Indexing..." : "Ingest"}
          </button>
        </form>

        {ingestStatus && (
          <p style={{ marginTop: "0.75rem", fontSize: "0.9rem" }}>{ingestStatus}</p>
        )}
      </section>

      {/* Query section */}
      <section
        style={{
          border: "1px solid #444",
          borderRadius: "0.75rem",
          padding: "1rem 1.25rem",
        }}
      >
        <h2 style={{ marginTop: 0 }}>Ask your code</h2>
        <form onSubmit={handleAsk}>
          <div style={{ marginBottom: "0.75rem" }}>
            <label style={{ display: "block", marginBottom: "0.25rem" }}>
              Collection name
            </label>
            <input
              type="text"
              value={queryCollection}
              onChange={(e) => setQueryCollection(e.target.value)}
              style={{ width: "100%", padding: "0.4rem", borderRadius: "0.5rem", border: "1px solid #555" }}
            />
          </div>

          <div style={{ marginBottom: "0.75rem" }}>
            <label style={{ display: "block", marginBottom: "0.25rem" }}>
              Question
            </label>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              rows={3}
              style={{ width: "100%", padding: "0.4rem", borderRadius: "0.5rem", border: "1px solid #555", resize: "vertical" }}
            />
          </div>

          <button
            type="submit"
            disabled={isAsking}
            style={{
              padding: "0.45rem 0.9rem",
              borderRadius: "999px",
              border: "none",
              cursor: "pointer",
              fontWeight: 600,
              opacity: isAsking ? 0.7 : 1,
            }}
          >
            {isAsking ? "Thinking..." : "Ask"}
          </button>
        </form>

        {answer && (
          <div
            style={{
              marginTop: "1rem",
              padding: "0.75rem 0.9rem",
              borderRadius: "0.5rem",
              backgroundColor: "#111",
              border: "1px solid #333",
              whiteSpace: "pre-wrap",
              fontSize: "0.95rem",
            }}
          >
            {answer}
          </div>
        )}
      </section>
    </div>
  );
}

export default App;
