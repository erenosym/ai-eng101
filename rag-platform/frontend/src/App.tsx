import { useState } from "react";
import "./App.css";

type Source = {
  filename: string | null;
  chunk_index: number | null;
  score: number;
};

type AskResponse = {
  question: string;
  answer: string;
  sources: Source[];
};

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
};

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [uploadMessage, setUploadMessage] = useState("");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);

  const uploadDocument = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setUploadMessage("");

    try {
      const response = await fetch(
        "http://127.0.0.1:8001/documents/parse",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error("Upload failed");
      }

      const data = await response.json();

      setUploadMessage(
        `${data.filename} indexed successfully — ${data.chunk_count} chunks`
      );
    } catch (error) {
      console.error(error);
      setUploadMessage("Upload failed.");
    } finally {
      setLoading(false);
    }
  };

  const askQuestion = async () => {
    if (!question.trim()) return;

    const currentQuestion = question;

    // User mesajını ekle.
    setMessages((prev) => [
      ...prev,
      {
        role: "user",
        content: currentQuestion,
      },
      {
        role: "assistant",
        content: "",
        sources: [],
      },
    ]);

    setQuestion("");
    setLoading(true);

    try {
      const response = await fetch(
        "http://127.0.0.1:8001/ask/stream",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            query: currentQuestion,
          }),
        }
      );

      if (!response.ok) {
        throw new Error("Request failed");
      }

      if (!response.body) {
        throw new Error("Streaming is not supported.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();

        if (done) break;

        buffer += decoder.decode(value, {
          stream: true,
        });

        const lines = buffer.split("\n");

        // Son satır yarım kalmış olabilir.
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.trim()) continue;

          const event = JSON.parse(line);

          if (event.type === "sources") {
            setMessages((prev) => {
              const updated = [...prev];

              const lastIndex = updated.length - 1;

              updated[lastIndex] = {
                ...updated[lastIndex],
                sources: event.sources,
              };

              return updated;
            });
          }

          if (event.type === "token") {
            setMessages((prev) => {
              const updated = [...prev];

              const lastIndex = updated.length - 1;

              const currentMessage =
                updated[lastIndex];

              updated[lastIndex] = {
                ...currentMessage,
                content:
                  currentMessage.content +
                  event.content,
              };

              return updated;
            });
          }
        }
      }
    } catch (error) {
      console.error(error);

      setMessages((prev) => {
        const updated = [...prev];

        const lastIndex = updated.length - 1;

        updated[lastIndex] = {
          role: "assistant",
          content:
            "Something went wrong while generating the answer.",
        };

        return updated;
      });
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (
    event: React.KeyboardEvent<HTMLTextAreaElement>
  ) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();

      if (!loading) {
        askQuestion();
      }
    }
  };

  return (
    <main className="container">
      <header className="header">
        <h1>AI Knowledge Assistant</h1>

        <p className="subtitle">
          Upload a document and ask grounded questions about it.
        </p>
      </header>

      <section className="card">
        <h2>Upload document</h2>

        <input
          type="file"
          accept=".pdf,.docx,.pptx"
          onChange={(event) =>
            setFile(event.target.files?.[0] ?? null)
          }
        />

        <button
          onClick={uploadDocument}
          disabled={!file || loading}
        >
          Upload
        </button>

        {uploadMessage && (
          <p className="success">{uploadMessage}</p>
        )}
      </section>

      <section className="card">
        <h2>Ask a question</h2>

        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask something about your document..."
        />

        <button
          onClick={askQuestion}
          disabled={!question.trim() || loading}
        >
          Ask
        </button>
      </section>

      <section className="card chat">
        <h2>Conversation</h2>

        {messages.length === 0 && (
          <p className="empty">
            Upload a document and ask your first question.
          </p>
        )}

        {messages.map((message, index) => (
          <div
            key={index}
            className={`message ${message.role}`}
          >
            <strong className="message-role">
              {message.role === "user" ? "You" : "Assistant"}
            </strong>

           <p>
           {message.content ||
            (message.role === "assistant"
             ? "Thinking..."
             : "")}
           </p>

            {message.sources &&
              message.sources.length > 0 && (
                <details className="debug-panel">
                  <summary>
                    Retrieval debug ({message.sources.length} chunks)
                  </summary>

                  <div className="sources">
                    {message.sources.map(
                      (source, sourceIndex) => (
                        <div
                          className="source"
                          key={sourceIndex}
                        >
                          <div>
                            <strong>
                              {source.filename ?? "Unknown document"}
                            </strong>
                          </div>

                          <div>
                            Chunk:{" "}
                            {source.chunk_index ?? "N/A"}
                          </div>

                          <div>
                            Similarity:{" "}
                            {source.score.toFixed(3)}
                          </div>
                        </div>
                      )
                    )}
                  </div>
                </details>
              )}
          </div>
        ))}
      </section>
    </main>
  );
}

export default App;