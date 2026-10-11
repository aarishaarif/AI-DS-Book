import React, { useState, useEffect, useRef } from 'react';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import ReactMarkdown from 'react-markdown';
import './chat.css';

interface Citation {
  title?: string;
  source_title?: string;
  module?: string;
  source_module?: string;
  section?: string;
  source_section?: string;
  lesson?: string;
  file_path?: string;
  source_file?: string;
  relevance_score?: number;
  text_excerpt?: string;
  chunk_id?: string;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  evidenceMet?: boolean;
}

const DEFAULT_SUGGESTIONS = [
  "What is the difference between a list and a tuple in Python?",
  "How does broadcasting work in NumPy?",
  "What is gradient descent and how is it used in ML?",
  "Explain backpropagation in deep neural networks",
  "What is the ReAct prompting framework in Agentic AI?",
  "How does Apache Spark handle distributed data?",
];

function getDocUrl(citation: Citation): string {
  const rawFile = citation.file_path || citation.source_file || '';
  const cleanName = rawFile.replace(/\.mdx?$/, '').replace(/^docs\//, '');
  const section = citation.section || citation.source_section || '';
  let anchor = '';
  if (section) {
    anchor = '#' + section.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
  }
  if (!cleanName || cleanName === 'unknown') {
    return '/docs/intro';
  }
  return `/docs/${cleanName}${anchor}`;
}

export default function ChatbotPage(): JSX.Element {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content:
        "👋 **Welcome to the AI & Data Science Book Assistant!**\n\nI answer questions **strictly grounded in the 9-module curriculum**. Every response includes citations linking directly to the relevant book chapters.\n\nAsk me about Python, Data Science, Statistics, Machine Learning, Deep Learning, MLOps, Big Data, LLMs, or Agentic AI!",
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const apiBaseUrl =
    (typeof process !== 'undefined' && process.env?.REACT_APP_API_URL) ||
    'http://localhost:8000';

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSendMessage = async (textToSend?: string) => {
    const question = (textToSend || inputValue).trim();
    if (!question || isLoading) return;

    setErrorMsg(null);
    const userMessage: Message = { role: 'user', content: question };
    setMessages((prev) => [...prev, userMessage]);
    if (!textToSend) setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch(`${apiBaseUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, include_citations: true }),
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const data = await response.json();
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.answer || "No response generated.",
        citations: data.citations || [],
        evidenceMet: data.evidence_threshold_met,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err: any) {
      console.error('Chat error:', err);
      setErrorMsg(
        'Unable to connect to the RAG backend. Ensure the FastAPI service is running at ' +
          apiBaseUrl
      );
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content:
            "I couldn't reach the RAG backend service. Please check your backend connection or ensure the API server is running.",
          evidenceMet: false,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearChat = () => {
    setMessages([
      {
        role: 'assistant',
        content:
          "Conversation restarted. Ask me any question grounded in the AI & Data Science Learning Book!",
      },
    ]);
    setErrorMsg(null);
  };

  return (
    <Layout
      title="RAG Chatbot | AI & Data Science Learning Book"
      description="Book-grounded AI assistant for AI & Data Science learning with clickable citations"
    >
      <main className="chatbot-wrapper">
        <header className="chatbot-header-card">
          <div className="header-meta">
            <span className="badge-tag">Grounded RAG</span>
            <Heading as="h1" className="chatbot-title">
              Curriculum RAG Assistant
            </Heading>
            <p className="chatbot-desc">
              Ask anything about the book. All responses are derived strictly from book content with verified citations.
            </p>
          </div>
          <button
            onClick={handleClearChat}
            className="btn-reset"
            title="Start new conversation"
          >
            🔄 New Conversation
          </button>
        </header>

        <section className="suggestions-bar">
          <span className="suggestions-label">Suggested Questions:</span>
          <div className="suggestions-list">
            {DEFAULT_SUGGESTIONS.map((q, idx) => (
              <button
                key={idx}
                className="chip-suggestion"
                onClick={() => handleSendMessage(q)}
                disabled={isLoading}
              >
                {q}
              </button>
            ))}
          </div>
        </section>

        {errorMsg && (
          <div className="chat-error-banner">
            ⚠️ {errorMsg}
          </div>
        )}

        <section className="chat-thread-container">
          {messages.map((msg, index) => {
            const isUser = msg.role === 'user';
            return (
              <article
                key={index}
                className={`chat-bubble-row ${isUser ? 'user-row' : 'assistant-row'}`}
              >
                <div className={`avatar-icon ${isUser ? 'user-avatar' : 'bot-avatar'}`}>
                  {isUser ? '👤' : '🤖'}
                </div>
                <div className="bubble-content-card">
                  <div className="bubble-markdown">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>

                  {msg.citations && msg.citations.length > 0 && (
                    <div className="citation-container">
                      <div className="citation-title">
                        <span>📚 Verified Book Sources ({msg.citations.length})</span>
                      </div>
                      <div className="citation-grid">
                        {msg.citations.map((cite, cIdx) => {
                          const docUrl = getDocUrl(cite);
                          const title = cite.title || cite.source_title || 'Book Section';
                          const section = cite.section || cite.source_section || '';
                          const score = cite.relevance_score;

                          return (
                            <a
                              key={cIdx}
                              href={docUrl}
                              className="citation-card-link"
                              target="_self"
                            >
                              <div className="citation-head">
                                <span className="citation-index">[{cIdx + 1}]</span>
                                <span className="citation-name">{title}</span>
                                {score !== undefined && (
                                  <span className="citation-score">
                                    {Math.round(score * 100)}% match
                                  </span>
                                )}
                              </div>
                              {section && (
                                <div className="citation-section-text">{section}</div>
                              )}
                              {cite.text_excerpt && (
                                <p className="citation-excerpt">
                                  "{cite.text_excerpt.slice(0, 140)}..."
                                </p>
                              )}
                              <span className="citation-action">Read in book →</span>
                            </a>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              </article>
            );
          })}

          {isLoading && (
            <div className="chat-bubble-row assistant-row">
              <div className="avatar-icon bot-avatar">🤖</div>
              <div className="bubble-content-card loading-card">
                <div className="dot-pulse">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
                <span className="loading-label">Retrieving evidence from book...</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </section>

        <footer className="chat-footer-form">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSendMessage();
            }}
            className="input-group"
          >
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask a question about any lesson or concept in the book..."
              disabled={isLoading}
              className="chat-text-input"
            />
            <button
              type="submit"
              disabled={isLoading || !inputValue.trim()}
              className="chat-submit-btn"
            >
              {isLoading ? 'Searching...' : 'Ask'}
            </button>
          </form>
          <div className="input-hint">
            Answers are grounded strictly in the 9 learning modules. Unsupported questions outside the curriculum are politely refused.
          </div>
        </footer>
      </main>
    </Layout>
  );
}
