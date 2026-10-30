import React, {useEffect, useRef, useState} from 'react';
import ReactMarkdown from 'react-markdown';
import './FloatingChatbot.css';

type Citation = {title?: string; source_route?: string; file_path?: string; section?: string; source_section?: string};
type Message = {role: 'user' | 'assistant'; content: string; citations?: Citation[]};
const suggestions = ['What is gradient descent?', 'How does RAG work?', 'What is a Python list?'];

function citationUrl(citation: Citation): string {
  if (citation.source_route) return citation.source_route;
  const path = (citation.file_path || 'intro').replace(/^docs\//, '').replace(/\.mdx?$/, '');
  const heading = citation.section || citation.source_section;
  const anchor = heading ? `#${heading.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '')}` : '';
  return `/docs/${path}${anchor}`;
}

export default function FloatingChatbot(): JSX.Element {
  const [open, setOpen] = useState(false);
  const [minimized, setMinimized] = useState(false);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);
  const apiBase = (typeof process !== 'undefined' && process.env?.REACT_APP_API_URL) || 'http://localhost:8000';

  useEffect(() => { if (open && !minimized) inputRef.current?.focus(); }, [open, minimized]);

  async function ask(value = input) {
    const question = value.trim();
    if (!question || loading) return;
    setInput('');
    setMessages((current) => [...current, {role: 'user', content: question}]);
    setLoading(true);
    try {
      const response = await fetch(`${apiBase}/api/chat`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question, include_citations: true}),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = typeof data.detail === 'string'
          ? data.detail
          : `Backend returned HTTP ${response.status}`;
        throw new Error(detail);
      }
      setMessages((current) => [...current, {role: 'assistant', content: data.answer || 'No answer returned.', citations: data.citations || []}]);
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Unknown chat error';
      setMessages((current) => [...current, {role: 'assistant', content: `The book assistant is unavailable: ${detail}`}]);
    } finally { setLoading(false); }
  }

  return (
    <div className="floating-chatbot">
      {open && !minimized && (
        <section className="floating-chat-panel" aria-label="Book assistant">
          <header className="floating-chat-header">
            <strong>Book Assistant</strong>
            <div><button aria-label="Minimize chatbot" onClick={() => setMinimized(true)}>−</button><button aria-label="Close chatbot" onClick={() => setOpen(false)}>×</button></div>
          </header>
          <div className="floating-chat-messages" aria-live="polite">
            {messages.length === 0 && <p className="floating-chat-hint">Ask about a lesson in the book.</p>}
            {messages.map((message, index) => <div key={index} className={`floating-message ${message.role}`}>
              <ReactMarkdown>{message.content}</ReactMarkdown>
              {message.citations?.map((citation, citationIndex) => <a key={citationIndex} href={citationUrl(citation)} className="floating-source">📚 {citation.title || citation.source_section || 'Book source'}</a>)}
            </div>)}
            {loading && <div className="floating-message assistant" aria-label="Loading">Searching the book…</div>}
          </div>
          <div className="floating-suggestions">{suggestions.map((suggestion) => <button key={suggestion} onClick={() => ask(suggestion)} disabled={loading}>{suggestion}</button>)}</div>
          <form className="floating-chat-form" onSubmit={(event) => {event.preventDefault(); void ask();}}>
            <input ref={inputRef} value={input} onChange={(event) => setInput(event.target.value)} placeholder="Ask about the book" aria-label="Question" disabled={loading} />
            <button type="submit" disabled={loading || !input.trim()}>Send</button>
          </form>
        </section>
      )}
      {open && minimized && <button className="floating-chat-restore" onClick={() => setMinimized(false)}>Open assistant</button>}
      {!open && <button className="floating-chat-launcher" aria-label="Open book assistant" onClick={() => {setOpen(true); setMinimized(false);}}>💬</button>}
    </div>
  );
}
