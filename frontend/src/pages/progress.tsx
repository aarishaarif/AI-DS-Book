import React, { useState, useEffect } from 'react';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import ProgressTracker from '../components/ProgressTracker';

export default function ProgressPage(): JSX.Element {
  const [bookmarks, setBookmarks] = useState<Array<{ title: string; url: string; addedAt: string }>>([]);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      try {
        const stored = localStorage.getItem('ai_ds_book_bookmarks');
        if (stored) {
          setBookmarks(JSON.parse(stored));
        }
      } catch (e) {
        console.error(e);
      }
    }
  }, []);

  const removeBookmark = (url: string) => {
    const updated = bookmarks.filter((b) => b.url !== url);
    setBookmarks(updated);
    if (typeof window !== 'undefined') {
      localStorage.setItem('ai_ds_book_bookmarks', JSON.stringify(updated));
    }
  };

  const addQuickBookmark = (title: string, url: string) => {
    if (bookmarks.some((b) => b.url === url)) return;
    const updated = [
      ...bookmarks,
      { title, url, addedAt: new Date().toLocaleDateString() },
    ];
    setBookmarks(updated);
    if (typeof window !== 'undefined') {
      localStorage.setItem('ai_ds_book_bookmarks', JSON.stringify(updated));
    }
  };

  return (
    <Layout
      title="Learning Progress & Bookmarks | AI & Data Science Book"
      description="Track your learning milestones across all 9 modules, test knowledge retention, and manage bookmarked lessons."
    >
      <main className="container margin-vert--lg" style={{ maxWidth: 900 }}>
        <header style={{ marginBottom: '2rem', textAlign: 'center' }}>
          <Heading as="h1">📊 Learning Progress & Bookmarks</Heading>
          <p style={{ color: 'var(--ifm-color-emphasis-700)', fontSize: '1.05rem' }}>
            Check off completed lessons, view your module completion percentages, and resume saved lessons. All progress is saved automatically in your browser.
          </p>
        </header>

        <section style={{ marginBottom: '2.5rem' }}>
          <ProgressTracker />
        </section>

        <section
          style={{
            background: 'var(--ifm-card-background-color, #ffffff)',
            border: '1px solid var(--ifm-color-emphasis-200, #e2e8f0)',
            borderRadius: 12,
            padding: '1.5rem',
            marginBottom: '2rem',
          }}
        >
          <Heading as="h2" style={{ fontSize: '1.3rem', marginBottom: '0.75rem' }}>
            📌 Saved Bookmarks
          </Heading>

          {bookmarks.length === 0 ? (
            <div>
              <p style={{ color: 'var(--ifm-color-emphasis-600)', marginBottom: '1rem' }}>
                You have not bookmarked any lessons yet. Quick add popular modules below:
              </p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                <button
                  className="button button--outline button--sm button--primary"
                  onClick={() =>
                    addQuickBookmark(
                      'Module 1: Python Foundations',
                      '/docs/module-01-python-foundations'
                    )
                  }
                >
                  + Bookmark Python Foundations
                </button>
                <button
                  className="button button--outline button--sm button--primary"
                  onClick={() =>
                    addQuickBookmark(
                      'Module 4: Machine Learning',
                      '/docs/module-04-machine-learning'
                    )
                  }
                >
                  + Bookmark Machine Learning
                </button>
                <button
                  className="button button--outline button--sm button--primary"
                  onClick={() =>
                    addQuickBookmark(
                      'Module 9: Agentic AI',
                      '/docs/module-09-agentic-ai'
                    )
                  }
                >
                  + Bookmark Agentic AI
                </button>
              </div>
            </div>
          ) : (
            <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
              {bookmarks.map((b, idx) => (
                <li
                  key={idx}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '0.75rem 0',
                    borderBottom: '1px solid var(--ifm-color-emphasis-200, #e2e8f0)',
                  }}
                >
                  <div>
                    <a
                      href={b.url}
                      style={{ fontWeight: 600, textDecoration: 'none' }}
                    >
                      {b.title}
                    </a>
                    <span
                      style={{
                        fontSize: '0.8rem',
                        color: 'var(--ifm-color-emphasis-500)',
                        marginLeft: '0.75rem',
                      }}
                    >
                      Added {b.addedAt}
                    </span>
                  </div>
                  <button
                    className="button button--link button--sm"
                    style={{ color: '#ef4444' }}
                    onClick={() => removeBookmark(b.url)}
                  >
                    Remove
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section style={{ textAlign: 'center', marginTop: '2rem' }}>
          <a href="/docs/intro" className="button button--primary button--lg" style={{ marginRight: '1rem' }}>
            Resume Reading Book →
          </a>
          <a href="/chat" className="button button--secondary button--lg">
            Ask RAG Chatbot →
          </a>
        </section>
      </main>
    </Layout>
  );
}
