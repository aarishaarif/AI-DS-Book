import React from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import CodeBlock from '@theme/CodeBlock';
import Link from '@docusaurus/Link';

const modules = [
  ['01', '🐍', 'Python Foundations', 'Build a strong base with syntax, data structures, functions, and files.', '/docs/module-01-python-foundations'],
  ['02', '📊', 'Data Science', 'Work with NumPy, Pandas, and visualisation workflows.', '/docs/module-02-python-for-data-science'],
  ['03', '∑', 'Math for ML', 'Understand probability, statistics, vectors, and optimisation.', '/docs/module-03-statistics-and-mathematics'],
  ['04', '🧠', 'Machine Learning', 'Train, evaluate, and improve practical ML models.', '/docs/module-04-machine-learning'],
  ['05', '✨', 'Deep Learning', 'Learn neural networks, CNNs, RNNs, and backpropagation.', '/docs/module-05-deep-learning'],
  ['06', '⚙️', 'MLOps', 'Move models from experiments into reliable production systems.', '/docs/module-06-mlops'],
  ['07', '☁️', 'Big Data & Cloud', 'Scale data workloads with Spark, storage, and streaming.', '/docs/module-07-big-data-and-cloud'],
  ['08', '💬', 'Generative AI', 'Explore transformers, embeddings, prompting, and RAG.', '/docs/module-08-generative-ai-and-llms'],
  ['09', '🤖', 'Agentic AI', 'Design agents with tools, memory, planning, and guardrails.', '/docs/module-09-agentic-ai'],
];

export default function Home(): JSX.Element {
  const { siteConfig } = useDocusaurusContext();

  return (
    <Layout
      title={`Welcome | ${siteConfig.title}`}
      description="AI & Data Science Learning Book - From Python foundations to Agentic AI"
    >
      <main className="home-page">
        <section className="hero-section">
          <div className="hero-orb hero-orb--one" />
          <div className="hero-orb hero-orb--two" />
          <span className="eyebrow">A practical path from code to intelligent systems</span>
          <Heading as="h1">Learn AI & Data Science by building.</Heading>
          <p>{siteConfig.tagline}. Short explanations, runnable examples, guided projects, and a book-grounded assistant.</p>
          <div className="hero-actions">
            <Link to="/docs/module-01-python-foundations" className="button button--primary button--lg">Start the book →</Link>
          </div>
          <div className="hero-stats">
            <div><strong>9</strong><span>learning modules</span></div>
            <div><strong>39+</strong><span>book-grounded chunks</span></div>
            <div><strong>3</strong><span>hands-on tools</span></div>
          </div>
        </section>

        <section className="home-section home-section--intro">
          <div>
            <span className="section-kicker">Your learning cockpit</span>
            <h2>A clear route through the whole stack</h2>
          </div>
          <p>Follow the modules in order, jump to a topic when you need it, or ask the assistant for a chapter-grounded explanation.</p>
        </section>

        <section className="modules-grid modules-grid--home">
          {modules.map(([number, icon, title, description, href]) => (
            <Link key={number} to={href} className="module-card module-card--home">
              <div className="module-card__top"><span className="module-card__icon">{icon}</span><span className="module-card__number">Module {number}</span></div>
              <h3 className="module-card__title">{title}</h3>
              <p className="module-card__description">{description}</p>
              <span className="module-card__link">Explore module →</span>
            </Link>
          ))}
        </section>

        <section className="home-lower-grid">
          <div className="quick-start home-panel">
            <span className="section-kicker">Start here</span>
            <h2>Your first learning loop</h2>
            <CodeBlock language="python" title="first_steps.py">{`# Your first Python program
print("Welcome to AI & Data Science!")

# Learn more about data science with numpy
import numpy as np
array = np.array([1, 2, 3, 4, 5])
print(f"Array sum: {np.sum(array)}")
`}</CodeBlock>
            <p className="panel-note">Read the explanation, run the example, then ask the RAG tutor to connect it to the next idea.</p>
          </div>
          <div className="home-panel home-panel--accent">
            <span className="section-kicker">Learn actively</span>
            <h2>Tools that keep you moving</h2>
            <div className="tool-list">
              <Link to="/projects"><span>🛠️</span><div><strong>Guided projects</strong><small>Turn concepts into portfolio work.</small></div></Link>
              <Link to="/progress"><span>✅</span><div><strong>Progress tracker</strong><small>Keep your learning momentum visible.</small></div></Link>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
