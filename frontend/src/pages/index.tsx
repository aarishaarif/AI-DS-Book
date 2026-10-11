import React from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import CodeBlock from '@theme/CodeBlock';

export default function Home(): JSX.Element {
  const { siteConfig } = useDocusaurusContext();

  return (
    <Layout
      title={`Welcome | ${siteConfig.title}`}
      description="AI & Data Science Learning Book - From Python foundations to Agentic AI"
    >
      <main className="container margin-vert--lg">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <Heading as="h1" className="hero__title">
              Welcome to {siteConfig.title}
            </Heading>
            <p className="hero__subtitle">
              {siteConfig.tagline}
            </p>

            <div className="homepage__features">
              <section className="features-section">
                <div className="feature">
                  <h3>Python Foundations</h3>
                  <p>Master programming concepts from variables to functions and control flow</p>
                </div>
                <div className="feature">
                  <h3>Data Science</h3>
                  <p>Learn NumPy, Pandas, and data visualization techniques</p>
                </div>
                <div className="feature">
                  <h3>Machine Learning</h3>
                  <p>Supervised, unsupervised, and advanced ML algorithms</p>
                </div>
                <div className="feature">
                  <h3>Deep Learning</h3>
                  <p>Neural networks, backpropagation, and modern architectures</p>
                </div>
                <div className="feature">
                  <h3>MLOps</h3>
                  <p>Production-ready ML pipelines and monitoring</p>
                </div>
                <div className="feature">
                  <h3>Agentic AI</h3>
                  <p>Autonomous agents, tool use, and multi-agent systems</p>
                </div>
              </section>
            </div>

            <div className="cta-section">
              <h2>Start Your Learning Journey</h2>
              <p>Begin with our comprehensive Python Foundations module or explore hands-on projects</p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem', marginTop: '1rem' }}>
                <a href="/docs/module-01-python-foundations" className="button button--primary button--lg">
                  Start Reading Book
                </a>
                <a href="/chat" className="button button--secondary button--lg">
                  Try RAG Chatbot
                </a>
                <a href="/projects" className="button button--outline button--primary button--lg">
                  Explore Projects
                </a>
                <a href="/progress" className="button button--outline button--secondary button--lg">
                  Track Progress
                </a>
              </div>
            </div>

            <div className="quick-start">
              <h2>Quick Start</h2>
              <CodeBlock language="python">{`# Your first Python program
print("Welcome to AI & Data Science!")

# Learn more about data science with numpy
import numpy as np
array = np.array([1, 2, 3, 4, 5])
print(f"Array sum: {np.sum(array)}")
`}</CodeBlock>
              <p>Need help? Ask our RAG chatbot about any topic in the book</p>
            </div>
          </div>
        </div>
      </main>
    </Layout>
  );
}