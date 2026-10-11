import React, { useState } from 'react';

export interface QuizOption {
  text: string;
  isCorrect: boolean;
}

export interface QuizQuestion {
  question: string;
  options: QuizOption[];
  explanation?: string;
}

interface QuizComponentProps {
  questions: QuizQuestion[];
  title?: string;
}

type AnswerState = 'unanswered' | 'correct' | 'incorrect';

interface QuestionState {
  selected: number | null;
  state: AnswerState;
  revealed: boolean;
}

export default function QuizComponent({ questions, title = 'Quiz' }: QuizComponentProps): JSX.Element {
  const [questionStates, setQuestionStates] = useState<QuestionState[]>(
    questions.map(() => ({ selected: null, state: 'unanswered', revealed: false }))
  );
  const [submitted, setSubmitted] = useState(false);

  const score = questionStates.filter(q => q.state === 'correct').length;
  const allAnswered = questionStates.every(q => q.selected !== null);

  function selectOption(qIdx: number, optIdx: number) {
    if (submitted || questionStates[qIdx].revealed) return;
    setQuestionStates(prev => {
      const next = [...prev];
      next[qIdx] = { ...next[qIdx], selected: optIdx };
      return next;
    });
  }

  function submitQuiz() {
    if (!allAnswered) return;
    setQuestionStates(prev =>
      prev.map((qs, i) => ({
        ...qs,
        state: questions[i].options[qs.selected!].isCorrect ? 'correct' : 'incorrect',
        revealed: true,
      }))
    );
    setSubmitted(true);
  }

  function resetQuiz() {
    setQuestionStates(questions.map(() => ({ selected: null, state: 'unanswered', revealed: false })));
    setSubmitted(false);
  }

  return (
    <div className="quiz-container" style={{ margin: '1.5rem 0' }}>
      <h3 style={{ marginBottom: '1rem', color: 'var(--ifm-color-primary)', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
        📝 {title}
      </h3>

      {questions.map((q, qIdx) => {
        const qs = questionStates[qIdx];
        return (
          <div key={qIdx} style={{ marginBottom: '1.5rem' }}>
            <p className="quiz-question" style={{ fontWeight: 600, marginBottom: '0.75rem' }}>
              {qIdx + 1}. {q.question}
            </p>
            <ul className="quiz-options" style={{ listStyle: 'none', padding: 0, margin: 0 }}>
              {q.options.map((opt, optIdx) => {
                let borderColor = 'var(--card-border, #e2e8f0)';
                let background = 'var(--ifm-background-color)';
                let color = 'inherit';

                if (qs.revealed) {
                  if (opt.isCorrect) {
                    borderColor = '#22c55e';
                    background = '#f0fdf4';
                    color = '#166534';
                  } else if (qs.selected === optIdx && !opt.isCorrect) {
                    borderColor = '#ef4444';
                    background = '#fef2f2';
                    color = '#991b1b';
                  }
                } else if (qs.selected === optIdx) {
                  borderColor = 'var(--ifm-color-primary)';
                  background = 'var(--ifm-color-primary-lightest, #dbeafe)';
                }

                return (
                  <li
                    key={optIdx}
                    onClick={() => selectOption(qIdx, optIdx)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.75rem',
                      padding: '0.65rem 1rem',
                      border: `1.5px solid ${borderColor}`,
                      borderRadius: '8px',
                      cursor: submitted ? 'default' : 'pointer',
                      transition: 'all 0.15s ease',
                      marginBottom: '0.4rem',
                      background,
                      color,
                      userSelect: 'none',
                    }}
                  >
                    <span style={{
                      width: '22px',
                      height: '22px',
                      borderRadius: '50%',
                      border: `2px solid ${borderColor}`,
                      display: 'inline-flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '0.7rem',
                      fontWeight: 700,
                      flexShrink: 0,
                      background: qs.selected === optIdx ? 'var(--ifm-color-primary)' : 'transparent',
                      color: qs.selected === optIdx ? '#fff' : 'inherit',
                    }}>
                      {String.fromCharCode(65 + optIdx)}
                    </span>
                    {opt.text}
                    {qs.revealed && opt.isCorrect && (
                      <span style={{ marginLeft: 'auto', fontSize: '1rem' }}>✅</span>
                    )}
                    {qs.revealed && qs.selected === optIdx && !opt.isCorrect && (
                      <span style={{ marginLeft: 'auto', fontSize: '1rem' }}>❌</span>
                    )}
                  </li>
                );
              })}
            </ul>

            {qs.revealed && q.explanation && (
              <div style={{
                marginTop: '0.5rem',
                padding: '0.65rem 1rem',
                background: '#eff6ff',
                border: '1px solid #bfdbfe',
                borderRadius: '6px',
                fontSize: '0.875rem',
                color: '#1e40af',
              }}>
                💡 {q.explanation}
              </div>
            )}
          </div>
        );
      })}

      <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', marginTop: '1rem' }}>
        {!submitted ? (
          <button
            onClick={submitQuiz}
            disabled={!allAnswered}
            style={{
              background: allAnswered ? 'var(--ifm-color-primary)' : '#9ca3af',
              color: '#fff',
              border: 'none',
              borderRadius: '8px',
              padding: '0.6rem 1.5rem',
              fontWeight: 600,
              cursor: allAnswered ? 'pointer' : 'not-allowed',
              fontSize: '0.9rem',
              transition: 'background 0.2s',
            }}
          >
            Submit Answers
          </button>
        ) : (
          <>
            <div style={{
              fontWeight: 700,
              fontSize: '1rem',
              color: score === questions.length ? '#059669' : score >= questions.length / 2 ? '#d97706' : '#dc2626',
            }}>
              Score: {score} / {questions.length}
              {score === questions.length && ' 🎉'}
            </div>
            <button
              onClick={resetQuiz}
              style={{
                background: 'transparent',
                color: 'var(--ifm-color-primary)',
                border: '1.5px solid var(--ifm-color-primary)',
                borderRadius: '8px',
                padding: '0.5rem 1.2rem',
                fontWeight: 600,
                cursor: 'pointer',
                fontSize: '0.875rem',
              }}
            >
              Try Again
            </button>
          </>
        )}
      </div>
    </div>
  );
}
