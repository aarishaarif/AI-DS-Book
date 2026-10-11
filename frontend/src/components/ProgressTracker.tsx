import React, { useState, useEffect } from 'react';

export interface Module {
  id: string;
  title: string;
  lessons: number;
}

const MODULES: Module[] = [
  { id: 'module-01', title: 'Python Foundations', lessons: 6 },
  { id: 'module-02', title: 'Python for Data Science', lessons: 4 },
  { id: 'module-03', title: 'Statistics & Mathematics', lessons: 7 },
  { id: 'module-04', title: 'Machine Learning', lessons: 9 },
  { id: 'module-05', title: 'Deep Learning', lessons: 7 },
  { id: 'module-06', title: 'MLOps', lessons: 9 },
  { id: 'module-07', title: 'Big Data & Cloud', lessons: 6 },
  { id: 'module-08', title: 'Generative AI & LLMs', lessons: 7 },
  { id: 'module-09', title: 'Agentic AI', lessons: 7 },
];

const STORAGE_KEY = 'ai_ds_book_progress';

type ProgressMap = Record<string, Set<number>>;

function loadProgress(): ProgressMap {
  if (typeof window === 'undefined') return {};
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as Record<string, number[]>;
    return Object.fromEntries(
      Object.entries(parsed).map(([k, v]) => [k, new Set(v)])
    );
  } catch {
    return {};
  }
}

function saveProgress(progress: ProgressMap) {
  if (typeof window === 'undefined') return;
  const serialisable = Object.fromEntries(
    Object.entries(progress).map(([k, v]) => [k, [...v]])
  );
  localStorage.setItem(STORAGE_KEY, JSON.stringify(serialisable));
}

// ── Inline progress bar component ─────────────────────────────────────────

interface ProgressBarProps {
  percent: number;
  height?: number;
  color?: string;
}

function ProgressBar({ percent, height = 8, color = 'var(--ifm-color-primary)' }: ProgressBarProps) {
  return (
    <div style={{
      width: '100%',
      height,
      background: 'var(--ifm-color-emphasis-200, #e2e8f0)',
      borderRadius: 9999,
      overflow: 'hidden',
    }}>
      <div style={{
        height: '100%',
        width: `${Math.min(100, percent)}%`,
        background: percent >= 100
          ? 'linear-gradient(90deg, #059669, #10b981)'
          : `linear-gradient(90deg, ${color}, #818cf8)`,
        borderRadius: 9999,
        transition: 'width 0.5s ease',
      }} />
    </div>
  );
}

// ── Main ProgressTracker component ─────────────────────────────────────────

interface ProgressTrackerProps {
  /** Show only this specific module (for use inside a module page). */
  moduleId?: string;
  /** When true shows a compact single-bar summary of overall progress. */
  compact?: boolean;
}

export default function ProgressTracker({ moduleId, compact = false }: ProgressTrackerProps): JSX.Element {
  const [progress, setProgress] = useState<ProgressMap>({});

  useEffect(() => {
    setProgress(loadProgress());
  }, []);

  function toggleLesson(modId: string, lessonIdx: number) {
    setProgress(prev => {
      const next = { ...prev };
      if (!next[modId]) next[modId] = new Set();
      else next[modId] = new Set(next[modId]);

      if (next[modId].has(lessonIdx)) {
        next[modId].delete(lessonIdx);
      } else {
        next[modId].add(lessonIdx);
      }
      saveProgress(next);
      return next;
    });
  }

  function markAllComplete(modId: string, total: number) {
    setProgress(prev => {
      const next = { ...prev, [modId]: new Set(Array.from({ length: total }, (_, i) => i)) };
      saveProgress(next);
      return next;
    });
  }

  function clearAll() {
    setProgress({});
    if (typeof window !== 'undefined') localStorage.removeItem(STORAGE_KEY);
  }

  const totalLessons = MODULES.reduce((s, m) => s + m.lessons, 0);
  const completedLessons = MODULES.reduce((s, m) => s + (progress[m.id]?.size ?? 0), 0);
  const overallPct = Math.round((completedLessons / totalLessons) * 100);

  // ── Compact overall summary ──────────────────────────────────────────────
  if (compact) {
    return (
      <div style={{
        background: 'var(--card-bg, #fff)',
        border: '1px solid var(--card-border, #e2e8f0)',
        borderRadius: 12,
        padding: '1rem 1.25rem',
        margin: '1rem 0',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
          <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>Overall Progress</span>
          <span style={{ fontWeight: 700, color: 'var(--ifm-color-primary)', fontSize: '0.9rem' }}>
            {completedLessons} / {totalLessons} lessons
          </span>
        </div>
        <ProgressBar percent={overallPct} />
        <div style={{ marginTop: '0.4rem', fontSize: '0.8rem', color: '#64748b', textAlign: 'right' }}>
          {overallPct}% complete
        </div>
      </div>
    );
  }

  // ── Filter to single module if moduleId is specified ─────────────────────
  const modulesToShow = moduleId ? MODULES.filter(m => m.id === moduleId) : MODULES;

  return (
    <div>
      {/* Overall summary */}
      <div style={{
        background: 'var(--card-bg, #fff)',
        border: '1px solid var(--card-border, #e2e8f0)',
        borderRadius: 12,
        padding: '1.25rem 1.5rem',
        marginBottom: '1.5rem',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
          <span style={{ fontWeight: 700, fontSize: '1.05rem' }}>📚 Your Learning Progress</span>
          <span style={{ fontWeight: 700, color: 'var(--ifm-color-primary)' }}>
            {overallPct}%
          </span>
        </div>
        <ProgressBar percent={overallPct} height={10} />
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.5rem', fontSize: '0.82rem', color: '#64748b' }}>
          <span>{completedLessons} lessons completed</span>
          <span>{totalLessons - completedLessons} remaining</span>
        </div>
        {completedLessons > 0 && (
          <button
            onClick={clearAll}
            style={{ marginTop: '0.75rem', background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer', fontSize: '0.8rem', padding: 0, textDecoration: 'underline' }}
          >
            Reset all progress
          </button>
        )}
      </div>

      {/* Per-module cards */}
      {modulesToShow.map(mod => {
        const done = progress[mod.id]?.size ?? 0;
        const pct = Math.round((done / mod.lessons) * 100);

        return (
          <div key={mod.id} style={{
            background: 'var(--card-bg, #fff)',
            border: '1px solid var(--card-border, #e2e8f0)',
            borderRadius: 10,
            padding: '1rem 1.25rem',
            marginBottom: '0.75rem',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>
                {pct === 100 ? '✅ ' : ''}{mod.title}
              </span>
              <span style={{ fontSize: '0.8rem', color: '#64748b' }}>{done}/{mod.lessons}</span>
            </div>
            <ProgressBar percent={pct} height={6} />

            {/* Lesson checkboxes */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginTop: '0.75rem' }}>
              {Array.from({ length: mod.lessons }, (_, i) => {
                const checked = progress[mod.id]?.has(i) ?? false;
                return (
                  <button
                    key={i}
                    onClick={() => toggleLesson(mod.id, i)}
                    title={`Lesson ${i + 1}`}
                    style={{
                      width: 28,
                      height: 28,
                      borderRadius: 6,
                      border: `1.5px solid ${checked ? '#22c55e' : 'var(--card-border, #e2e8f0)'}`,
                      background: checked ? '#dcfce7' : 'transparent',
                      color: checked ? '#166534' : '#94a3b8',
                      cursor: 'pointer',
                      fontSize: '0.7rem',
                      fontWeight: 700,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      transition: 'all 0.15s',
                    }}
                  >
                    {checked ? '✓' : i + 1}
                  </button>
                );
              })}
              {done < mod.lessons && (
                <button
                  onClick={() => markAllComplete(mod.id, mod.lessons)}
                  style={{
                    padding: '0 0.6rem',
                    height: 28,
                    borderRadius: 6,
                    border: '1.5px solid var(--ifm-color-primary)',
                    background: 'transparent',
                    color: 'var(--ifm-color-primary)',
                    cursor: 'pointer',
                    fontSize: '0.7rem',
                    fontWeight: 600,
                  }}
                >
                  Mark all
                </button>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
