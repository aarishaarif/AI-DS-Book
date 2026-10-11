import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  bookSidebar: [
    {
      type: 'doc',
      id: 'intro',
      label: '📖 Introduction',
    },
    {
      type: 'category',
      label: 'Module 1: Python Foundations',
      items: ['module-01-python-foundations'],
    },
    {
      type: 'category',
      label: 'Module 2: Python for Data Science',
      items: ['module-02-python-for-data-science'],
    },
    {
      type: 'category',
      label: 'Module 3: Statistics & Mathematics',
      items: ['module-03-statistics-and-mathematics'],
    },
    {
      type: 'category',
      label: 'Module 4: Machine Learning',
      items: ['module-04-machine-learning'],
    },
    {
      type: 'category',
      label: 'Module 5: Deep Learning',
      items: ['module-05-deep-learning'],
    },
    {
      type: 'category',
      label: 'Module 6: MLOps',
      items: ['module-06-mlops'],
    },
    {
      type: 'category',
      label: 'Module 7: Big Data & Cloud',
      items: ['module-07-big-data-and-cloud'],
    },
    {
      type: 'category',
      label: 'Module 8: Generative AI & LLMs',
      items: ['module-08-generative-ai-and-llms'],
    },
    {
      type: 'category',
      label: 'Module 9: Agentic AI',
      items: ['module-09-agentic-ai'],
    },
  ],
};

export default sidebars;
