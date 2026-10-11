import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/chat',
    component: ComponentCreator('/chat', '4b9'),
    exact: true
  },
  {
    path: '/progress',
    component: ComponentCreator('/progress', '9ab'),
    exact: true
  },
  {
    path: '/projects',
    component: ComponentCreator('/projects', '21b'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', '80d'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', '5d3'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', '9e6'),
            routes: [
              {
                path: '/docs/intro',
                component: ComponentCreator('/docs/intro', 'cda'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/docs/module-01-python-foundations',
                component: ComponentCreator('/docs/module-01-python-foundations', '018'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/docs/module-02-python-for-data-science',
                component: ComponentCreator('/docs/module-02-python-for-data-science', 'bd1'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/docs/module-03-statistics-and-mathematics',
                component: ComponentCreator('/docs/module-03-statistics-and-mathematics', '514'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/docs/module-04-machine-learning',
                component: ComponentCreator('/docs/module-04-machine-learning', 'aa1'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/docs/module-05-deep-learning',
                component: ComponentCreator('/docs/module-05-deep-learning', '12d'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/docs/module-06-mlops',
                component: ComponentCreator('/docs/module-06-mlops', '801'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/docs/module-07-big-data-and-cloud',
                component: ComponentCreator('/docs/module-07-big-data-and-cloud', 'e30'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/docs/module-08-generative-ai-and-llms',
                component: ComponentCreator('/docs/module-08-generative-ai-and-llms', '7db'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/docs/module-09-agentic-ai',
                component: ComponentCreator('/docs/module-09-agentic-ai', 'e4e'),
                exact: true,
                sidebar: "bookSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/',
    component: ComponentCreator('/', 'e5f'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
