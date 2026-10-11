import { themes as prismThemes } from 'prism-react-renderer';
import type { Config } from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'AI & Data Science Learning Book',
  tagline: 'An interactive journey from Python foundations to Agentic AI',
  favicon: 'img/favicon.ico',

  url: 'https://your-docusaurus-site.example.com',
  baseUrl: '/',

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: '/docs',
          editUrl: 'https://github.com/your-org/ai-ds-learning-book/edit/main/frontend/',
          showLastUpdateAuthor: false,
          showLastUpdateTime: false,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/social-card.png',
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'AI & DS Book',
      logo: {
        alt: 'AI & DS Book Logo',
        src: 'img/logo.svg',
        srcDark: 'img/logo-dark.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'bookSidebar',
          position: 'left',
          label: 'Book',
        },
        {
          to: '/projects',
          label: 'Projects',
          position: 'left',
        },
        {
          to: '/progress',
          label: 'Progress',
          position: 'left',
        },
        {
          to: '/chat',
          label: 'RAG Chatbot',
          position: 'right',
        },
        {
          href: 'https://github.com/your-org/ai-ds-learning-book',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Book',
          items: [
            {
              label: 'Introduction',
              to: '/docs/intro',
            },
            {
              label: 'Python Foundations',
              to: '/docs/module-01-python-foundations',
            },
            {
              label: 'Projects Catalog',
              to: '/projects',
            },
          ],
        },
        {
          title: 'Learning Tools',
          items: [
            {
              label: 'RAG Chatbot',
              to: '/chat',
            },
            {
              label: 'Progress Tracker',
              to: '/progress',
            },
          ],
        },
        {
          title: 'Connect',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/your-org/ai-ds-learning-book',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} AI & Data Science Learning Book. Independent educational project — not affiliated with Saylani.`,
    },
    prism: {
      theme: prismThemes.vsLight,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'json', 'sql', 'docker'],
    },
    mermaid: {
      theme: { light: 'neutral', dark: 'forest' },
    },
  } satisfies Preset.ThemeConfig,

  themes: ['@docusaurus/theme-mermaid'],
};

export default config;
