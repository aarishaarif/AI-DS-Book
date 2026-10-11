/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "AI & Data Science Learning Book",
  tagline: "An interactive journey from Python foundations to Agentic AI",
  favicon: "img/favicon.ico",

  url: "https://your-docusaurus-site.netlify.app",
  baseUrl: "/",

  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "warn",

  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          routeBasePath: "/",
          sidebarPath: require.resolve("./sidebars.js"),
          editUrl: "https://github.com/your-org/ai-ds-learning-book/edit/main/",
          showLastUpdateAuthor: true,
          showLastUpdateTime: true,
        },
        blog: false,
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: "AI & Data Science Book",
        logo: {
          alt: "AI & DS Book Logo",
          src: "img/logo.svg",
        },
        items: [
          {
            type: "docSidebar",
            sidebarId: "bookSidebar",
            position: "left",
            label: "Book",
          },
          {
            href: "/chat",
            label: "RAG Chatbot",
            position: "right",
          },
          {
            href: "https://github.com/your-org/ai-ds-learning-book",
            label: "GitHub",
            position: "right",
          },
        ],
      },
      footer: {
        style: "dark",
        links: [
          {
            title: "Book",
            items: [
              {
                label: "Start Here",
                to: "/intro",
              },
              {
                label: "Modules",
                to: "/module-01-python-foundations",
              },
            ],
          },
          {
            title: "Resources",
            items: [
              {
                label: "RAG Chatbot",
                to: "/chat",
              },
              {
                label: "Projects",
                to: "/projects",
              },
            ],
          },
          {
            title: "Connect",
            items: [
              {
                label: "GitHub",
                href: "https://github.com/your-org/ai-ds-learning-book",
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} AI & Data Science Learning Book. This is an independent educational project and is not officially affiliated with Saylani AI & Data Science or any other organization.`,
      },
      prism: {
        theme: require("prism-react-renderer/themes/vsLight"),
        darkTheme: require("prism-react-renderer/themes/dracula"),
        additionalLanguages: ["python", "javascript", "json", "bash", "sql", "dockerfile"],
      },
    }),

  themes: ["@docusaurus/theme-mermaid"],
  plugins: [
    [
      "@docusaurus/plugin-content-docs",
      {
        id: "chatbot",
        path: "chatbot",
        routeBasePath: "/chat",
        sidebarPath: require.resolve("./sidebars.js"),
      },
    ],
    "posthog-docusaurus",
  ],

  scripts: [],
};

module.exports = config;