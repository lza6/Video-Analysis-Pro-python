import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'zh-CN',
  title: 'TingFeng Hermes 文档',
  description: 'TingFeng Hermes（听风·赫尔墨斯）通用全能 Agent 桌面平台官方文档',
  lastUpdated: true,
  cleanUrls: true,

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '快速开始', link: '/guide/getting-started' },
      { text: '架构', link: '/guide/architecture' },
      { text: '插件开发', link: '/guide/plugin-development' },
      { text: 'API 参考', link: '/guide/api-reference' },
      { text: 'GitHub', link: 'https://github.com/lza6/Video-Analysis-Pro-python' }
    ],

    sidebar: {
      '/guide/': [
        {
          text: '入门',
          collapsed: false,
          items: [
            { text: '快速开始', link: '/guide/getting-started' },
            { text: '架构总览', link: '/guide/architecture' }
          ]
        },
        {
          text: '开发者',
          collapsed: false,
          items: [
            { text: '插件开发', link: '/guide/plugin-development' },
            { text: 'API 参考', link: '/guide/api-reference' }
          ]
        }
      ]
    },

    outline: { level: [2, 3], label: '本页导航' },
    docFooter: { prev: '上一页', next: '下一页' },
    lastUpdatedText: '最后更新',
    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '菜单',
    darkModeSwitchLabel: '主题',
    socialLinks: [
      { icon: 'github', link: 'https://github.com/lza6/Video-Analysis-Pro-python' }
    ]
  }
})
