# PPT 版式模板参考（4 类）

参考 dashi-ppt 的版式组合思路：每页从以下 4 类模板中选型，封面/目录/内容/结尾。
纯 Markdown + HTML 双轨输出，无 LLM 也能按骨架渲染。

## 1. 封面页

```markdown
# <大标题>
## <副标题>
演讲人：<姓名> | <日期>
```

HTML 预览骨架：

```html
<header class="slide cover">
  <h1><大标题></h1>
  <p class="subtitle"><副标题></p>
  <p class="meta"><姓名> · <日期></p>
</header>
```

## 2. 目录页

```markdown
## 目录
1. <章节一>
2. <章节二>
3. <章节三>
```

HTML 预览骨架：

```html
<section class="slide toc">
  <h2>目录</h2>
  <ol>
    <li><章节一></li>
    <li><章节二></li>
  </ol>
</section>
```

## 3. 内容页（可复用，每章一页）

```markdown
## <小节标题>
- 要点 1（一句话结论）
- 要点 2（支撑数据）
| 指标 | 数值 |
|------|------|
```

HTML 预览骨架：

```html
<section class="slide content">
  <h2><小节标题></h2>
  <ul>
    <li><要点 1></li>
    <li><要点 2></li>
  </ul>
  <table><tr><th>指标</th><th>数值</th></tr></table>
</section>
```

## 4. 结尾页

```markdown
## 总结
- <核心结论>
## 行动项
- <下一步>
## 致谢
```

HTML 预览骨架：

```html
<section class="slide closing">
  <h2>总结</h2>
  <p><核心结论></p>
  <p class="thanks">谢谢</p>
</section>
```

## 组合规则

- 第一页必须用封面，第二页默认目录。
- 正文按大纲章节逐页选内容模板。
- 最后一页用结尾。
- 4 页以内可省略目录页。
