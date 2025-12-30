import os
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QPushButton, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt

class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video Analysis Pro - 使用说明与白皮书")
        self.resize(800, 700)
        
        layout = QVBoxLayout(self)
        
        # Markdown Content Browser
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setHtml(self._get_help_content())
        layout.addWidget(self.browser)
        
        # Bottom Buttons
        btn_layout = QHBoxLayout()
        btn_close = QPushButton("明白，开启智能视频之旅")
        btn_close.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; font-weight: bold;")
        btn_close.clicked.connect(self.accept)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _get_help_content(self):
        # We use HTML/Markdown hybrid for QTextBrowser
        return """
        <style>
            body { font-family: 'Microsoft YaHei', sans-serif; line-height: 1.6; }
            h1 { color: #2196F3; border-bottom: 2px solid #2196F3; padding-bottom: 10px; }
            h2 { color: #4CAF50; margin-top: 20px; }
            h3 { color: #FF9800; }
            code { background-color: rgba(128, 128, 128, 0.1); padding: 2px 4px; border-radius: 4px; font-family: Consolas; }
            .box { background-color: rgba(33, 150, 243, 0.1); border-left: 5px solid #2196F3; padding: 10px; margin: 10px 0; }
            .warning { background-color: rgba(255, 152, 0, 0.1); border-left: 5px solid #FF9800; padding: 10px; margin: 10px 0; }
        </style>
        
        <h1>🎬 Video Analysis Pro：小白从零到一上手指南</h1>
        
        <div class="box">
            <b>这是什么？</b><br>
            这款软件是一个“智能视频助理”。它能利用最新的人工智能（AI）技术，帮你自动看完视频、听完音频，并把视频里的画面内容和语音内容整理成一份详细的文字总结报告，甚至帮你剪辑出精彩片段。
        </div>

        <h2>一、 核心原理（大白话版）</h2>
        <p>软件的工作逻辑像是一个三步走的流水线：</p>
        <ol>
            <li><b>眼睛看（画面分析）：</b> 利用 <code>YOLO</code> 算法自动识别视频里出现的猫、狗、人、车等物体。</li>
            <li><b>耳朵听（语音转文字）：</b> 利用 <code>Whisper</code> 技术把视频里的说话声变成文字。</li>
            <li><b>大脑想（AI 总结）：</b> 把看到的画面信息和听到的文字信息一股脑塞给 <code>AI 大语言模型</code>（如 DeepSeek、GPT），让它写出一份人类看得懂的总结。</li>
        </ol>

        <h2>二、 快速开始：三步走</h2>
        <p>1. <b>选视频：</b> 点击顶部的“选择视频文件”按钮。<br>
           2. <b>载模型：</b> 在“模型配置”选一个你喜欢的 AI（初学者推荐用默认的 API 模式，又快又省事）。<br>
           3. <b>点开始：</b> 点击右下角的“🚀 启动分析”，然后泡杯咖啡，等它干完活。
        </p>

        <h2>三、 进阶：如何使用本地模型 (GGUF)？</h2>
        <div class="warning">
            <b>适合：</b> 有显卡、追求隐私、或者网络环境不佳的小伙伴。
        </div>
        <p>如果你想自己下载模型使用，操作极其简单：</p>
        <ul>
            <li>下载 <code>.gguf</code> 格式的模型文件（如从 HuggingFace 下载）。</li>
            <li>在软件所在的目录下，找到一个叫 <b><code>models</code></b> 的文件夹。</li>
            <li>把模型文件直接<b>丢进去</b>。</li>
            <li>回到软件首页，客户端类型选择 <b>“本地模型文件 (.gguf/.pt)”</b>，刷新一下就能看到它了！</li>
        </ul>

        <h2>四、 优势与不足</h2>
        <h3>✅ 好处</h3>
        <ul>
            <li><b>省时间：</b> 1 小时的课程或会议，3 分钟就能拿到摘要。</li>
            <li><b>高准确：</b> 音画同步分析，比单纯看文稿更准确。</li>
            <li><b>灵活性：</b> 支持各种 API，也支持完全断网在本地跑模型，保护隐私。</li>
        </ul>
        <h3>⚠️ 局限</h3>
        <ul>
            <li><b>硬件要求：</b> 本地跑大模型需要电脑显卡（GPU）稍微好一点，否则会比较慢。</li>
            <li><b>成本：</b> 使用在线 API 可能会产生少量费用（虽然很多都有免费额度）。</li>
        </ul>

        <h2>五、 安全性与实用性</h2>
        <p>本软件<b>完全开源且运行在本地</b>。除了你主动填写的 API 密钥会发给大模型服务商外，你的视频、音频数据<b>永远不会上传到我们的服务器</b>。对于政府、企业等高度敏感的场景，采用“本地模型”模式可以实现 100% 物理隔绝，绝对安全。</p>
        
        <p><i>祝您在使用过程中效率翻倍！如有疑问，请点击“模型管理”查看组件健康状态。</i></p>
        """
