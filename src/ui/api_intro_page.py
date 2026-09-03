from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextBrowser, QPushButton, QHBoxLayout
from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices

class APIIntroPage(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        self.title = QLabel("<h2>🌟 免费 AI API 获取指南</h2>")
        self.layout.addWidget(self.title)
        
        self.content = QTextBrowser()
        self.content.setOpenExternalLinks(True)
        
        html = """
        <div style='font-family: sans-serif; line-height: 1.6;'>
            <p>如果您不想使用本地模型（Ollama），可以通过 API 接入强大的云端大模型。以下是一些推荐的免费 API 来源：</p>
            
            <div style='background-color: rgba(33, 150, 243, 0.1); padding: 15px; border-radius: 8px;'>
                <h3 style='color: #2196F3;'>1. 心流开放平台 (iflow.cn) 🌟 推荐</h3>
                <p><b>特点：</b> 无需付费，支持多种大模型，响应速度快。</p>
                <p><b>获取地址：</b> <a href='https://platform.iflow.cn/'>https://platform.iflow.cn/</a></p>
                <p><b>设置方法：</b></p>
                <ul>
                    <li>注册并创建 API Key。</li>
                    <li>在软件侧边栏选择 <b>API 网关</b>。</li>
                    <li>填写 API URL (通常为 https://api.iflow.cn/v1)。</li>
                    <li>填入您的 API Key。</li>
                </ul>
            </div>
            
            <br>
            
            <div style='background-color: rgba(76, 175, 80, 0.1); padding: 15px; border-radius: 8px;'>
                <h3 style='color: #4CAF50;'>2. SiliconFlow (硅基流动)</h3>
                <p><b>特点：</b> 算力强劲，提供 DeepSeek, Llama3 等顶级开源模型免费配额。</p>
                <p><b>获取地址：</b> <a href='https://siliconflow.cn/'>https://siliconflow.cn/</a></p>
            </div>
            
            <br>
            
            <div style='background-color: rgba(255, 152, 0, 0.1); padding: 15px; border-radius: 8px;'>
                <h3 style='color: #FF9800;'>3. 其他白嫖渠道</h3>
                <p>您也可以关注 GitHub 上的开源 API 转发项目，搜索 <b>"One API Free"</b> 关键词。</p>
            </div>
            
            <p style='color: gray; margin-top: 20px;'><i>注意：免费 API 通常有并发限制（如并发1），单次处理较慢属于正常现象。</i></p>
        </div>
        """
        self.content.setHtml(html)
        self.layout.addWidget(self.content)
        
        btn_layout = QHBoxLayout()
        btn_open = QPushButton("🌐 打开心流平台官网")
        btn_open.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://platform.iflow.cn/")))
        btn_layout.addStretch()
        btn_layout.addWidget(btn_open)
        self.layout.addLayout(btn_layout)
