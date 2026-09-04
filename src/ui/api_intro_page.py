from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextBrowser, QPushButton, QHBoxLayout
from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices

class APIIntroPage(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)

        self.title = QLabel("<h2>🌐 AI API 接入指南（OpenAI 兼容）</h2>")
        self.layout.addWidget(self.title)

        self.content = QTextBrowser()
        self.content.setOpenExternalLinks(True)

        html = """
        <div style='font-family: sans-serif; line-height: 1.6;'>
            <p>本项目支持任何 <b>OpenAI 兼容 API</b>。您可以选择接入云端大模型服务，也可以使用本地模型（Ollama）。</p>
            <p>以下为常见的可选接入渠道，<b>排名不分先后，均非推荐</b>，请根据自身需求与所在地区自行选择：</p>

            <div style='background-color: rgba(33, 150, 243, 0.1); padding: 15px; border-radius: 8px;'>
                <h3 style='color: #2196F3;'>1. DeepSeek 官方</h3>
                <p><b>特点：</b> DeepSeek 官方 API，价格透明，提供 V3 / R1 等模型。</p>
                <p><b>官网：</b> <a href='https://platform.deepseek.com/'>https://platform.deepseek.com/</a></p>
            </div>

            <br>

            <div style='background-color: rgba(156, 39, 176, 0.1); padding: 15px; border-radius: 8px;'>
                <h3 style='color: #9C27B0;'>2. 阿里云百炼（DashScope）</h3>
                <p><b>特点：</b> 提供 Qwen 系列等多模型选择，新用户通常含免费额度，接口兼容 OpenAI 格式。</p>
                <p><b>官网：</b> <a href='https://www.aliyun.com/product/bailian'>https://www.aliyun.com/product/bailian</a></p>
            </div>

            <br>

            <div style='background-color: rgba(76, 175, 80, 0.1); padding: 15px; border-radius: 8px;'>
                <h3 style='color: #4CAF50;'>3. SiliconFlow (硅基流动)</h3>
                <p><b>特点：</b> 聚合 DeepSeek、Qwen、Llama 等开源模型，部分模型提供免费配额。</p>
                <p><b>官网：</b> <a href='https://siliconflow.cn/'>https://siliconflow.cn/</a></p>
            </div>

            <br>

            <div style='background-color: rgba(255, 152, 0, 0.1); padding: 15px; border-radius: 8px;'>
                <h3 style='color: #FF9800;'>4. OpenRouter</h3>
                <p><b>特点：</b> 国际聚合网关，一个 Key 访问多家模型，部分模型免费。</p>
                <p><b>官网：</b> <a href='https://openrouter.ai/'>https://openrouter.ai/</a></p>
            </div>

            <br>

            <div style='background-color: rgba(96, 125, 139, 0.1); padding: 15px; border-radius: 8px;'>
                <h3 style='color: #607D8B;'>5. 自建网关（One API 等开源项目）</h3>
                <p><b>特点：</b> 自托管聚合转发，统一管理多个上游 API Key，适合已有多个渠道的用户。</p>
                <p><b>项目地址：</b> <a href='https://github.com/songquanpeng/one-api'>https://github.com/songquanpeng/one-api</a></p>
            </div>

            <br>

            <div style='background-color: rgba(0, 188, 212, 0.1); padding: 15px; border-radius: 8px;'>
                <h3 style='color: #00BCD4;'>6. 本地模型（Ollama）</h3>
                <p><b>特点：</b> 完全本地运行，无需 API Key，隐私性好，但速度取决于本机硬件。</p>
                <p><b>官网：</b> <a href='https://ollama.com/'>https://ollama.com/</a></p>
            </div>

            <br>

            <div style='background-color: rgba(158, 158, 158, 0.1); padding: 15px; border-radius: 8px;'>
                <h3 style='color: #616161;'>通用设置方法</h3>
                <ul>
                    <li>在所选平台注册并创建 API Key。</li>
                    <li>在软件侧边栏选择 <b>API 网关</b>。</li>
                    <li>填写 API URL（通常形如 https://api.example.com/v1，具体以服务商文档为准）。</li>
                    <li>填入您的 API Key。</li>
                </ul>
            </div>

            <p style='color: gray; margin-top: 20px;'><i>注意：免费或低价 API 通常有并发限制（如并发 1），单次处理较慢属于正常现象。</i></p>
        </div>
        """
        self.content.setHtml(html)
        self.layout.addWidget(self.content)

        btn_layout = QHBoxLayout()
        btn_open = QPushButton("🌐 了解本地模型方案（Ollama 官网）")
        btn_open.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://ollama.com/")))
        btn_layout.addStretch()
        btn_layout.addWidget(btn_open)
        self.layout.addLayout(btn_layout)
