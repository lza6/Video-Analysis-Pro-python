import { Placeholder } from "@/components/shell/Placeholder";

export default function SettingsPage() {
  return (
    <Placeholder
      title="设置"
      desc="配置 LLM 提供商(Ollama / OpenAI 兼容网关 / NVIDIA)、API 预设、提示词模板。"
      origin="src/ui/provider_config_dialog.py + main_window.py sidebar + api_intro_page.py"
      capabilities={[
        "Provider 配置(URL/Key/模型),测活性不真实调付费 API",
        "API Key 走 OS 密钥环(失败降级 ini + 告警)",
        "API 预设管理(新增/保存/删除)",
        "提示词模板编辑",
        "获取 API 接入指南(DeepSeek/百炼/SiliconFlow/OpenRouter/Ollama)",
      ]}
    />
  );
}
