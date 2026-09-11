# Create GitHub Release for v10.3.1 (+ upload Setup.exe)
# 用法: 仓库根执行 powershell -ExecutionPolicy Bypass -File scripts/create-release-v10.3.1.ps1
# 依赖: git credential 已登录（或 GITHUB_TOKEN 环境变量）
$ErrorActionPreference = 'Stop'

if ($env:GITHUB_TOKEN) {
  $tok = $env:GITHUB_TOKEN
} else {
  $cred = (& git credential fill 2>$null | Out-String)
  $tok = (($cred -split "`n" | Where-Object { $_ -like 'password=*' }) -replace '^password=','').Trim()
}
if (-not $tok) { Write-Output 'ERR: no token (git credential or GITHUB_TOKEN)'; exit 1 }

$repo = 'lza6/Video-Analysis-Pro-python'
$tag  = 'v10.3.1'
$exe  = Join-Path $PSScriptRoot '..\desktop\dist\TingFeng Hermes Setup 10.3.1.exe'

$releaseBody = @'
## v10.3.1 交付闭环 — 把 v10.3.0 已建成的能力真正送到默认用户手上

### 修了什么
- **默认路径接通**: Agent 引擎默认切 react —— 监督层/写审批/分层记忆/skills roster 此前只挂在灰度路径，默认用户完全拿不到
- **审批 SSE 契约修复**: react 模式审批事件此前到不了前端（写工具静默挂 60s 后拒绝），现 /chat→/run_stream 契约打通
- **工具面统一**: 媒体生成(4) + 浏览器自动化(12) 此前只在 MCP 可用，现统一进 Web agent（16→32 工具）
- **skills 正文注入**: 此前只注入技能名字，SKILL.md 的模板/决策表从未到达模型 —— 现在 skills 真的能用
- **工具风险分类修正**: 字幕/短视频/配音（真实写盘）与 cdp_evaluate（任意 JS）此前被静默放行，现强制审批
- **沙箱防自杀**: Windows Job Object 移除 KILL_ON_JOB_CLOSE（exit 会杀掉后端自身）
- **压缩反增修复**: 上下文压缩结果此前被丢弃，摘要反而增加 token
- **工具调用解析**: 兼容 GLM 系真实输出 function=NAME 变体（此前静默丢失工具调用）
- **安装包修正**: 此前包内 0 个 skill（SKILL.md 被 filter 排除），现 12 个全部进包
- **大白话解释器 eli5 接线**: 每次工具调用产出一句话人话摘要

### 验证
- 全量测试 1336 passed + 2 skipped / 覆盖率 80%
- Playwright E2E 23 passed（真实 chromium）
- 真实后端审批链路 E2E: 允许/拒绝两路全通
- 独立 Critic 审查: CONDITIONAL PASS → 修复 BLOCKER 后 PASS
- 安装包: 697MB, resources/config/skills 实测 12 目录
'@

$body = @{
  tag_name = $tag
  name = 'v10.3.1 交付闭环（Deliver, not Build）'
  body = $releaseBody
  draft = $false
  prerelease = $false
} | ConvertTo-Json

Write-Output ("Creating release " + $tag + " on " + $repo + " ...")
$rel = Invoke-RestMethod -Method Post -Uri ("https://api.github.com/repos/$repo/releases") `
  -Headers @{ Authorization = "token $tok" } -ContentType 'application/json' -Body $body
Write-Output ("CREATED " + $rel.html_url + " id=" + $rel.id)

if (Test-Path $exe) {
  Write-Output ("Uploading asset " + $exe + " (697MB, 请耐心等几分钟)...")
  $name = [uri]::EscapeDataString((Split-Path $exe -Leaf) -replace ' ', '.')
  $uri = "https://uploads.github.com/repos/$repo/releases/" + $rel.id + "/assets?name=" + $name
  $asset = Invoke-RestMethod -Method Post -Uri $uri `
    -Headers @{ Authorization = "token $tok"; 'Content-Type' = 'application/octet-stream' } -InFile $exe
  Write-Output ("ASSET state=" + $asset.state + " url=" + $asset.browser_download_url)
} else {
  Write-Output ("WARN: asset not found at " + $exe)
}
Write-Output "DONE"
