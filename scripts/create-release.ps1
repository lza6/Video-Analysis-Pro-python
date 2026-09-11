# Create GitHub Release for v10.3.0 (+ upload Setup.exe)
# 用法（在能访问 api.github.com 的网络下，仓库根执行）:
#    powershell -ExecutionPolicy Bypass -File scripts/create-release.ps1
# 依赖：git credential 已登录（或 GITHUB_TOKEN 环境变量）
$ErrorActionPreference = 'Stop'

if ($env:GITHUB_TOKEN) {
  $tok = $env:GITHUB_TOKEN
} else {
  $cred = (& git credential fill 2>$null | Out-String)
  $tok = (($cred -split "`n" | Where-Object { $_ -like 'password=*' }) -replace '^password=','').Trim()
}
if (-not $tok) { Write-Output 'ERR: no token (git credential or GITHUB_TOKEN)'; exit 1 }

$repo = 'lza6/Video-Analysis-Pro-python'
$tag  = 'v10.3.0'
$exe  = Join-Path $PSScriptRoot '..\desktop\dist\TingFeng Hermes Setup 10.3.0.exe'

$body = @{
  tag_name = $tag
  name = 'v10.3.0 全功能闭环'
  body = @'
## v10.3.0 全功能闭环

- Agent 监督层(StuckDetector / ContextCompressor / BudgetGuard)
- 写审批 + SSE 审批弹窗 + ApprovalBus(asyncio.Event 不阻塞)
- 记忆分层(Working 热层 + FTS5 混合 + triplestore 时序图)
- skills 闭环(蒸馏→三重验证→棘轮 + SkillSpector 安全扫描 + roster 按需 + /api/skills/{ratchet,distill})
- Electron 黑匣子(实时日志 + 崩溃断路器 + 诊断导出 + autoUpdater)
- 视频工具集 / 电商·PPT·营销 skills / 浏览器·GUI 自动化
- MCP stdio 双形态 + 白名单双闸
- 反 AI 美学升级(charts + DESIGN.md)
- Windows 安装包 Setup.exe(665MB, 含 venv, 小白双击即用)

## 验证
- 覆盖率 80.25%(CI ≥80) / 1324 passed + 2 skipped
- Playwright E2E 23 passed(真实 chromium)
- pyflakes 零告警 / ruff F 零错误 / 密钥扫描干净
'@
  draft = $false
  prerelease = $false
} | ConvertTo-Json

Write-Output ("Creating release " + $tag + " on " + $repo + " ...")
$rel = Invoke-RestMethod -Method Post -Uri ("https://api.github.com/repos/$repo/releases") `
  -Headers @{ Authorization = "token $tok" } -ContentType 'application/json' -Body $body
Write-Output ("CREATED " + $rel.html_url)

if (Test-Path $exe) {
  Write-Output ("Uploading asset " + $exe + " ...")
  $asset = Invoke-RestMethod -Method Post -Uri ("https://uploads.github.com/repos/$repo/releases/" + $rel.id + "/assets?name=" + [uri]::EscapeDataString((Split-Path $exe -Leaf))) `
    -Headers @{ Authorization = "token $tok"; 'Content-Type' = 'application/octet-stream' } -InFile $exe
  Write-Output ("ASSET " + $asset.browser_download_url)
} else {
  Write-Output ("WARN: asset not found at " + $exe)
}
Write-Output "DONE"