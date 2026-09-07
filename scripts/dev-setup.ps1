# 开发者 onboarding 脚本（Windows PowerShell）
# TingFeng Hermes v9 — 一键配齐开发环境
# 用法：powershell -ExecutionPolicy Bypass -File scripts\dev-setup.ps1 [-WhatIf]

param([switch]$WhatIf)

Set-StrictMode -Version 3.0
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Test-Cmd { param([string]$n) [bool](Get-Command $n -ErrorAction SilentlyContinue) }

function Get-PyVer {
    foreach ($c in @('py','python')) {
        if (-not (Test-Cmd $c)) { continue }
        try {
            $o = if ($c -eq 'py') { & py -3 --version 2>$null } else { & python --version 2>$null }
            if ($o -match 'Python\s+(\d+)\.(\d+)') { return @{ cmd=$c; major=[int]$Matches[1]; minor=[int]$Matches[2]; ver=$o.Trim() } }
        } catch { continue }
    }
    return $null
}

Write-Host "=== TingFeng Hermes 开发环境配置 ===" -ForegroundColor Cyan

# 1. Python 3.10+
Write-Host "[1/8] 检测 Python 3.10+ ..." -ForegroundColor Yellow
$py = Get-PyVer
if (-not $py) { Write-Error "未检测到 Python，请装 3.10+：https://www.python.org/downloads/"; exit 1 }
if ($py.major -lt 3 -or ($py.major -eq 3 -and $py.minor -lt 10)) { Write-Error "Python $($py.ver) < 3.10，请升级"; exit 1 }
Write-Host "  OK: $($py.cmd) -> $($py.ver)" -ForegroundColor Green

# 2. Node 18+
Write-Host "[2/8] 检测 Node 18+ ..." -ForegroundColor Yellow
if (-not (Test-Cmd node)) { Write-Error "未检测到 Node，请装 18+：https://nodejs.org/"; exit 1 }
$nv = & node --version 2>$null
if ($nv -match 'v(\d+)') { $nm = [int]$Matches[1] } else { Write-Error "Node 版本解析失败: $nv"; exit 1 }
if ($nm -lt 18) { Write-Error "Node $nv < 18，请升级"; exit 1 }
Write-Host "  OK: node -> $nv" -ForegroundColor Green

# 3. FFmpeg（缺失只 warning，不退出）
Write-Host "[3/8] 检测 FFmpeg ..." -ForegroundColor Yellow
if (Test-Cmd ffmpeg) { Write-Host "  OK: ffmpeg 已安装" -ForegroundColor Green }
else { Write-Warning "FFmpeg 未安装（不阻断。可 pip 装 imageio-ffmpeg 自带）" }

if ($WhatIf) { Write-Host "`n[WhatIf] 跳过 venv/npm/cp/smoke 实际执行" -ForegroundColor Magenta; exit 0 }

# 4. 建 venv（已存在跳过）
Write-Host "[4/8] 建 venv ..." -ForegroundColor Yellow
if (Test-Path "venv\Scripts\python.exe") { Write-Host "  venv 已存在，跳过" -ForegroundColor Green }
else {
    if ($py.cmd -eq 'py') { & py -3 -m venv venv } else { & python -m venv venv }
    if (-not $?) { Write-Error "venv 创建失败"; exit 1 }
}

# 5. pip install
Write-Host "[5/8] pip install -r requirements.txt ..." -ForegroundColor Yellow
& "venv\Scripts\python.exe" -m pip install -r requirements.txt
if (-not $?) { Write-Error "pip install 失败，检查 requirements.txt / 网络"; exit 1 }

# 6. npm install (webapp + desktop)
Write-Host "[6/8] npm install (webapp + desktop) ..." -ForegroundColor Yellow
Push-Location webapp; npm install; $ok1 = $?; Pop-Location
if (-not $ok1) { Write-Error "webapp npm install 失败"; exit 1 }
Push-Location desktop; npm install; $ok2 = $?; Pop-Location
if (-not $ok2) { Write-Error "desktop npm install 失败"; exit 1 }

# 7. .env.example -> .env（已存在跳过）
Write-Host "[7/8] 配置 .env ..." -ForegroundColor Yellow
if (Test-Path ".env") { Write-Host "  .env 已存在，跳过" -ForegroundColor Green }
else {
    Copy-Item ".env.example" ".env"
    Write-Host "  已创建 .env，请填 API key（VAP_HEADLESS_TOKEN / VAP_LLM_* 等）" -ForegroundColor Yellow
}

# 8. smoke test（失败只 warning，不阻断——可能依赖未配 key）
Write-Host "[8/8] smoke test: tests/test_web_api.py ..." -ForegroundColor Yellow
$prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
& "venv\Scripts\python.exe" -m pytest tests/test_web_api.py -q --no-header
$smokeOk = $?
$ErrorActionPreference = $prev
if ($smokeOk) { Write-Host "  smoke test 通过" -ForegroundColor Green }
else { Write-Warning "smoke test 未全绿（可能依赖未配 key），不阻断 onboarding" }

Write-Host "`n✅ 开发环境就绪：双击 start-desktop.bat 启动" -ForegroundColor Green
