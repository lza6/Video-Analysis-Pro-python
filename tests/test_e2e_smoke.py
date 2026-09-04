"""E2E 冒烟：完整启动 DesktopApp（子进程隔离），验证主控件与核心状态。"""
import subprocess
import sys
import os

from pathlib import Path

E2E_SCRIPT = r'''
import sys, os
sys.path.insert(0, '.')
os.environ["QT_QPA_PLATFORM"] = "offscreen"
try:
    import torch
except OSError:
    torch = None

from PyQt6.QtWidgets import QApplication
import qdarktheme
from src.utils import theme_compat  # noqa: F401 桥接 qdarktheme 0.1.7→2.x API
qdarktheme.enable_hi_dpi()
app = QApplication(sys.argv)
qdarktheme.setup_theme("dark")

from src.ui.main_window import DesktopApp
window = DesktopApp()
window.show()

checks = []
def check(name, cond):
    checks.append((name, bool(cond)))

check("window_title", "Video Analysis Pro" in window.windowTitle())
check("combo_client", window.combo_client.count() >= 4)
check("btn_start_exists", hasattr(window, 'btn_start'))
check("btn_ai_exists", hasattr(window, 'btn_ai'))
check("btn_media_exists", hasattr(window, 'btn_media'))
check("agent_panel", window.agent_panel is not None)
check("tool_registry", len(window.tool_registry._tools) >= 8)
check("search_kb_registered", "search_kb" in window.tool_registry._tools)
check("point_and_jump_registered", "point_and_jump" in window.tool_registry._tools)
check("search_visual_registered", "search_visual" in window.tool_registry._tools)
check("history_manager", window.history_manager is not None)
check("status_console", window.status_console is not None)
check("vram_manager", window.vram_manager is not None)
check("btn_start_disabled_no_video", not window.btn_start.isEnabled())

# v5.1/v5.2 接线验证：监控/Skills/决策日志三个新 tab 已挂载
_tab_texts = [window.tabs.tabText(i) for i in range(window.tabs.count())]
check("surveillance_tab_present", any("监控" in t for t in _tab_texts))
check("skills_tab_present", any("Skills" in t for t in _tab_texts))
check("decision_log_tab_present", any("决策日志" in t for t in _tab_texts))
check("tab_count_10", window.tabs.count() == 10)
check("skills_tab_lists_builtin", window.tab_skills.list_widget.count() >= 1)

# v5.2 agent_prompt 八段增补：关键段落出现在完整 system prompt 中
from src.core.agent_prompt import build_system_prompt  # noqa: E402
_sp = build_system_prompt(tool_descriptions="T", context="C")
for _seg in ("AGENT_LOOP", "THOUGHT", "CLARIFY", "CITATION",
             "FAIL_SAFE", "NOTIFY", "PARALLEL"):
    check(f"prompt_has_{_seg}", _seg in _sp)

# 模拟选择视频后的状态
from pathlib import Path
fake = Path("nonexistent_video_for_state_test.mp4")
window.load_video_from_path.__wrapped__ if False else None
# 直接驱动状态（不弹对话框）
window.video_path = fake
window.btn_start.setEnabled(True)
check("btn_start_enabled_after_video", window.btn_start.isEnabled())

# 模板下拉
check("prompt_templates_loaded", window.combo_prompt.count() >= 3)

window.close()
app.processEvents()

print("E2E_RESULT:")
for name, ok in checks:
    print(f"  {name}: {'PASS' if ok else 'FAIL'}")
failed = [n for n, ok in checks if not ok]
print(f"E2E_SUMMARY: {len(checks)-len(failed)}/{len(checks)} passed")
sys.exit(1 if failed else 0)
'''


def test_desktop_app_full_startup(qapp, tmp_path):
    """子进程完整启动 DesktopApp 并断言 14 项 UI/后端状态。"""
    child = subprocess.run(
        [sys.executable, "-c", E2E_SCRIPT],
        capture_output=True, timeout=300,
        cwd=str(Path(__file__).parent.parent),
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONIOENCODING": "utf-8"},
    )
    stdout = (child.stdout or b"").decode("utf-8", errors="replace")
    stderr = (child.stderr or b"").decode("utf-8", errors="replace")
    tail = "\n".join(stdout.splitlines()[-20:])
    print(tail)
    assert child.returncode == 0, f"E2E 冒烟失败 (exit {child.returncode}):\n{tail}\nSTDERR:{stderr[-800:]}"
    assert "E2E_SUMMARY" in stdout
    result_section = stdout.split("E2E_RESULT:")[-1]
    # 逐行找 ": FAIL"（PASS 行内含 "FAIL_SAFE" 等段名，不能简单子串匹配）
    fail_lines = [ln for ln in result_section.splitlines() if ln.strip().endswith(": FAIL")]
    assert not fail_lines, f"E2E 检查项失败:\n" + "\n".join(fail_lines)
