import runpy
import traceback
import sys
import os

# Ensure valid cwd
os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    print("Running src.ui.main_window via runpy...")
    runpy.run_module('src.ui.main_window', run_name='__main__')
except Exception:
    print("CRASH DETECTED!")
    tb = traceback.format_exc()
    print(tb)
    with open("crash_debug.txt", "w", encoding='utf-8') as f:
        f.write(tb)
