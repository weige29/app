# logger.py
import time
import os
from typing import Optional

# 供 GUI 绑定的 Listbox（如果存在）
log_listbox = None

LOG_FILENAME = "自动分辨率日志.txt"
LOG_FILE = "日志.log"

def set_log_widget(widget):
    """在 gui 中绑定日志 Listbox"""
    global log_listbox
    log_listbox = widget

# 确保日志文件存在（写入 BOM 以便记事本识别）
if not os.path.exists(LOG_FILENAME):
    with open(LOG_FILENAME, "w", encoding="utf-8-sig") as f:
        f.write("")

def write_log(message: str):
    """写一行到日志窗口与日志文件（兼容中文）"""
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_message = f"[{current_time}] {message}"

    # 更新 Listbox（如果绑定了）
    try:
        if log_listbox is not None:
            log_listbox.insert("end", log_message)
            log_listbox.yview_moveto(1.0)
    except Exception:
        pass

    try:
        with open(LOG_FILENAME, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")
    except Exception as e:
        print("写日志到文件失败:", e)

def log_event(event_type: str, message: str):
    """写到单独的事件日志文件（简洁）"""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{event_type}] {message}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
