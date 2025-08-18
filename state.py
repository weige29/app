# state.py
import threading
from typing import Optional

# 线程/状态（供各模块共享）
check_camera_event: threading.Event = threading.Event()
switch_event: threading.Event = threading.Event()
pause_event: threading.Event = threading.Event()
pause_event.set()  # 初始未暂停

# 摄像头相关 flag
camera_ok_flag: Optional[bool] = None
camera_thread_active: bool = False

# 占位：在 gui 中创建一个 IntVar 并赋值给这里以供 switcher 使用
loop_interval_var = None
