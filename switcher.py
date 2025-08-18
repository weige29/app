# switcher.py
import time
import threading
from logger import write_log, log_event
from display_utils import change_display_resolution
from state import check_camera_event, switch_event, pause_event, camera_ok_flag, camera_thread_active
from camera_monitor import start_camera_monitor
import state as state_mod

def start_loop_switching(modes, loop_interval_var):
    """在独立线程中循环切换 modes（[(device,width,height,hz), ...]）"""
    write_log("[系统] 开始循环切换分辨率...")
    first_cycle = True
    while switch_event.is_set():
        for device, width, height, refresh_rate in modes:
            if not switch_event.is_set():
                break
            # 等待未被暂停
            pause_event.wait()
            change_display_resolution(device, width, height, refresh_rate)

            # 如果启用了摄像头检测，则容忍一段时间再触发检测
            if state_mod.camera_thread_active:
                time.sleep(5)  # POST_SWITCH_GRACE，简化直接用 5 秒（或把常量从 config 引入）
                # 触发一次检测
                state_mod.camera_ok_flag = None
                check_camera_event.set()
                while check_camera_event.is_set() and switch_event.is_set():
                    time.sleep(0.1)
                if state_mod.camera_ok_flag:
                    write_log("[系统] 切换分辨率后检测：摄像头出图正常")
                else:
                    write_log("[系统] 切换分辨率后检测：摄像头黑屏")
                    # 再次确认
                    state_mod.camera_ok_flag = None
                    check_camera_event.set()
                    while check_camera_event.is_set() and switch_event.is_set():
                        time.sleep(0.1)
                    if not first_cycle and state_mod.camera_ok_flag is False:
                        write_log("[系统] 检测到持续黑屏，暂停自动切换分辨率……")
                        pause_event.clear()
            else:
                write_log("[系统] 未启用摄像头监测，继续切换分辨率")

            first_cycle = False
            # 等待 interval 秒（响应停止）
            interval = loop_interval_var.get() if loop_interval_var else 10
            for _ in range(interval):
                if not switch_event.is_set():
                    break
                time.sleep(1)
    write_log("[系统] 分辨率循环线程已结束。")

def start_loop_combined(valid_modes, loop_interval_var, root):
    """根据用户是否启用摄像头来启动分辨率循环（会弹窗让用户选择）"""
    if not valid_modes:
        import tkinter as tk
        tk.messagebox.showwarning("提示", "当前没有可用的分辨率，无法启动循环切换。")
        return

    import tkinter as tk
    use_cam = tk.messagebox.askyesno("画面检测", "是否同时进行摄像头画面监测？")
    if use_cam:
        state_mod.camera_thread_active = True
        switch_event.set()
        # 先启动摄像头监控线程
        threading.Thread(target=lambda: start_camera_monitor(root), daemon=True).start()
        # 再启动分辨率循环线程
        threading.Thread(target=lambda: start_loop_switching(valid_modes, loop_interval_var), daemon=True).start()
    else:
        state_mod.camera_thread_active = False
        switch_event.set()
        threading.Thread(target=lambda: start_loop_switching(valid_modes, loop_interval_var), daemon=True).start()

def stop_switching():
    write_log("循环切换已停止。")
    switch_event.clear()
