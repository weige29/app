# gui_builder.py
import time
import os
import ctypes
import tkinter as tk
from tkinter import Listbox, Frame, Label, messagebox, Button, ttk
from display_utils import list_display_monitors, list_display_devices, get_edid_device_mapping, get_display_modes
from audio_utils import find_monitor_audio, play_audio, stop_audio
from logger import write_log, set_log_widget
from switcher import start_loop_combined, stop_switching
import state
from display_utils import change_display_resolution
from config import FONT_PATH

def build_gui():
    root = tk.Tk()
    root.title("屏幕分辨率管理")
    root.update_idletasks()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    win_w, win_h = 900, 800
    x = (screen_w - win_w) // 2
    y = (screen_h - win_h) // 2
    root.geometry(f"{win_w}x{win_h}+{x}+{y}")
    root.resizable(True, True)

    # 获取初始信息
    monitor_names = list_display_monitors()
    devices = list_display_devices()
    edid_to_device = get_edid_device_mapping()

    # 顶部显示器列表
    monitor_frame = Frame(root)
    monitor_frame.pack(pady=4, fill="x")
    monitor_label = Label(monitor_frame, text="显示器名称 (EDID):", font=("微软雅黑", 12))
    monitor_label.pack()
    monitor_listbox = Listbox(monitor_frame, width=50, height=4)
    monitor_listbox.pack()
    for name in monitor_names:
        monitor_listbox.insert(tk.END, name)

    # 音频设备下拉框
    audio_frame = Frame(root)
    audio_frame.pack(pady=2)
    audio_label = Label(audio_frame, text="外部显示器音频名称:", font=("微软雅黑", 12))
    audio_label.pack()
    device_combobox = ttk.Combobox(audio_frame, width=50)
    device_combobox.pack()

    # 启动时检测音频设备
    audio_devices = find_monitor_audio()
    if audio_devices:
        device_combobox['values'] = list(set(audio_devices))
        try:
            device_combobox.current(0)
        except Exception:
            pass
    else:
        device_combobox.set("未找到外部显示器音频设备")

    # ---- 把笔记本/台式机/多显按钮放在外部显示器音频名称下拉框下面（居中） ----
    # 定义回调函数（实现在下面可以安全调用）
    def set_notebook():
        resolution_listbox.delete(0, "end")
        widgets["valid_modes"].clear()
        monitor_names_local = list_display_monitors()
        widgets["log_listbox"].insert("end", f"检测到的显示器名称（EDID）：{monitor_names_local}")
        device_2 = r'\\.\DISPLAY2'
        monitor_name_1 = None
        for edid, dev in widgets["edid_to_device"].items():
            if dev == r'\\.\DISPLAY1':
                monitor_name_1 = edid
                break
        if not monitor_name_1:
            widgets["log_listbox"].insert("end", f"没有找到与 \\.\DISPLAY1 对应的显示器名称")
            return
        modes = get_display_modes(device_2)
        seen_res = set()
        filtered = []
        for w,h,r in modes:
            if (w,h) in [(3840,2160),(2560,1440),(3440,1440),(1920,1080),(1920,1200),(7680,4320),(7680,2160),(5120,1440),(3840,1600)] and r in [240,144,120,60,30]:
                key=(w,h,r)
                if key not in seen_res:
                    filtered.append((w,h,r)); seen_res.add(key)
        filtered.sort(key=lambda x:(x[0]*x[1], x[2]), reverse=True)
        for w,h,r in filtered:
            widgets["valid_modes"].append((device_2,w,h,r))
            widgets["resolution_listbox"].insert("end", f"{monitor_name_1}: {w}x{h} @ {r}Hz")
        if not widgets["valid_modes"]:
            widgets["log_listbox"].insert("end", "未找到符合条件的显示模式。")

    def set_desktop():
        resolution_listbox.delete(0, "end")
        widgets["valid_modes"].clear()
        devices_list = list_display_devices()
        if devices_list:
            desktop_device = devices_list[0][0]
            widgets["log_listbox"].insert("end", f"主显示器设备: {desktop_device}")
            modes = get_display_modes(desktop_device)
            seen = set()
            for w,h,r in modes:
                if r in [120,60,30] and (w,h) in [(7680,4320),(7680,2160),(5120,1440),(3840,2160),(2560,1440),(3440,1440),(1920,1080)]:
                    key=(w,h,r)
                    if key not in seen:
                        widgets["valid_modes"].append((desktop_device,w,h,r))
                        widgets["resolution_listbox"].insert("end", f"主显示器: {w}x{h} @ {r}Hz")
                        seen.add(key)
        else:
            widgets["log_listbox"].insert("end", "没有检测到显示器设备")
        if not widgets["valid_modes"]:
            widgets["log_listbox"].insert("end", "未找到符合条件的显示模式。")

    def set_multi_display():
        sel = widgets["monitor_listbox"].curselection()
        if not sel:
            widgets["log_listbox"].insert("end", "请先选择一个显示器EDID名称。")
            return
        idx = sel[0]
        edid_name = widgets["monitor_listbox"].get(idx)
        if idx == 0:
            device = r'\\.\DISPLAY2'
        elif idx == 1:
            device = r'\\.\DISPLAY3'
        elif idx == 2:
            device = r'\\.\DISPLAY1'
        else:
            widgets["log_listbox"].insert("end", "多显模式仅支持三个外接显示器。")
            return
        widgets["resolution_listbox"].delete(0, "end")
        widgets["valid_modes"].clear()
        modes = get_display_modes(device)
        if not modes:
            widgets["log_listbox"].insert("end", f"设备 {edid_name} 没有返回任何有效的显示模式，使用强制模式。")
            modes = [
                (3840,2160,60),(2560,1440,60),(1920,1080,60),(1920,1080,30),(720,576,50),(720,480,60)
            ]
        seen = set()
        filtered = []
        for w,h,r in modes:
            if r in [240,120,60,30]:
                if (w,h) == (3840,2160) or (w,h) in [(2560,1440),(3440,1440)] or (w,h) == (1920,1080) or (w,h) in [(720,576),(720,480)]:
                    key=(w,h,r)
                    if key not in seen:
                        filtered.append((device,w,h,r)); seen.add(key)
        filtered.sort(key=lambda x:(x[1]*x[2], x[3]), reverse=True)
        for mode in filtered:
            _,w,h,r = mode
            widgets["valid_modes"].append(mode)
            widgets["resolution_listbox"].insert("end", f"{edid_name}: {w}x{h} @ {r}Hz")
        if not widgets["valid_modes"]:
            widgets["log_listbox"].insert("end", "未找到符合条件的显示模式。")

    # 居中按钮区域（放在音频下方）
    btn_frame = Frame(root)
    btn_frame.pack(pady=8, fill="x")
    inner = Frame(btn_frame)
    inner.pack(anchor="center")
    Button(inner, text="笔记本", command=set_notebook, font=("微软雅黑", 12)).pack(side="left", padx=12)
    Button(inner, text="台式机", command=set_desktop, font=("微软雅黑", 12)).pack(side="left", padx=12)
    Button(inner, text="多显", command=set_multi_display, font=("微软雅黑", 12)).pack(side="left", padx=12)

    # 分辨率显示区
    resolution_frame = Frame(root)
    resolution_frame.pack(pady=10)
    resolution_label = Label(resolution_frame, text="显示器分辨率:", font=("微软雅黑", 12))
    resolution_label.pack()
    resolution_listbox = Listbox(resolution_frame, width=50, height=10)
    resolution_listbox.pack()

    # 日志区域
    log_frame = Frame(root)
    log_frame.pack(pady=10)
    log_label = Label(log_frame, text="日志:", font=("微软雅黑", 12))
    log_label.pack()
    log_listbox = Listbox(log_frame, width=100, height=10, font=("微软雅黑", 10))
    log_listbox.pack()
    set_log_widget(log_listbox)

    # 循环间隔变量（不立即重复创建控件）
    loop_interval_var = tk.IntVar(value=10)
    state.loop_interval_var = loop_interval_var

    # 保存控件到 widgets
    widgets = {
        "root": root,
        "monitor_listbox": monitor_listbox,
        "device_combobox": device_combobox,
        "resolution_listbox": resolution_listbox,
        "log_listbox": log_listbox,
        "monitor_names": monitor_names,
        "devices": devices,
        "valid_modes": [],
        "edid_to_device": edid_to_device,
        "loop_interval_var": loop_interval_var
    }

    # 操作按钮区（把循环时间放在启动按钮旁边）
    op_frame = Frame(root)
    op_frame.pack(pady=10)

    # inline interval UI (placed next to Start button)
    interval_inline = Frame(op_frame)
    interval_inline.pack(side="left", padx=(0,8))
    tk.Label(interval_inline, text="用户选择循环时间(s):", font=("微软雅黑", 12)).pack(side="left")
    tk.Spinbox(interval_inline, from_=1, to=3600, textvariable=loop_interval_var, width=5, font=("微软雅黑", 12)).pack(side="left")

    Button(op_frame, text="启动分辨率循环", command=lambda: start_loop_combined(widgets["valid_modes"], loop_interval_var, root), font=("微软雅黑", 12)).pack(side="left", padx=6)
    Button(op_frame, text="停止分辨率循环", command=stop_switching, font=("微软雅黑", 12)).pack(side="left", padx=6)
    Button(op_frame, text="刷新", command=lambda: refresh_page(widgets), font=("微软雅黑", 12)).pack(side="left", padx=6)

    # 系统操作
    def restart_pc():
        write_log("系统正在重启...")
        time.sleep(2)
        os.system("shutdown /r /f /t 0")
    def sleep_pc():
        write_log("系统正在进入睡眠状态...")
        try:
            ctypes.windll.user32.LockWorkStation()
        except Exception:
            pass
    def hibernate_pc():
        write_log("系统正在进入休眠状态...")
        try:
            os.system("shutdown /h")
        except Exception:
            pass

    Button(op_frame, text="重启计算机", command=restart_pc, font=("微软雅黑", 12)).pack(side="left", padx=6)
    Button(op_frame, text="睡眠", command=sleep_pc, font=("微软雅黑", 12)).pack(side="left", padx=6)
    Button(op_frame, text="休眠", command=hibernate_pc, font=("微软雅黑", 12)).pack(side="left", padx=6)

    # 刷新函数（需在顶部使用 widgets）
    def refresh_page(w):
        write_log("页面已刷新。")
        w["valid_modes"].clear()
        w["monitor_names"] = list_display_monitors()
        w["devices"] = list_display_devices()
        w["monitor_listbox"].delete(0, "end")
        for name in w["monitor_names"]:
            w["monitor_listbox"].insert("end", name)
        audio_devices_local = find_monitor_audio()
        if audio_devices_local:
            w["device_combobox"]['values'] = list(set(audio_devices_local))
            try:
                w["device_combobox"].current(0)
            except Exception:
                pass
        else:
            w["device_combobox"].set("未找到外部显示器音频设备")
        w["resolution_listbox"].delete(0, "end")

    # 保存一些控件到 state（如果其他模块需要）
    state.gui_widgets = widgets

    return root
