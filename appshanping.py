# import tkinter as tk
# from tkinter import Listbox, Frame, Label, messagebox, Button, ttk
# import wmi
# import win32api
# import win32con
# import time
# import threading
# import subprocess
# import os
# import ctypes
# import sounddevice as sd
# import cv2
# import numpy as np
# import time
# import pygame
# import os
# from pygrabber.dshow_graph import FilterGraph
# from PIL import Image, ImageDraw, ImageFont
# from skimage.metrics import structural_similarity as ssim
# from collections import deque
# import threading
# import queue
# from typing import Optional
# # 线程间事件与状态（模块顶层，确保静态分析器能看到）
# check_camera_event: threading.Event = threading.Event()
# camera_ok_flag: Optional[bool] = None
# last_camera_ok_flag: Optional[bool] = None
# # ================= 全局配置 =================
# FPS = 15
# RECORD_DURATION = 30
# ALERT_DURATION = 1
# BLACK_ALERT_THRESHOLD = 4
# FONT_PATH = FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
# LOG_FILE = "日志.txt"
# CAM_LOG_FILE = "camera_日志.log"
# switch_event = threading.Event()
# pause_event = threading.Event()       # 控制“是否暂停”切换
# pause_event.set()                     # 初始状态：未暂停
# # 容忍并强制跳过黑屏的阈值（秒）
# SKIP_ON_BLACK_DURATION = 30.0

# # 当检测到长时间黑屏时由摄像头线程触发，通知切换线程强制跳过当前分辨率
# FORCE_SKIP_ON_PERSISTENT_BLACK = threading.Event()

# # 放在文件顶部全局区，与其它事件常量一起
# suppress_next_black_popup = False

# # 用于指示摄像头监控线程是否已启动
# camera_thread_active = False

# # ================= 辅助函数 =================

# # 初始化线程控制事件
# switch_event = threading.Event()

# # 通过 WMI 获取显示器 EDID 名称列表
# def list_display_monitors():
#     w = wmi.WMI(namespace='wmi')
#     monitors = w.WmiMonitorID()
#     monitor_names = []
#     for monitor in monitors:
#         if monitor.UserFriendlyName:
#             name = "".join([chr(c) for c in monitor.UserFriendlyName if c > 0])
#             monitor_names.append(name)
#     return monitor_names

# # 枚举显示设备，返回列表中每项为 (DeviceName, DeviceString)
# def list_display_devices():
#     devices = []
#     i = 0
#     while True:
#         try:
#             dev = win32api.EnumDisplayDevices(None, i)
#             devices.append((dev.DeviceName, dev.DeviceString))
#             i += 1
#         except Exception:
#             break
#     return devices

# # 构建 EDID 与设备的映射字典，采用模糊匹配
# def get_edid_device_mapping():
#     monitor_names = list_display_monitors()
#     devices_list = list_display_devices()  # 每项为 (DeviceName, DeviceString)
#     mapping = {}
#     used_devices = set()
#     for i, m in enumerate(monitor_names):
#         found = False
#         for dev_name, dev_str in devices_list:
#             if dev_name in used_devices:
#                 continue
#             if m.lower() in dev_str.lower():
#                 mapping[m] = dev_name
#                 used_devices.add(dev_name)
#                 found = True
#                 break
#         if not found:
#             if i < len(devices_list):
#                 mapping[m] = devices_list[i][0]
#                 used_devices.add(devices_list[i][0])
#     return mapping

# # 获取所有显示模式
# def get_display_modes(display_device):
#     modes = []
#     i = 0
#     while True:
#         try:
#             mode = win32api.EnumDisplaySettings(display_device, i)
#             if mode.PelsWidth > 0 and mode.PelsHeight > 0 and mode.DisplayFrequency > 0:
#                 modes.append((mode.PelsWidth, mode.PelsHeight, mode.DisplayFrequency))
#             i += 1
#         except Exception:
#             break
#     return modes

# # 切换显示分辨率
# def change_display_resolution(device_name, width, height, refresh_rate):
#     dm = win32api.EnumDisplaySettings(device_name, win32con.ENUM_CURRENT_SETTINGS)
#     dm.PelsWidth = width
#     dm.PelsHeight = height
#     dm.DisplayFrequency = refresh_rate
#     dm.Fields = win32con.DM_PELSWIDTH | win32con.DM_PELSHEIGHT | win32con.DM_DISPLAYFREQUENCY
#     result = win32api.ChangeDisplaySettingsEx(device_name, dm)
#     if result == win32con.DISP_CHANGE_SUCCESSFUL:
#         write_log(f"切换分辨率成功: {width}x{height} @ {refresh_rate}Hz")
#     else:
#         write_log(f"分辨率更改失败，错误代码：{result}")
#     return result


# LOG_FILENAME = "自动分辨率日志.txt"
# # 如果不存在就创建并写入 BOM，使 Windows 记事本能正确识别 UTF-8
# if not os.path.exists(LOG_FILENAME):
#     with open(LOG_FILENAME, "w", encoding="utf-8-sig") as f:
#         f.write("")  # 仅创建文件并写入 BOM

# def write_log(message):
#     # 获取当前时间
#     current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#     log_message = f"[{current_time}] {message}"

#     # 更新 Listbox（确保 log_listbox 已创建）
#     try:
#         log_listbox.insert(tk.END, log_message)
#         # 自动滚动到末尾
#         log_listbox.yview_moveto(1.0)
#     except Exception:
#         # 如果 GUI 还没创建，忽略
#         pass

#     # 将日志写入到文件，使用 utf-8（或 utf-8-sig），保证中文正常
#     try:
#         with open(LOG_FILENAME, "a", encoding="utf-8") as log_file:
#             log_file.write(log_message + "\n")
#     except Exception as e:
#         print("写日志到文件失败:", e)

# def open_camera(idx=1):
#     cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
#     if not cap.isOpened():
#         log_event("错误", f"无法打开摄像头 {idx}")
#         return None
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
#     cap.set(cv2.CAP_PROP_FPS, FPS)
#     cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
#     cap.set(cv2.CAP_PROP_EXPOSURE, -4)
#     cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
#     cap.set(cv2.CAP_PROP_AUTO_WB, 0)
#     return cap

# # 显示器选择事件
# def on_monitor_select(event):
#     selected_index = get_selected_index()  # 获取选中的索引
#     if selected_index is not None:
#         update_resolution_listbox(selected_index)  # 更新分辨率列表框

# # 分辨率选择事件
# def on_resolution_select(event):
#     selected_index = resolution_listbox.curselection()
#     if selected_index:
#         device, width, height, refresh_rate = valid_modes[selected_index[0]]
#         resolution_str = f"{width}x{height} @ {refresh_rate}Hz"
#         confirm = messagebox.askyesno("确认更改分辨率", f"您选择了分辨率: {resolution_str}\n是否要将显示器的分辨率更改为该设置？")
#         if confirm:
#             change_display_resolution(device, width, height, refresh_rate)

# # 获取选择的显示器索引
# def get_selected_index():
#     selected_index = monitor_listbox.curselection()  # 获取选中的索引
#     return selected_index[0] if selected_index else None  # 直接返回第一个索引值

# # 更新分辨率列表框
# def update_resolution_listbox(selected_index):
#     resolution_listbox.delete(0, tk.END)  # 清空现有内容
#     if selected_index is not None:
#         selected_name = monitor_listbox.get(selected_index)  # 获取显示器名称
#         selected_device = devices[selected_index]  # 获取设备名称
#         resolutions = get_display_modes(selected_device)  # 获取显示模式

#         displayed_resolutions = set()  # 用于检查和避免重复的分辨率
#         for resolution in resolutions:
#             width, height, refresh_rate = resolution
#             resolution_str = f"{width}x{height} @ {refresh_rate}Hz"
#             if resolution_str not in displayed_resolutions:
#                 displayed_resolutions.add(resolution_str)
#                 resolution_listbox.insert(tk.END, f"{selected_name}: {resolution_str}")  # 将分辨率显示在列表框

# # 查找外部显示器音频设备
# def find_monitor_audio():
#     # 获取所有可用的音频设备
#     device_list = sd.query_devices()
#     external_devices = []
#     numeric_devices = []

#     # 定义多个外部显示器的音频设备关键字
#     monitor_keywords = ["HDMI", "Display", "Monitor", "External", "DP", "英特尔(R)", "显示器音频", "NVIDIA", "AMD", "TV"]

#     for device in device_list:
#         # 筛选条件：设备名称包含多个关键字中的任何一个，并且有输出通道
#         if device['max_output_channels'] > 0 and any(keyword in device['name'] for keyword in monitor_keywords):
#             # 排除名称中包含 "Output" 的设备
#             if "Output" in device['name']:
#                 continue

#             # 如果设备名称中包含数字，则把它放到 numeric_devices 列表中
#             if any(char.isdigit() for char in device['name']):
#                 numeric_devices.append(device['name'])
#             else:
#                 external_devices.append(device['name'])

#     # 将带有数字的设备放到列表的开头
#     external_devices = numeric_devices + external_devices

#     # 返回筛选后的外部音频设备列表
#     return external_devices

# def display_audio_device():
#     # 获取所有外部音频设备
#     devices = find_monitor_audio()

#     # 清空下拉框中的所有选项
#     device_combobox['values'] = []

#     if devices:
#         # 防止重复显示设备
#         unique_devices = list(set(devices))  # 使用 set 去重
#         # 设置下拉框的选项
#         device_combobox['values'] = unique_devices
#         if unique_devices:
#             device_combobox.current(0)  # 默认选中第一个设备
#     else:
#         device_combobox.set("未找到外部显示器音频设备")  # 设置默认提示文字


# # ================= 全局配置 =================
# LOG_FILE = "日志.log"
# VIDEO_DIR = "shiping"
# AUDIO_PATH = "shengyin/atest_1KHz_0dB_L+R.mp3"
# FONT_PATH = FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
# # 用户自定义循环间隔（秒）
# loop_interval_var = None  # 程序初始化时再绑定到 Tk 变量


# # 摄像头检测参数
# FPS = 15
# RECORD_DURATION = 30        # 缓存总时长
# TEST_DURATION = 20          # 每种分辨率下检测运行时长（秒）
# BLACK_ALERT_THRESHOLD = 4
# ALERT_DURATION = 1
# BLACK_ALERT_PCT = 0.7         # 需要更多像素为暗才算黑（从 0.5 -> 0.7）
# BLACK_ALERT_LB = 50           # 更低亮度阈值（从 60 -> 50）
# # 分辨率循环线程控制
# switch_event = threading.Event()

# # ================ 日志 & 音频 ================
# def log_event(event_type, message):
#     ts = time.strftime("%Y-%m-%d %H:%M:%S")
#     line = f"[{ts}] [{event_type}] {message}"
#     print(line)
#     with open(LOG_FILE, "a", encoding="utf-8") as f:
#         f.write(line + "\n")

# def play_audio():
#     if os.path.exists(AUDIO_PATH):
#         pygame.mixer.init(frequency=22050, size=-16, channels=2)
#         pygame.mixer.music.load(AUDIO_PATH)
#         pygame.mixer.music.play(loops=-1)
#         log_event("系统", "音频开始播放")

# def stop_audio():
#     try:
#         # 只有在 mixer 初始化并且有音乐在播放时才调用 stop
#         if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
#             pygame.mixer.music.stop()
#             log_event("系统", "音频已停止")
#     except Exception as e:
#         # 捕获任何 pygame 错误，避免线程崩溃
#         log_event("异常", f"stop_audio 错误: {e}")


# # ============= 摄像头检测模块 =================
# def get_camera_name(idx=0):
#     devs = FilterGraph().get_input_devices()
#     return devs[idx] if idx < len(devs) else f"Camera {idx}"

# def draw_text(img, text, pos, font_size=30, color=(0,0,255)):
#     pil = Image.fromarray(img)
#     draw = ImageDraw.Draw(pil)
#     try:
#         font = ImageFont.truetype(FONT_PATH, font_size)
#     except:
#         font = ImageFont.load_default()
#     draw.text(pos, text, font=font, fill=color)
#     return np.array(pil)

# def save_video(frames, tag):
#     def _save(buf):
#         os.makedirs(VIDEO_DIR, exist_ok=True)
#         ts = time.strftime("%Y%m%d_%H%M%S")
#         path = f"{VIDEO_DIR}/{tag}_{ts}.avi"
#         h, w = buf[0].shape[:2]
#         vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"XVID"), FPS, (w,h))
#         for f in buf: vw.write(f)
#         vw.release()
#         log_event(tag, f"视频已保存: {path}")
#     threading.Thread(target=_save, args=(frames.copy(),), daemon=True).start()

# def is_black(frame, pct=0.5, lb=60):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     return np.sum(gray < lb) / gray.size > pct

# def preprocess(frame):
#     g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     b = cv2.GaussianBlur(g, (31,31), 0)
#     return cv2.medianBlur(b, 5)

# def compare(f1, f2):
#     ga = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
#     gb = f2 if f2.ndim==2 else cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
#     if ga.shape != gb.shape:
#         gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]))
#     d = cv2.absdiff(ga, gb)
#     _, th = cv2.threshold(d, 30, 255, cv2.THRESH_BINARY)
#     return np.mean(th)

# def detect_artifacts(frame, bs=64, var_th=1000, edge_th=0.25):
#     g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),(5,5),0)
#     sx, sy = np.gradient(g.astype(float))
#     em = np.hypot(sx, sy)
#     h, w = g.shape; cnt = tot = 0
#     for y in range(0, h, bs):
#         for x in range(0, w, bs):
#             blk = g[y:y+bs, x:x+bs]
#             if blk.size < bs: continue
#             if np.var(blk) < var_th and np.mean(em[y:y+bs, x:x+bs]) < edge_th:
#                 cnt += 1
#             tot += 1
#     return cnt/tot if tot else 0

# def detect_jitter_orb(prev_gray, cur_gray, disp_th=1.0, pct_th=0.05):
#     """
#     用 ORB 特征匹配+RANSAC 估计平移。
#     - disp_th: 单点平均位移阈值
#     - pct_th: 匹配后有效点比例阈值
#     """
#     orb = cv2.ORB_create(500)
#     kp1, des1 = orb.detectAndCompute(prev_gray, None)
#     kp2, des2 = orb.detectAndCompute(cur_gray,  None)
#     if des1 is None or des2 is None:
#         return False

#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(des1, des2)
#     if len(matches) < 10:
#         return False

#     # 过滤最优匹配
#     matches = sorted(matches, key=lambda x: x.distance)[:50]
#     pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
#     pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

#     # 用 RANSAC 估计仿射，只取平移分量
#     M, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3)
#     if M is None:
#         return False
#     dx, dy = M[0,2], M[1,2]
#     inliers = mask.flatten().sum()
#     if inliers / len(matches) < pct_th:
#         return False
#     return np.hypot(dx, dy) > disp_th


# def show_camera_feed():
#     """
#     摄像头检测模块（改良版）：
#     - 稳健处理空帧；
#     - 新增基于灰度平均亮度突变的闪屏(flicker)检测；
#     - 在检测到闪屏/抖动/花屏/黑屏时，使用 root.after 调用 Tkinter messagebox（线程安全）。
#     """
#     global camera_ok_flag, last_camera_ok_flag

#     first_check = True

#     cap = open_camera(1)
#     if not cap:
#         write_log("错误", "无法打开摄像头1，无法进入 show_camera_feed")
#         return

#     window = get_camera_name(0)
#     cv2.namedWindow(window, cv2.WINDOW_NORMAL)
#     cv2.resizeWindow(window, 640, 480)

#     # 缓存帧用来遇到黑屏/闪屏/花屏时录制录像
#     buf = deque(maxlen=FPS * RECORD_DURATION)
#     prev = None
#     black_start = 0
#     last_alarm = 0
#     flash_cnt = 0
#     art_cnt = 0
#     frame_idx = 0
#     jitter_cnt = 0
#     last_art = 0
#     alert_start = 0
#     alert_img = None
#     # ==== 闪屏（flicker）参数（局部） ====
#     _flicker_count = 0
#     last_flicker_time = 0.0
#     FLICKER_BRIGHTNESS_TH = 10.0    # 可调，越小越灵敏
#     COMPARE_PROP_TH = 0.02
#     FLICKER_CONFIRM_FRAMES = 1
#     FLICKER_COOLDOWN = 2.0

#     # ==== 抖屏（jitter）持久化报警参数 ====
#     jitter_start = None              # 抖动开始时间（None 表示当前不在抖动中）
#     JITTER_ALERT_DURATION = 2.0      # 抖屏必须持续 >= 5 秒才提示
#     JITTER_DISP_TH = 0.5             # ORB 位移阈值，按需调小
#     JITTER_PCT_TH = 0.01             # ORB 匹配比例阈值，按需调小




#     # 闪屏检测参数（局部变量，避免作用域问题）
#     FLICKER_BRIGHTNESS_TH = 60.0   # 灰度平均值突变阈值（可调）
#     # FLICKER_COOLDOWN = 2.0         # 间隔冷却（秒）
#     # last_flicker_time = 0.0
#     COMPARE_PROP_TH = 0.10         # compare 返回值除以 255 -> 比例阈值，0.02 表示 >2%像素变化
#     FLICKER_CONFIRM_FRAMES = 4     # 需要连续多少帧满足条件才触发；1 非常灵敏，2 更稳健
#     FLICKER_COOLDOWN = 2.0
#     last_flicker_time = 0.0
#     _flicker_count = 0             # 局部计数器（若你在函数外定义，请用 local）
#     FLICKER_WINDOW_SEC = 1.0        # 在过去 1 秒窗口内统计变化
#     FLICKER_DIFF_TH = 12.0          # 相邻帧亮度差阈值（比单帧突变更敏感/更小）
#     FLICKER_MIN_CHANGES = 6        # 窗口内至少要有多少次大幅变化才考虑
#     FLICKER_SIGN_RATIO = 0.6       # 在这些大幅变化里，正负交替比例需 >= 60%
#     brightness_history = deque()   # 存 (timestamp, mean_brightness)
    
#     # ====================================================================

#     # 先播放监控音
#     play_audio()
#     log_event("系统", "摄像头检测线程已启动，绘制窗口以关闭结束监控")

#     try:
#         while True:
#             # 检查窗口是否被用户关闭
#             try:
#                 if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
#                     log_event("系统", "用户关闭了摄像头监控窗口，show_camera_feed 退出")
#                     switch_event.clear()
#                     break
#             except Exception as e:
#                 import traceback
#                 tb = traceback.format_exc()
#                 log_event("异常", tb)
#                 from tkinter import messagebox
#                 root.after(0, lambda: messagebox.showerror("监控线程出错", f"{e}\n\n详见日志"))

#             # 响应外部请求（分辨率线程请检测一次）
#             if check_camera_event.is_set():
#                 ret_chk, frame_chk = cap.read()
#                 if ret_chk and not is_black(frame_chk, pct=BLACK_ALERT_PCT, lb=BLACK_ALERT_LB):
#                     camera_ok_flag = True
#                 else:
#                     camera_ok_flag = False
#                 check_camera_event.clear()
#                 black_start = 0
#                 first_check = False

#             # 读取帧
#             ret, frame = cap.read()

#             # 空帧稳健处理
#             if not ret or frame is None or (hasattr(frame, "size") and frame.size == 0):
#                 log_event("异常", "读取到空帧，跳过本次循环（建议检查摄像头/驱动）")
#                 # 重试短暂等待，继续循环
#                 time.sleep(0.01)
#                 continue

#             # 缓存帧
#             buf.append(frame.copy())

#             # 黑屏监控（现有逻辑）
#             if is_black(frame):
#                 if black_start == 0:
#                     black_start = time.time()
#                 if time.time() - black_start > BLACK_ALERT_THRESHOLD and time.time() - last_alarm > ALERT_DURATION:
#                     # 暂停分辨率循环（线程不退出）
#                     if pause_event.is_set():
#                         pause_event.clear()
#                         write_log(f"[系统] 检测到持续黑屏 {time.time() - black_start:.1f}s，已暂停自动切换分辨率")

#                     # 如果黑屏持续到达强制跳过阈值，则触发强制跳过事件：
#                     if (time.time() - black_start) >= SKIP_ON_BLACK_DURATION:
#                         # 记录并通知切换线程：强制跳过当前分辨率，继续下一个
#                         write_log(f"[系统] 黑屏持续超过 {SKIP_ON_BLACK_DURATION:.0f}s，触发强制跳过当前分辨率并继续")
#                         FORCE_SKIP_ON_PERSISTENT_BLACK.set()
#                         # 允许切换线程继续（即使之前我们调用了 pause_event.clear()）
#                         pause_event.set()

#                         # ====== 关键：重置摄像头线程内部的黑屏计时与告警状态，
#                         # 这样在切换到下一个分辨率后会从 0 秒重新开始计时 ======
#                         # black_start = 0
#                         alert_start = 0
#                         alert_img = None
#                         last_alarm = time.time()   # 防止立即又触发告警
#                         # 一次性抑制下一次短时黑屏弹窗（由新分辨率立刻触发）
#                         globals()['suppress_next_black_popup'] = True
#                         # 若短时 cv2 窗口仍然存在，立即销毁它以避免残留
#                         try:
#                             cv2.destroyWindow("屏幕警告")
#                         except Exception:
#                             pass

#                         # write_log("[系统] 已重置摄像头黑屏计时，下一分辨率将重新开始计时（并抑制一次立刻弹窗）")

#                     last_alarm = time.time()
#                     save_video(list(buf), "black_screen")
#                     # 如果被标记为“抑制下一次弹窗”，则只做记录/保存，不显示短时屏幕警告窗口
#                     if globals().get('suppress_next_black_popup', False):
#                         globals()['suppress_next_black_popup'] = False   # 只抑制一次
#                         # write_log("[系统] 已抑制本次短时黑屏提示（由强制跳过触发）")
#                         # 不创建 alert_img，不设置 alert_start，也不 stop_audio()（如需可保留 stop_audio）
#                     else:
#                         # 正常行为：创建警告图像、停止音频并显示短时警告窗口
#                         alert_img = draw_text(
#                             np.zeros((200, 500, 3), np.uint8),
#                             f"黑屏 {time.time() - black_start:.1f}s",
#                             (50, 80)
#                         )
#                         alert_start = time.time()
#                         stop_audio()
#                     # 弹窗提示（线程安全）
#                     try:
#                         root.after(0, lambda m=messagebox: m.showwarning("黑屏报警", f"检测到持续黑屏 {time.time() - black_start:.1f}s"))
#                     except Exception as e:
#                         log_event("异常", f"黑屏弹窗失败: {e}")
#             else:
#                 if black_start:
#                     log_event("系统", f"黑屏恢复，持续 {time.time() - black_start:.1f}s")
#                 if camera_thread_active and not pause_event.is_set():
#                     pause_event.set()
#                     write_log("[系统] 摄像头恢复正常，自动切换分辨率已恢复")
#                     black_start = 0
#                     play_audio()

#             # ===== 新版闪屏检测：亮度突变 OR 帧差比例（并支持连续帧确认） =====
#             if prev is not None and hasattr(prev, "size") and prev.size > 0:
#                 try:
#                     gray_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
#                     gray_cur  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                 except Exception as e:
#                     log_event("异常", f"闪屏检测灰度转换失败: {e}")
#                 else:
#                     mean_prev = float(np.mean(gray_prev))
#                     mean_cur  = float(np.mean(gray_cur))
#                     brightness_diff = abs(mean_cur - mean_prev)

#                     # 用 compare() 计算帧差的像素比例（0..1）
#                     try:
#                         diff_val = compare(prev, frame)          # compare 返回 0..255 的平均二值化差异
#                         diff_prop = diff_val / 255.0
#                     except Exception as e:
#                         diff_prop = 0.0
#                         log_event("异常", f"compare() 失败: {e}")

#                     # 判断是否为闪屏（任一条件成立则视为闪屏帧）
#                     is_flicker_frame = (brightness_diff >= FLICKER_BRIGHTNESS_TH) or (diff_prop >= COMPARE_PROP_TH)

#                     if is_flicker_frame:
#                         _flicker_count += 1
#                     else:
#                         _flicker_count = 0

#                     now = time.time()
#                     if _flicker_count >= FLICKER_CONFIRM_FRAMES and (now - last_flicker_time) >= FLICKER_COOLDOWN and (now - last_alarm) > ALERT_DURATION:
#                         # 排除黑屏误判
#                         if not is_black(prev) and not is_black(frame):
#                             last_flicker_time = now
#                             last_alarm = now
#                             _flicker_count = 0
#                             save_video(list(buf), "flicker")
#                             log_event("闪屏", f"检测到闪屏 brightness_diff={brightness_diff:.1f} diff_prop={diff_prop:.4f}")

#                             alert_img = draw_text(
#                                 np.zeros((200, 500, 3), np.uint8),
#                                 "检测到闪屏！",
#                                 (50, 80),
#                                 color=(0, 255, 255)
#                             )
#                             alert_start = time.time()
#                             try:
#                                 root.after(0, lambda m=messagebox: m.showwarning("闪屏报警", "检测到摄像头画面闪屏，已保存录像"))
#                             except Exception as e:
#                                 log_event("异常", f"闪屏弹窗失败: {e}")

# # ====== 2) 再检测：抖屏 (jitter)，需持续 >= JITTER_ALERT_DURATION 才报警 ======
#             # 使用连续持续时长判断（jitter_start），而非短窗口计数
#             if prev is not None and hasattr(prev, "size") and prev.size > 0:
#                 try:
#                     gray_prev_j = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
#                     gray_cur_j  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                     detected = detect_jitter_orb(gray_prev_j, gray_cur_j, disp_th=JITTER_DISP_TH, pct_th=JITTER_PCT_TH)
#                 except Exception as e:
#                     detected = False
#                     log_event("异常", f"ORB 抖动检测异常: {e}")

#                 now = time.time()
#                 if detected:
#                     # 如果刚开始检测到抖动，记录开始时间
#                     if jitter_start is None:
#                         jitter_start = now
#                     # 如果抖动持续时间超过阈值并且 cooldown 允许，则报警
#                     if jitter_start is not None and (now - jitter_start) >= JITTER_ALERT_DURATION and (now - last_alarm) > ALERT_DURATION:
#                         last_alarm = now
#                         last_jitter_alert = now    # 记录抖屏报警时间（用于抑制闪屏）
#                         save_video(list(buf), "jitter")
#                         log_event("抖动", f"检测到持续抖屏，持续 {(now - jitter_start):.1f}s (阈值 {JITTER_ALERT_DURATION}s)")
#                         alert_img = draw_text(np.zeros((200, 500, 3), np.uint8),
#                                               "检测到抖动！", (50, 80), color=(255, 0, 255))
#                         alert_start = time.time()
#                         try:
#                             root.after(0, lambda m=messagebox: m.showwarning("抖动报警", "检测到持续抖屏，已保存录像"))
#                         except Exception as e:
#                             log_event("异常", f"抖动弹窗失败: {e}")
#                         # 报警后重置 jitter_start，避免重复连续报警；下次再检测到抖动会重新计时
#                         jitter_start = None
#                 else:
#                     # 若本帧未检测到抖动，则重置抖动开始时间（要求连续抖动）
#                     jitter_start = None
#             # —— 抖动（ORB）/花屏（SSIM）检测维持你原有逻辑，但加上空帧保护与弹窗 —— 

#             # 先做预处理（你原有）
#             proc = preprocess(frame)

#             # ORB 抖动（原来的两处报警合并并加弹窗）
#             if jitter_cnt >= 1 and time.time() - last_alarm > ALERT_DURATION:
#                 log_event("抖动", "ORB 检测到抖动 平移>0.5px")
#                 last_alarm = time.time()
#                 save_video(list(buf), "jitter_orb")
#                 alert_img = draw_text(
#                     np.zeros((200, 500, 3), np.uint8),
#                     "检测到抖动！",
#                     (50, 80),
#                     color=(255, 0, 255)
#                 )
#                 alert_start = time.time()
#                 try:
#                     root.after(0, lambda: messagebox.showwarning("抖动报警", "检测到摄像头画面抖动/闪烁，已保存录像"))
#                 except Exception as e:
#                     log_event("异常", f"抖动弹窗失败: {e}")
#                 jitter_cnt = 0

#             # 更新 prev（放到处理后，确保 prev 总是上一帧）
#             prev = frame.copy()

#             # 只在 frame_idx - last_art >= 10 时做花屏/更复杂检测（保留你的逻辑）
#             frame_idx += 1
#             if frame_idx - last_art >= 10:
#                 try:
#                     gray_prev2 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
#                     gray_cur2  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                     sim_val = ssim(gray_prev2, gray_cur2)
#                     ratio = detect_artifacts(frame)
#                 except Exception as e:
#                     log_event("异常", f"花屏检测异常: {e}")
#                     sim_val = 1.0
#                     ratio = 0.0

#                 if sim_val < 0.9 and ratio > 0.5:
#                     art_cnt += 1
#                 else:
#                     art_cnt = 0

#                 if art_cnt >= 3 and time.time() - last_alarm > ALERT_DURATION:
#                     log_event("花屏", f"SSIM={sim_val:.2f} ratio={ratio:.2f}")
#                     last_alarm = time.time()
#                     save_video(list(buf), "artifact")
#                     alert_img = draw_text(
#                         np.zeros((200, 500, 3), np.uint8),
#                         "检测到花屏！",
#                         (50, 80),
#                         color=(0, 255, 255)
#                     )
#                     alert_start = time.time()
#                     art_cnt = 0
#                     last_art = frame_idx
#                     try:
#                         root.after(0, lambda: messagebox.showwarning("花屏报警", "检测到画面花屏/花屏样式异常，已保存录像"))
#                     except Exception as e:
#                         log_event("异常", f"花屏弹窗失败: {e}")

#                 # 在这里插入抖动监控（使用 prev 和 frame 做 ORB 检测）
#                 if prev is not None:
#                     try:
#                         gray_prev_orb = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
#                         gray_cur_orb  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                         if detect_jitter_orb(gray_prev_orb, gray_cur_orb, disp_th=0.5, pct_th=0.01):
#                             jitter_cnt += 1
#                         else:
#                             jitter_cnt = 0
#                     except Exception as e:
#                         log_event("异常", f"ORB 抖动检测异常: {e}")
#                         jitter_cnt = 0

#             # 屏幕警告窗口显示（OpenCV 窗口，仍保留）
#             if alert_start:
#                 if time.time() - alert_start <= ALERT_DURATION:
#                     try:
#                         cv2.imshow("屏幕警告", alert_img)
#                     except Exception:
#                         pass
#                 else:
#                     try:
#                         cv2.destroyWindow("屏幕警告")
#                     except:
#                         pass
#                     alert_start = 0

#             # 显示实时画面
#             cv2.imshow(window, frame)
#             cv2.waitKey(1)

#     finally:
#         try:
#             cap.release()
#         except:
#             pass
#         cv2.destroyAllWindows()
#         stop_audio()
#         log_event("系统", "摄像头检测线程已结束")


# # 循环切换分辨率
#     write_log("[系统] 开始循环切换分辨率...")
#     while switch_event.is_set():
#         for mode in valid_modes:
#             if not switch_event.is_set():
#                 break

#             device, width, height, refresh_rate = mode
#             change_display_resolution(device, width, height, refresh_rate)

#             # 切完分辨率后，请求 show_camera_feed 立刻“检测一次黑屏/正常”
#             camera_ok_flag = None
#             check_camera_event.set()
#             # 等待 show_camera_feed 把 camera_ok_flag 设置完毕
#             while check_camera_event.is_set() and switch_event.is_set():
#                 time.sleep(0.1)

#             # 此时 show_camera_feed 已写日志，也把 camera_ok_flag 设置好
#             if camera_ok_flag is False:
#                 # 若摄像头线程发出了“强制跳过当前分辨率”的信号，则不再无限等待，直接跳到下一个分辨率
#                 if FORCE_SKIP_ON_PERSISTENT_BLACK.is_set():
#                     write_log("[系统] 强制跳过当前分辨率（长时间黑屏）并继续下一个分辨率")
#                     # 清除事件，避免影响后续分辨率
#                     FORCE_SKIP_ON_PERSISTENT_BLACK.clear()
#                     # 清理 camera_ok_flag，确保下一次分辨率检测是干净状态
#                     camera_ok_flag = None
#                     # 直接跳过当前分辨率，进入下一个 for-loop 项
#                     continue


#                 # 否则，按原逻辑等待画面恢复（每秒重新请求检测）
#                 while switch_event.is_set():
#                     # 等待 1 秒，再次请求 show_camera_feed 检测
#                     time.sleep(1)
#                     camera_ok_flag = None
#                     check_camera_event.set()
#                     while check_camera_event.is_set() and switch_event.is_set():
#                         time.sleep(0.1)
#                     if camera_ok_flag:
#                         # 恢复正常了，写日志后跳出等待
#                         write_log("系统", "摄像头恢复正常，继续切换分辨率")
#                         break

#             # 间隔 10 秒再切下一个分辨率
#             time.sleep(10)

# def stop_switching():
#     write_log("循环切换已停止。")
#     switch_event.clear()

# # ========== 启动/停止复合逻辑 ==========
# def start_loop_combined(res_list):
#     """
#     点击“启动分辨率循环”时，先让用户选择：
#       - 如果点击“是”：设置 camera_thread_active=True，启动摄像头监控线程，并启动分辨率循环线程。
#       - 如果点击“否”：设置 camera_thread_active=False，仅启动分辨率循环线程，跳过摄像头检测。
#     """
#     global camera_thread_active

#     if not res_list:
#         messagebox.showwarning("提示", "当前没有可用的分辨率，无法启动循环切换。")
#         return

#     use_cam = messagebox.askyesno("画面检测", "是否同时进行摄像头画面监测？")
#     if use_cam:
#         camera_thread_active = True
#         switch_event.set()
#         # 先启动摄像头监控线程
#         threading.Thread(target=show_camera_feed, daemon=True).start()
#         # 再启动分辨率循环线程，并传入 res_list
#         threading.Thread(target=lambda: start_loop_switching(res_list), daemon=True).start()
#     else:
#         camera_thread_active = False
#         # 仅启动分辨率循环线程
#         switch_event.set()
#         threading.Thread(target=lambda: start_loop_switching(res_list), daemon=True).start()



# def stop_loop():
#     switch_event.clear()

# # ================== 修改后的 start_loop_switching ==================
# # 在全局配置区域，加上切换后检测前的“容忍期”（秒）
# POST_SWITCH_GRACE = 5

# def start_loop_switching(modes):
#     """
#     传入一个分辨率模式列表 modes，循环切换其中的每一种：
#       modes 是 [(device1, width1, height1, hz1), ...]
#     当 camera_thread_active=True 时：
#       - 每次切换后先等待一个短暂的“容忍期”，再触发摄像头检测；
#       - 若黑屏持续超过容忍期，则暂停循环，直到恢复正常后才继续。
#     """
#     global camera_ok_flag
#     write_log("[系统] 开始循环切换分辨率...")
#     first_cycle = True

#     while switch_event.is_set():
#         for device, width, height, refresh_rate in modes:
#             if not switch_event.is_set():
#                 break

#             # —— 如果被“暂停”了，就一直等 —— 
#             pause_event.wait()

#             # —— 真正执行本次分辨率切换 —— 
#             change_display_resolution(device, width, height, refresh_rate)

#             # 切换后短暂容忍，再触发摄像头检测（仅当启用摄像头监测时）
#             if camera_thread_active:
#                 time.sleep(POST_SWITCH_GRACE)
#                 camera_ok_flag = None
#                 check_camera_event.set()
#                 # 等待摄像头线程处理完检测请求
#                 while check_camera_event.is_set() and switch_event.is_set():
#                     time.sleep(0.1)

#                 # --- 写一次检测结果日志 ---
#                 if camera_ok_flag:
#                     write_log("[系统] 切换分辨率后检测：分辨率切换成功，摄像头出图正常")
#                 else:
#                     write_log("[系统] 切换分辨率后检测：分辨率切换成功，摄像头监测黑屏")

#                     # 再次容忍并做一次确认检测（避免首轮误判）
#                     camera_ok_flag = None
#                     check_camera_event.set()
#                     while check_camera_event.is_set() and switch_event.is_set():
#                         time.sleep(0.1)

#                     if not first_cycle and camera_ok_flag is False:
#                         write_log("[系统] 检测到持续黑屏，暂停自动切换分辨率……")
#                         pause_event.clear()
#             else:
#                 # 未启用摄像头检测，直接继续切换（写个日志便于调试）
#                 write_log("[系统] 未启用摄像头监测，继续切换分辨率")


#             first_cycle = False

#             # —— 间隔等待，可随时响应停止 —— 
#             interval = loop_interval_var.get()
#             for _ in range(interval):
#                 if not switch_event.is_set():
#                     break
#                 time.sleep(1)

#     write_log("[系统] 分辨率循环线程已结束。")


# # 控制面板操作按钮
# def refresh_page():
#     write_log("页面已刷新。")
#     # 清空有效分辨率列表
#     valid_modes.clear()

#     # 重新获取显示器信息
#     monitor_names.clear()
#     monitor_names.extend(list_display_monitors())
#     devices.clear()
#     devices.extend(list_display_devices())

#     # 清空并更新显示器名称列表框
#     monitor_listbox.delete(0, tk.END)
#     for name in monitor_names:
#         monitor_listbox.insert(tk.END, name)
    
#     # 强制刷新显示器列表框
#     monitor_listbox.update_idletasks()

#     # 清空并更新音频设备下拉框
#     device_combobox['values'] = []  # 清空音频设备
#     device_combobox.set('')  # 设置为空
#     audio_devices = find_monitor_audio()  # 重新查找音频设备

#     if audio_devices:
#         device_combobox['values'] = list(set(audio_devices))  # 去重后插入设备列表
#         device_combobox.current(0)  # 默认选择第一个设备
#     else:
#         device_combobox.set("未找到外部显示器音频设备")

#     # 强制刷新音频设备下拉框
#     device_combobox.update_idletasks()

#     # 清空并更新分辨率列表框
#     resolution_listbox.delete(0, tk.END)

#     # 如果有选中的显示器，重新更新分辨率列表框
#     selected_index = get_selected_index()
#     if selected_index is not None:
#         update_resolution_listbox(selected_index)
    
#     # 强制刷新分辨率列表框
#     resolution_listbox.update_idletasks()


# def restart_pc():
#     write_log("系统正在重启...")
#     time.sleep(2)
#     os.system("shutdown /r /f /t 0")

# def sleep_pc():
#     write_log("系统正在进入睡眠状态...")
#     ctypes.windll.user32.LockWorkStation()

# def hibernate_pc():
#     write_log("系统正在进入休眠状态...")
#     os.system("shutdown /h")
    
# # 笔记本模式：选择外接显示器，使用 \\.\DISPLAY2
# def set_notebook():
#     log_listbox.insert(tk.END, "选择笔记本模式\n")
#     resolution_listbox.delete(0, tk.END)
#     valid_modes.clear()

#     monitor_names = list_display_monitors()
#     log_listbox.insert(tk.END, f"检测到的显示器名称（EDID）：{monitor_names}\n")
    
#     # 强制获取 \\.\DISPLAY1 的显示器名称（EDID）
#     device_1 = r'\\.\DISPLAY1'  # 使用 DISPLAY1 设备
    
#     # 获取与 \\.\DISPLAY1 对应的显示器名称（EDID）
#     monitor_name_1 = None
#     for edid, dev in edid_to_device.items():
#         if dev == device_1:
#             monitor_name_1 = edid  # 获取与 \\.\DISPLAY1 对应的显示器名称（EDID）
#             break
    
#     if not monitor_name_1:
#         log_listbox.insert(tk.END, f"没有找到与 {device_1} 对应的显示器名称\n")
#         return
    
#     # 获取 \\.\DISPLAY2 的显示模式
#     device_2 = r'\\.\DISPLAY2'  # 使用 DISPLAY2 设备
#     modes = get_display_modes(device_2)
    
#     if not modes:
#         log_listbox.insert(tk.END, f"设备 {device_2} 没有返回任何显示模式。\n")
#     else:
#         # 记录所有显示模式
#         # log_listbox.insert(tk.END, f"获取到的显示模式: {modes}\n")

#         # 用集合来避免重复的分辨率
#         seen_resolutions = set()  # 用于跟踪已经显示过的分辨率
#         filtered_modes = []

#         for mode in modes:
#             width, height, refresh_rate = mode
#             # 筛选 4K、2K 和 1080P 分辨率，并只保留 240Hz、144Hz、120Hz、60Hz 和 30Hz
#             if (width, height) in [(3840, 2160), (2560, 1440), (3440, 1440), (1920, 1080), (1920, 1200), (7680, 4320), (7680, 2160), (5120, 1440), (3840, 1600)]:
#                 if refresh_rate in [240, 144, 120, 60, 30]:
#                     # 检查分辨率是否已经出现过
#                     resolution_key = (width, height, refresh_rate)
#                     if resolution_key not in seen_resolutions:
#                         filtered_modes.append((width, height, refresh_rate))
#                         seen_resolutions.add(resolution_key)

#         # 按分辨率面积（宽 × 高）和刷新率从大到小排序
#         filtered_modes.sort(key=lambda x: (x[0] * x[1], x[2]), reverse=True)

#         # 显示符合条件的分辨率模式
#         for mode in filtered_modes:
#             width, height, refresh_rate = mode
#             res_str = f"{width}x{height} @ {refresh_rate}Hz"
#             valid_modes.append((device_2, width, height, refresh_rate))
#             resolution_listbox.insert(tk.END, f"{monitor_name_1}: {res_str}")
    
#     if not valid_modes:
#         log_listbox.insert(tk.END, "未找到符合条件的显示模式。\n")


# # 台式机模式：选择主显示器（第一个设备）
# def set_desktop():
#     log_listbox.insert(tk.END, "选择台式机模式\n")
#     resolution_listbox.delete(0, tk.END)
#     valid_modes.clear()

#     devices_list = list_display_devices()
#     if devices_list:
#         desktop_device = devices_list[0][0]
#         log_listbox.insert(tk.END, f"主显示器设备: {desktop_device}\n")
        
#         modes = get_display_modes(desktop_device)
        
#         # 用集合来避免重复的分辨率
#         seen_resolutions = set()  # 用于跟踪已经显示过的分辨率
        
#         for mode in modes:
#             width, height, refresh_rate = mode
#             # 筛选 120Hz、60Hz 和 30Hz 的分辨率
#             if refresh_rate in [120, 60, 30]:
#                 resolution_key = (width, height, refresh_rate)
#                 # 检查分辨率是否已经出现过
#                 if resolution_key not in seen_resolutions:
#                     # 检查是否是需要的分辨率
#                     if (width, height) in [(7680, 4320), (7680, 2160),(5120, 1440), (3840, 2160), (2560, 1440), (3440, 1440), (1920, 1080)]:
#                         valid_modes.append((desktop_device, width, height, refresh_rate))
#                         res_str = f"{width}x{height} @ {refresh_rate}Hz"
#                         resolution_listbox.insert(tk.END, f"主显示器: {res_str}")
#                         seen_resolutions.add(resolution_key)
#     else:
#         log_listbox.insert(tk.END, "没有检测到显示器设备\n")
    
#     if not valid_modes:
#         log_listbox.insert(tk.END, "未找到符合条件的显示模式。\n")


# # 多显模式：
# # 根据用户选择的显示器EDID名称，直接将第一个和第二个外接显示器分别指定为 '\\.\DISPLAY2' 和 '\\.\DISPLAY3'
# # 只输出 4K（3840×2160）、2K（2560×1440 或 3440×1440）及 1080P（1920×1080，仅保留 60Hz 和 30Hz）模式，
# # 并按分辨率面积及刷新率从大到小排序
# # 改进的 set_multi_display 函数
# def set_multi_display():
#     selected_index = monitor_listbox.curselection()
#     if not selected_index:
#         log_listbox.insert(tk.END, "请先选择一个显示器EDID名称。\n")
#         return

#     monitor_index = selected_index[0]
#     edid_name = monitor_listbox.get(monitor_index)
    
#     # 根据选择的索引固定使用对应的设备名
#     if monitor_index == 0:
#         device = r'\\.\DISPLAY2'
#     elif monitor_index == 1:
#         device = r'\\.\DISPLAY3'
#     elif monitor_index == 2:
#         device = r'\\.\DISPLAY1'
#     else:
#         log_listbox.insert(tk.END, "多显模式仅支持三个外接显示器（即 '\\.\\DISPLAY2'、'\\.\DISPLAY3' 和 '\\.\DISPLAY1'）。\n")
#         return

#     log_listbox.insert(tk.END, f"获取设备 {edid_name} 的显示模式\n")
#     resolution_listbox.delete(0, tk.END)
#     valid_modes.clear()
    
#     # 获取显示模式
#     modes = get_display_modes(device)
    
#     if not modes:
#         log_listbox.insert(tk.END, f"设备 {edid_name} 没有返回任何有效的显示模式，强制设置外部分辨率。\n")
#         modes = [
#             (3840, 2160, 60),  # 4K 60Hz
#             (2560, 1440, 60),  # 2K 60Hz
#             (1920, 1080, 60),  # 1080p 60Hz
#             (1920, 1080, 30),  # 1080p 30Hz
#             (720, 576, 50),    # 720x576 50Hz
#             (720, 480, 60),    # 720x480 60Hz
#         ]
    
#     # 用集合来避免重复的分辨率
#     seen_resolutions = set()  # 用于跟踪已经显示过的分辨率
#     filtered_modes = []

#     for mode in modes:
#         width, height, refresh_rate = mode
#         # 筛选 240Hz、120Hz、60Hz 和 30Hz 的分辨率
#         if refresh_rate in [240, 120, 60, 30]:
#             resolution_key = (width, height, refresh_rate)
#             # 检查分辨率是否已经出现过
#             if resolution_key not in seen_resolutions:
#                 # 筛选符合条件的分辨率（4K、2K、1080P、720p）
#                 if (width, height) == (3840, 2160) or (width, height) in [(2560, 1440), (3440, 1440)] or (width, height) == (1920, 1080) or (width, height) == (720, 576) or (width, height) == (720, 480):
#                     filtered_modes.append((device, width, height, refresh_rate))
#                     seen_resolutions.add(resolution_key)

#     # 按分辨率面积（宽 × 高）和刷新率从大到小排序
#     filtered_modes.sort(key=lambda x: (x[1] * x[2], x[3]), reverse=True)
    
#     if filtered_modes:
#         for mode in filtered_modes:
#             _, width, height, refresh_rate = mode
#             res_str = f"{width}x{height} @ {refresh_rate}Hz"
#             valid_modes.append(mode)
#             resolution_listbox.insert(tk.END, f"{edid_name}: {res_str}")
#     else:
#         log_listbox.insert(tk.END, f"设备 {edid_name} 没有符合条件的显示模式。\n")

# # -------------------- 窗口界面设置 --------------------
# root = tk.Tk()
# root.title("屏幕分辨率管理")
# #root.geometry("900x800")
# # ——— 新增：自动居中窗口 ———
# root.update_idletasks()
# screen_w = root.winfo_screenwidth()
# screen_h = root.winfo_screenheight()
# win_w, win_h = 900, 800
# x = (screen_w - win_w) // 2
# y = (screen_h - win_h) // 2
# root.geometry(f"{win_w}x{win_h}+{x}+{y}")
# # —————————————————————————————————
# # root.resizable(False, False)
# root.resizable(True, True)

# def alert_black(duration):
#     # 由摄像头线程调用
#     root.after(0, lambda: messagebox.showwarning("黑屏报警", f"检测到持续 {duration:.1f}s 黑屏"))
    
# # 获取EDID名称和设备信息，并建立映射
# monitor_names = list_display_monitors()
# devices = list_display_devices()  # 每项为 (DeviceName, DeviceString)
# valid_modes = []  # 保存当前显示模式列表
# edid_to_device = get_edid_device_mapping()  # {EDID_name: DeviceName}

# # 日志中输出初始映射关系
# initial_log = "初始EDID与设备映射：\n"
# for edid, dev in edid_to_device.items():
#     initial_log += f"{edid} -> {dev}\n"

# # 显示 EDID 名称列表
# monitor_frame = Frame(root)
# # monitor_frame.pack(pady=10)
# # monitor_frame.pack(pady=10, fill="both", expand=True)
# monitor_frame.pack(pady=4, fill="x")   # 把 expand=True 去掉，pady 改小
# monitor_label = Label(monitor_frame, text="显示器名称 (EDID):", font=("微软雅黑", 12))
# monitor_label.pack()
# monitor_listbox = Listbox(monitor_frame, width=50, height=4)
# monitor_listbox.pack()
# for name in monitor_names:
#     monitor_listbox.insert(tk.END, name)
# monitor_listbox.bind('<<ListboxSelect>>', on_monitor_select)

# # 音频设备下拉框
# audio_frame = Frame(root)
# audio_frame.pack(pady=2)
# audio_label = Label(audio_frame, text="外部显示器音频名称:", font=("微软雅黑", 12))
# audio_label.pack()

# device_combobox = ttk.Combobox(audio_frame, width=50)
# device_combobox.pack()

# # 启动时自动检测音频设备
# display_audio_device()

# # ================== 新增：外部显示器音频监测 ==================
# # 全局控制变量
# audio_monitor_event = threading.Event()   # 停止标志（置位表示停止）
# audio_monitor_thread = None
# audio_monitor_active = False
# # 音频静音窗口相关全局
# silence_window = None
# silence_label_var = None
# silence_current_device = None

# # 确保告警冷却变量被初始化
# _audio_last_silence_alert = 0.0

# # 参数（可按需调整）
# AUDIO_SR = 44100                   # 采样率
# AUDIO_CHUNK_SEC = 0.2              # 每次读取时长（秒）
# AUDIO_SILENCE_RMS = 1e-4           # 低于该 RMS 视为“近似静音”
# AUDIO_SILENCE_DURATION = 5.0       # 连续静音多少秒算“静音报警”
# AUDIO_STUTTER_DROP_PCT = 0.3       # 短期内低于 RMS 阈值块的比例超过该值视为卡顿
# AUDIO_STUTTER_WINDOW_SEC = 3.0     # 用于判断卡顿的滑动窗口长度（秒）
# AUDIO_ALERT_COOLDOWN = 5           # 报警冷却（秒），避免频繁弹窗
# # 新增：判断是否“有播放活动”的阈值与窗口
# PLAYBACK_ACTIVITY_RMS = 1e-3         # 若最近块中有 rms > 该值，则认为正在播放（按需微调）
# PLAYBACK_ACTIVITY_WINDOW_SEC = 3.0   # 查看最近多少秒来判定“是否有播放”
# PLAYBACK_WARMUP_SEC = 1.0            # 启动检测后给些稳态时间，避免刚启动误判
# # 去抖与计时用的全局变量（放在文件开头参数区）
# last_shown_silence = -1
# _last_sound_ts = time.time()

# # 内部状态
# _audio_rms_deque = deque()          # 存放最近的 rms 值（时间戳, rms）
# _audio_last_silence_alert = 0
# _audio_last_stutter_alert = 0

# def find_device_index_by_name(name):
#     """按设备名称查找 sounddevice 的设备索引（先精确再模糊匹配）"""
#     devs = sd.query_devices()
#     for i, d in enumerate(devs):
#         if d and 'name' in d and d['name'] == name:
#             return i, d
#     for i, d in enumerate(devs):
#         if d and 'name' in d and name.lower() in d['name'].lower():
#             return i, d
#     return None, None

# def _audio_alert_popup(title, msg):
#     """主线程弹窗+写日志（安全地从后台线程调用）"""
#     try:
#         write_log(f"[音频] {title}: {msg}")
#         root.after(0, lambda: messagebox.showwarning(title, msg))
#     except Exception as e:
#         log_event("异常", f"_audio_alert_popup 异常: {e}")

# def show_silence_window(duration_sec, device_name):
#     global silence_window, silence_label_var, silence_current_device
#     if silence_window is not None:
#         silence_current_device = device_name
#         try:
#             silence_label_var.set(f"检测到持续静音 {int(duration_sec)} 秒\n设备: {device_name}")
#         except Exception:
#             pass
#         return

#     silence_window = tk.Toplevel(root)
#     silence_window.title("外部显示器无声音")
#     w, h = 360, 120
#     try:
#         rx = root.winfo_rootx(); ry = root.winfo_rooty(); rw = root.winfo_width() or 900
#         silence_window.geometry(f"{w}x{h}+{rx + rw//3}+{ry + 100}")
#     except Exception:
#         silence_window.geometry(f"{w}x{h}")
#     silence_window.resizable(False, False)

#     def _on_close():
#         close_silence_window()
#     silence_window.protocol("WM_DELETE_WINDOW", _on_close)

#     silence_current_device = device_name
#     silence_label_var = tk.StringVar()
#     silence_label_var.set(f"检测到持续静音 {int(duration_sec)} 秒\n设备: {device_name}")

#     lbl = tk.Label(silence_window, textvariable=silence_label_var, font=("微软雅黑", 12), justify="center")
#     lbl.pack(padx=10, pady=(12,6))
#     btn = tk.Button(silence_window, text="关闭", width=10, command=_on_close)
#     btn.pack(pady=(0,10))


# def update_silence_window(duration_sec, device_name=None):
#     global silence_label_var, silence_current_device
#     if silence_window is None:
#         return
#     if device_name:
#         silence_current_device = device_name
#     try:
#         silence_label_var.set(f"检测到持续静音 {int(duration_sec)} 秒\n设备: {silence_current_device}")
#     except Exception:
#         pass


# def close_silence_window():
#     global silence_window, silence_label_var, silence_current_device
#     try:
#         if silence_window is not None:
#             silence_window.destroy()
#     except Exception:
#         pass
#     silence_window = None
#     silence_label_var = None
#     silence_current_device = None


# def audio_monitor_worker(device_name):
#     """
#     改良版 audio_monitor_worker：
#     - 兼容新/旧 sounddevice WASAPI loopback 调用方式；
#     - 优先按名称解析设备，若设备为纯输出(in=0)则尝试查找 loopback 条目；
#     - 找不到 loopback 则回退到 Stereo Mix / VB-Cable（如存在）；
#     - 使用 callback+queue 读取数据，主循环计算 RMS 并报警。
#     """
#     global last_shown_silence, _last_sound_ts
#     # global _audio_rms_deque, _audio_last_silence_alert, _audio_last_stutter_alert, audio_monitor_active

#     audio_monitor_event.clear()
#     audio_monitor_active = True

#     # 初始化线程内状态
#     playback_seen = False
#     last_activity_ts = 0.0
#     start_time = time.time()

#     chunk = int(AUDIO_SR * AUDIO_CHUNK_SEC)

#     # 安全获取设备列表 helper
#     def get_devices():
#         try:
#             return sd.query_devices()
#         except Exception:
#             return []

#     # 按名字查索引（精确->子串->关键词）
#     def find_device_index_by_name_prefer_exact(name):
#         if not name:
#             return None, None
#         target = (name or "").strip().lower()
#         devs = get_devices()
#         for i, d in enumerate(devs):
#             if (d.get('name') or '').strip().lower() == target:
#                 return i, d
#         for i, d in enumerate(devs):
#             if target in (d.get('name') or '').lower():
#                 return i, d
#         keywords = target.split()
#         for i, d in enumerate(devs):
#             n = (d.get('name') or '').lower()
#             if all(k in n for k in keywords):
#                 return i, d
#         return None, None

#     # 查找 loopback 条目（名字含 'loopback' 或设备名里含原始名字且有 input 通道）
#     def find_loopback_device_for_name(name_substr=None):
#         devs = get_devices()
#         # 首先找带 loopback 的条目
#         for i, d in enumerate(devs):
#             n = (d.get('name') or '').lower()
#             if 'loopback' in n and d.get('max_input_channels', 0) > 0:
#                 return i, d
#         # 再找名字包含目标名且有 input 通道
#         if name_substr:
#             for i, d in enumerate(devs):
#                 n = (d.get('name') or '').lower()
#                 if name_substr.lower() in n and d.get('max_input_channels', 0) > 0:
#                     return i, d
#         return None, None

#     # Stereo Mix 回退查找
#     def find_stereo_mix_device_local():
#         devs = get_devices()
#         for ii, dd in enumerate(devs):
#             name = (dd.get('name') or "").lower()
#             if dd.get('max_input_channels', 0) > 0 and 'stereo' in name and 'mix' in name:
#                 return ii, dd
#         for ii, dd in enumerate(devs):
#             name = (dd.get('name') or "").lower()
#             if dd.get('max_input_channels', 0) > 0 and ('mix' in name or '立体声' in name or 'wave out' in name or 'stereo' in name):
#                 return ii, dd
#         return None, None

#     # 测试能否用 blocking 模式打开（用于探测）
#     def try_open_blocking_by_index(idx, ch, use_wasapi, device_name_hint=None):
#         try:
#             if use_wasapi:
#                 # 尝试两种策略：
#                 # 1) 若 sd.WasapiSettings 支持 loopback 参数，直接使用
#                 try:
#                     was = sd.WasapiSettings(loopback=True)
#                     s = sd.InputStream(device=idx, channels=ch,
#                                        samplerate=AUDIO_SR, blocksize=chunk,
#                                        dtype='float32', extra_settings=was)
#                     s.start(); s.stop(); s.close()
#                     return True, None
#                 except TypeError:
#                     # 新版 sounddevice 不支持 loopback 参数：尝试查找 (loopback) 设备并直接打开该 index
#                     loop_idx, _ = find_loopback_device_for_name(device_name_hint)
#                     if loop_idx is not None:
#                         s = sd.InputStream(device=loop_idx, channels=ch,
#                                            samplerate=AUDIO_SR, blocksize=chunk,
#                                            dtype='float32')
#                         s.start(); s.stop(); s.close()
#                         return True, ("used_loopback_index", loop_idx)
#                     else:
#                         return False, "wasapi_loopback_not_supported_and_no_loopback_device"
#                 except Exception as e:
#                     return False, e
#             else:
#                 s = sd.InputStream(device=idx, channels=ch,
#                                    samplerate=AUDIO_SR, blocksize=chunk,
#                                    dtype='float32')
#                 s.start(); s.stop(); s.close()
#                 return True, None
#         except Exception as e:
#             return False, e

#     # ========== 开始解析设备 ==========
#     # 先按传入 device_name 找索引
#     try:
#         dev_idx, dev_info = find_device_index_by_name_prefer_exact(device_name)
#     except Exception:
#         dev_idx, dev_info = None, None

#     tried = []
#     stream_device_index = None
#     stream_channels = 1
#     resolved_device_name = None
#     wasapi_available = True
#     try:
#         _ = sd.WasapiSettings()  # 仅检测能否构造 WasapiSettings (不传 loopback)
#     except Exception:
#         wasapi_available = False

#     # 优先尝试用户指定的设备（如果找到）
#     if dev_idx is not None and dev_info is not None:
#         dev_max_ch = int(dev_info.get('max_output_channels', 1) or 1)
#         tried.append((dev_idx, dev_max_ch))
#         ok = False; err = None
#         # 如果该设备本身有 input 通道，可以直接尝试打开（无需 loopback）
#         if dev_info.get('max_input_channels', 0) > 0:
#             ok, err = try_open_blocking_by_index(dev_idx, min(2, dev_info.get('max_input_channels', 1)), False, device_name)
#             if ok:
#                 stream_device_index = dev_idx
#                 stream_channels = min(2, dev_info.get('max_input_channels', 1))
#                 resolved_device_name = dev_info.get('name')
#         else:
#             # 设备为输出（in==0），尝试用 WASAPI loopback（若可用）
#             if wasapi_available:
#                 ok, err = try_open_blocking_by_index(dev_idx, dev_max_ch, True, device_name)
#                 if ok:
#                     # 如果 try_open_blocking_by_index 返回用 loopback index，处理返回值
#                     if isinstance(err, tuple) and err[0] == "used_loopback_index":
#                         stream_device_index = err[1]
#                         stream_channels = dev_max_ch
#                         resolved_device_name = sd.query_devices()[stream_device_index].get('name')
#                     else:
#                         stream_device_index = dev_idx
#                         stream_channels = dev_max_ch
#                         resolved_device_name = dev_info.get('name')
#             # 若以上都不行，尝试直接以输出设备打开（可能失败，但在某些系统可行）
#             if stream_device_index is None:
#                 ok, err = try_open_blocking_by_index(dev_idx, dev_max_ch, False, device_name)
#                 if ok:
#                     stream_device_index = dev_idx
#                     stream_channels = dev_max_ch
#                     resolved_device_name = dev_info.get('name')

#     # 如果还没找到，扫描系统输出设备（优先非 HDMI）
#     if stream_device_index is None:
#         all_devs = get_devices()
#         candidates = []
#         for i, d in enumerate(all_devs):
#             if d.get('max_output_channels', 0) > 0:
#                 name = (d.get('name') or "").lower()
#                 priority = 1 if ('monitor' in name or 'hdmi' in name or 'display' in name) else 0
#                 candidates.append((priority, i, d))
#         candidates.sort(key=lambda x: x[0])
#         for priority, i, d in candidates:
#             dev_max_ch = int(d.get('max_output_channels', 1) or 1)
#             tried.append((i, dev_max_ch))
#             # 若设备本身有 input 通道，直接尝试
#             if d.get('max_input_channels', 0) > 0:
#                 ok, err = try_open_blocking_by_index(i, min(2, d.get('max_input_channels', 1)), False, d.get('name'))
#                 if ok:
#                     stream_device_index = i
#                     stream_channels = min(2, d.get('max_input_channels', 1))
#                     resolved_device_name = d.get('name')
#                     break
#             else:
#                 # 尝试以 wasapi loopback 方式（若可）
#                 if wasapi_available:
#                     ok, err = try_open_blocking_by_index(i, dev_max_ch, True, d.get('name'))
#                     if ok:
#                         if isinstance(err, tuple) and err[0] == "used_loopback_index":
#                             stream_device_index = err[1]
#                             stream_channels = dev_max_ch
#                             resolved_device_name = sd.query_devices()[stream_device_index].get('name')
#                         else:
#                             stream_device_index = i
#                             stream_channels = dev_max_ch
#                             resolved_device_name = d.get('name')
#                         break
#                 # 尝试普通打开（不常见）
#                 ok, err = try_open_blocking_by_index(i, dev_max_ch, False, d.get('name'))
#                 if ok:
#                     stream_device_index = i
#                     stream_channels = dev_max_ch
#                     resolved_device_name = d.get('name')
#                     break

#     # 回退：优先查找 VB-Cable / Stereo Mix 输入设备
#     if stream_device_index is None:
#         # 优先找 vb-cable-like (名字含 'cable')
#         devs = get_devices()
#         for ii, dd in enumerate(devs):
#             n = (dd.get('name') or '').lower()
#             if 'cable' in n and dd.get('max_input_channels', 0) > 0:
#                 stream_device_index = ii
#                 stream_channels = min(2, dd.get('max_input_channels', 1))
#                 resolved_device_name = dd.get('name')
#                 log_event("音频", f"回退到虚拟线设备: {resolved_device_name} (index={ii})")
#                 break
#         # 再找 Stereo Mix
#         if stream_device_index is None:
#             sm_idx, sm_info = find_stereo_mix_device_local()
#             if sm_idx is not None:
#                 stream_device_index = sm_idx
#                 stream_channels = min(2, max(1, int(sm_info.get('max_input_channels', 1) or 1)))
#                 resolved_device_name = sm_info.get('name')
#                 log_event("音频", f"回退到录音设备: {resolved_device_name} (index={sm_idx})")

#     if stream_device_index is None:
#         _audio_alert_popup(
#             "音频监测启动失败",
#             (f"无法为设备 '{device_name}' 打开回环录音（尝试过索引: {tried}）。\n"
#              "建议：启用 Stereo Mix、或安装 VB-Cable、或选择非 HDMI 的扬声器输出。")
#         )
#         audio_monitor_active = False
#         return

#     # 最终用于创建流的 device 参数：优先使用解析到的设备名称字符串（更稳健）
#     try:
#         dev_info_dbg = sd.query_devices()[stream_device_index]
#         device_for_stream = dev_info_dbg.get('name') or stream_device_index
#         # 写日志确认实际使用的设备
#         root.after(0, lambda m=f"[音频] 使用监测设备: index={stream_device_index}, name='{device_for_stream}', in:{dev_info_dbg.get('max_input_channels')} out:{dev_info_dbg.get('max_output_channels')}": write_log(m))
#     except Exception:
#         device_for_stream = stream_device_index
#         root.after(0, lambda: write_log(f"[音频] 使用监测设备 index={stream_device_index}"))

#     # 创建回调队列与回调
#     q = queue.Queue(maxsize=60)
#     def callback(indata, frames, time_info, status):
#         try:
#             if status:
#                 log_event("音频", f"callback status: {status}")
#             q.put(indata.copy(), block=False)
#         except queue.Full:
#             try:
#                 _ = q.get_nowait()
#                 q.put(indata.copy(), block=False)
#             except Exception:
#                 pass

#     # 创建 InputStream（如果需要 extraSettings: 先尝试构造 WasapiSettings(loopback=True)，若不支持则直接用 device_for_stream）
#     try:
#         use_extra = False
#         extra = None
#         # 判断是否需要/能使用 WasapiSettings(loopback=True)
#         # 如果 resolved_device_name 含 'loopback'，直接用 device_for_stream(index/name) 不加 extra
#         if wasapi_available:
#             try:
#                 # 试探性构造（新版本可能不接受 loopback kw）
#                 extra_try = sd.WasapiSettings(loopback=True)
#                 extra = extra_try
#                 use_extra = True
#             except TypeError:
#                 # 新版本不接受 loopback kw：不抛，使用 loopback-index如果上面找到过
#                 loop_idx, loop_info = find_loopback_device_for_name(device_name)
#                 if loop_idx is not None:
#                     device_for_stream = loop_idx
#                     use_extra = False
#                 else:
#                     use_extra = False
#             except Exception:
#                 use_extra = False

#         if use_extra and extra is not None:
#             stream_cb = sd.InputStream(device=device_for_stream,
#                                        channels=stream_channels,
#                                        samplerate=AUDIO_SR,
#                                        blocksize=chunk,
#                                        dtype='float32',
#                                        callback=callback,
#                                        extra_settings=extra)
#         else:
#             stream_cb = sd.InputStream(device=device_for_stream,
#                                        channels=stream_channels,
#                                        samplerate=AUDIO_SR,
#                                        blocksize=chunk,
#                                        dtype='float32',
#                                        callback=callback)
#         stream_cb.start()
#         log_event("音频", f"音频回调流已启动: index={stream_device_index}, channels={stream_channels}, wasapi_extra_used={use_extra}")
#     except Exception as e:
#         # 记录原因并提示回退
#         log_event("音频", f"创建回调流失败: {e}")
#         # 若未尝试过 stereo mix，尝试回退
#         sm_idx, sm_info = find_stereo_mix_device_local()
#         if sm_idx is not None and sm_idx != stream_device_index:
#             try:
#                 device_for_stream = sm_idx
#                 stream_cb = sd.InputStream(device=device_for_stream,
#                                            channels=min(2, sm_info.get('max_input_channels',1)),
#                                            samplerate=AUDIO_SR,
#                                            blocksize=chunk,
#                                            dtype='float32',
#                                            callback=callback)
#                 stream_cb.start()
#                 stream_device_index = sm_idx
#                 stream_channels = min(2, sm_info.get('max_input_channels',1))
#                 resolved_device_name = sm_info.get('name')
#                 log_event("音频", f"回退并成功使用 Stereo Mix: index={sm_idx}, name={resolved_device_name}")
#             except Exception as e2:
#                 _audio_alert_popup("音频监测启动失败", f"创建回调流失败(回退 Stereo Mix 也失败): {e2}")
#                 audio_monitor_active = False
#                 return
#         else:
#             _audio_alert_popup("音频监测启动失败", f"创建回调流失败: {e}")
#             audio_monitor_active = False
#             return

#     # ===== 启动时短暂自校准噪声基线（放在 stream_cb.start() 后，进入主循环前） =====
#     CALIB_SECONDS = 2.0
#     CALIB_SAMPLE_COUNT = max(3, int(CALIB_SECONDS / AUDIO_CHUNK_SEC))
#     baseline_samples = []
#     for _ in range(CALIB_SAMPLE_COUNT):
#         try:
#             block = q.get(timeout=1.0)
#         except queue.Empty:
#             baseline_samples.append(0.0)
#             continue
#         try:
#             arr = np.array(block, dtype=np.float32)
#             if arr.ndim > 1:
#                 sample_rms = float(np.sqrt(np.mean(np.square(arr))))
#             else:
#                 sample_rms = float(np.sqrt(np.mean(np.square(arr))))
#         except Exception:
#             sample_rms = 0.0
#         baseline_samples.append(sample_rms)

#     baseline_noise = float(np.median(baseline_samples)) if baseline_samples else 0.0
#     ABSOLUTE_MIN_ACTIVITY = 1e-6
#     ACTIVITY_FACTOR = 1.5
#     activity_threshold = max(baseline_noise * ACTIVITY_FACTOR, ABSOLUTE_MIN_ACTIVITY)
#     try:
#         root.after(0, lambda: write_log(f"[音频] 自校准: baseline_noise={baseline_noise:.6e}, activity_threshold={activity_threshold:.6e}"))
#     except Exception:
#         print(f"[音频] 自校准: baseline_noise={baseline_noise:.6e}, activity_threshold={activity_threshold:.6e}")
#     # ===== 校准结束 =====

#     # ===== 主循环：从队列取数据，计算 RMS 并执行静音/卡顿判定 =====
#     try:
#         while not audio_monitor_event.is_set():
#             try:
#                 data_block = q.get(timeout=1.0)
#                 has_data = True
#             except queue.Empty:
#                 data_block = None
#                 has_data = False

#             ts = time.time()
#             if not has_data:
#                 rms = 0.0
#             else:
#                 try:
#                     arr = np.array(data_block, dtype=np.float32)
#                     rms = float(np.sqrt(np.mean(np.square(arr))))
#                 except Exception:
#                     rms = 0.0

#             # 滑动队列记录
#             _audio_rms_deque.append((ts, rms))
#             while _audio_rms_deque and (_audio_rms_deque[0][0] < ts - AUDIO_STUTTER_WINDOW_SEC - 1.0):
#                 _audio_rms_deque.popleft()

#             # 计算短窗口内最大值
#             activity_window_start = ts - PLAYBACK_ACTIVITY_WINDOW_SEC
#             recent_vals = [r for t, r in _audio_rms_deque if t >= activity_window_start]
#             recent_max = max(recent_vals) if recent_vals else 0.0

#             # 备用静音阈值（避免过分依赖单一常量）
#             silence_threshold_dynamic = max(AUDIO_SILENCE_RMS, (globals().get('baseline_noise', 0.0)) * 0.6)
#             playback_fallback_threshold = max((globals().get('activity_threshold', activity_threshold)) * 0.8,
#                                             silence_threshold_dynamic * 3.0, 1e-7)

#             # 判定播放（若短窗口内出现显著音量）
#             if recent_max > activity_threshold or recent_max > playback_fallback_threshold or rms > playback_fallback_threshold:
#                 last_activity_ts = ts
#                 playback_seen = True
#                 _last_sound_ts = ts   # <- 这里记录“最后一次有声音”的时间戳

#             # 用时间戳计算从最后一次检测到有声音到现在的静音时长（更稳定）
#             cont_silent_seconds = max(0.0, ts - _last_sound_ts) if playback_seen else 0.0

#             # 调试日志（线程安全）
#             debug_line = (f"[音频-调试] t={time.strftime('%H:%M:%S', time.localtime(ts))} "
#                         f"rms={rms:.6f} recent_max={recent_max:.6f} "
#                         f"baseline={globals().get('baseline_noise',0.0):.6e} activity_th={activity_threshold:.6e} "
#                         f"fallback_th={playback_fallback_threshold:.6e} silence_th={silence_threshold_dynamic:.6e} "
#                         f"playback_seen={playback_seen} cont_silent={cont_silent_seconds:.2f}s")
#             try:
#                 root.after(0, lambda ln=debug_line: write_log(ln))
#             except Exception:
#                 print(debug_line)

#             # 单一静音窗口逻辑（线程安全 via root.after）
#             # 使用全局 AUDIO_SILENCE_DURATION 作为阈值（你可以调整这个常量）
#             if playback_seen and cont_silent_seconds >= AUDIO_SILENCE_DURATION:
#                 sec_int = int(cont_silent_seconds)
#                 # 仅在整数秒变化时更新 UI（去抖）
#                 if sec_int != globals().get('last_shown_silence', -1):
#                     globals()['last_shown_silence'] = sec_int
#                     root.after(0, lambda s=cont_silent_seconds, d=device_name: (
#                         show_silence_window(s, d) if silence_window is None else update_silence_window(s, d)
#                     ))
#             else:
#                 # 一旦检测到“有声音恢复”，自动关闭提示窗口
#                 globals()['last_shown_silence'] = -1
#                 if silence_window is not None:
#                     root.after(0, close_silence_window)



#             # 卡顿判定
#             # if playback_seen and (time.time() - start_time) > PLAYBACK_WARMUP_SEC:
#             #     window_start_ts = ts - AUDIO_STUTTER_WINDOW_SEC
#             #     vals = [r for t, r in _audio_rms_deque if t >= window_start_ts]
#             #     if vals:
#             #         low_count = sum(1 for v in vals if v < (AUDIO_SILENCE_RMS * 2))
#             #         drop_pct = low_count / len(vals)
#             #         if drop_pct >= AUDIO_STUTTER_DROP_PCT and (ts - (_audio_last_stutter_alert or 0)) > AUDIO_ALERT_COOLDOWN:
#             #             _audio_last_stutter_alert = ts
#             #             root.after(0, lambda: _audio_alert_popup("外部显示器声音卡顿", f"检测到短期内声音中断/抖动（{drop_pct*100:.0f}%）设备: {device_name}"))

#             # time.sleep(0.005)

#     finally:
#         try:
#             stream_cb.stop()
#             stream_cb.close()
#         except Exception:
#             pass
#         audio_monitor_active = False
#         log_event("系统", f"音频监测线程已结束: {device_name}")


# def start_audio_monitoring():
#     """GUI 调用：开始监测当前下拉框选中的外部设备"""
#     global audio_monitor_thread, audio_monitor_event, audio_monitor_active
#     if audio_monitor_active:
#         messagebox.showinfo("提示", "音频监测已在运行")
#         return
#     dev = device_combobox.get()
#     if not dev or "未找到" in dev:
#         messagebox.showwarning("提示", "请先在下拉框选择外部显示器音频设备")
#         return
#     audio_monitor_event.clear()
#     audio_monitor_thread = threading.Thread(target=audio_monitor_worker, args=(dev,), daemon=True)
#     audio_monitor_thread.start()
#     write_log(f"[系统] 音频监测已请求启动: {dev}")

# def stop_audio_monitoring():
#     """GUI 调用：停止音频监测"""
#     global audio_monitor_event, audio_monitor_thread
#     if not audio_monitor_active and (audio_monitor_thread is None or not audio_monitor_thread.is_alive()):
#         write_log("[系统] 音频监测未运行")
#         return
#     audio_monitor_event.set()
#     write_log("[系统] 请求停止音频监测（等待线程结束）")

# btn_frame_audio = Frame(audio_frame)
# btn_frame_audio.pack(pady=5)

# Button(btn_frame_audio, text="启动音频监测", command=start_audio_monitoring, font=("微软雅黑", 10)).pack(side=tk.LEFT, padx=5)
# Button(btn_frame_audio, text="停止音频监测", command=stop_audio_monitoring, font=("微软雅黑", 10)).pack(side=tk.LEFT, padx=5)


# # 按钮区域
# button_frame = Frame(root)
# button_frame.pack(pady=10)


# Button(button_frame, text="笔记本", command=set_notebook, font=("微软雅黑", 12)).pack(side=tk.LEFT, padx=10)
# Button(button_frame, text="台式机", command=set_desktop, font=("微软雅黑", 12)).pack(side=tk.LEFT, padx=10)
# Button(button_frame, text="多显", command=set_multi_display, font=("微软雅黑", 12)).pack(side=tk.LEFT, padx=10)

# # 分辨率列表区域
# resolution_frame = Frame(root)
# resolution_frame.pack(pady=10)
# resolution_label = Label(resolution_frame, text="显示器分辨率:", font=("微软雅黑", 12))
# resolution_label.pack()
# resolution_listbox = Listbox(resolution_frame, width=50, height=10)
# resolution_listbox.pack()
# resolution_listbox.bind('<<ListboxSelect>>', on_resolution_select)

# # 日志显示区域
# log_frame = Frame(root)
# log_frame.pack(pady=10)
# log_label = Label(log_frame, text="日志:", font=("微软雅黑", 12))
# log_label.pack()
# # log_listbox = Listbox(log_frame, width=100, height=10)
# log_listbox = Listbox(log_frame, width=100, height=10, font=("微软雅黑", 10))
# log_listbox.pack()
# log_listbox.insert(tk.END, initial_log)

# # 按钮区域
# button_frame = Frame(root)
# button_frame.pack(pady=10)

# # ——— 新增：循环间隔输入框 ———
# interval_frame = Frame(button_frame)
# interval_frame.pack(side=tk.LEFT, padx=5)

# tk.Label(interval_frame, text="用户选择循环时间(s):", font=("微软雅黑", 12)).pack(side=tk.LEFT)
# loop_interval_var = tk.IntVar(value=10)
# tk.Spinbox(
#     interval_frame,
#     from_=1, to=3600,
#     textvariable=loop_interval_var,
#     width=5,
#     font=("微软雅黑", 12)
# ).pack(side=tk.LEFT)

# Button(button_frame,
#        text="启动分辨率循环",
#        command=lambda: start_loop_combined(valid_modes),
#        font=("微软雅黑", 12)
# ).pack(side=tk.LEFT, padx=10)

# Button(button_frame, text="停止分辨率循环", command=stop_switching, font=("微软雅黑", 12)).pack(side=tk.LEFT, padx=10)
# Button(button_frame, text="刷新", command=refresh_page, font=("微软雅黑", 12)).pack(side=tk.LEFT, padx=10)
# Button(button_frame, text="重启计算机", command=restart_pc, font=("微软雅黑", 12)).pack(side=tk.LEFT, padx=10)
# Button(button_frame, text="睡眠", command=sleep_pc, font=("微软雅黑", 12)).pack(side=tk.LEFT, padx=10)
# Button(button_frame, text="休眠", command=hibernate_pc, font=("微软雅黑", 12)).pack(side=tk.LEFT, padx=10)

# root.mainloop()

import tkinter as tk
from tkinter import Listbox, Frame, Label, messagebox, Button, ttk
import wmi
import win32api
import win32con
import time
import threading
import subprocess
import os
import ctypes
import sounddevice as sd
import cv2
import numpy as np
import time
import pygame
import os
from pygrabber.dshow_graph import FilterGraph
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim
from collections import deque
import threading
import queue
from typing import Optional
# 线程间事件与状态（模块顶层，确保静态分析器能看到）
check_camera_event: threading.Event = threading.Event()
camera_ok_flag: Optional[bool] = None
last_camera_ok_flag: Optional[bool] = None
# ================= 全局配置 =================
FPS = 15
RECORD_DURATION = 30
ALERT_DURATION = 1
BLACK_ALERT_THRESHOLD = 4
FONT_PATH = FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
LOG_FILE = "日志.txt"
CAM_LOG_FILE = "camera_日志.log"
switch_event = threading.Event()
pause_event = threading.Event()       # 控制“是否暂停”切换
pause_event.set()                     # 初始状态：未暂停
# 容忍并强制跳过黑屏的阈值（秒）
SKIP_ON_BLACK_DURATION = 30.0

# 当检测到长时间黑屏时由摄像头线程触发，通知切换线程强制跳过当前分辨率
FORCE_SKIP_ON_PERSISTENT_BLACK = threading.Event()

# 放在文件顶部全局区，与其它事件常量一起
suppress_next_black_popup = False

# 用于指示摄像头监控线程是否已启动
camera_thread_active = False

# ================= 辅助函数 =================

# 初始化线程控制事件
switch_event = threading.Event()

# 通过 WMI 获取显示器 EDID 名称列表
def list_display_monitors():
    w = wmi.WMI(namespace='wmi')
    monitors = w.WmiMonitorID()
    monitor_names = []
    for monitor in monitors:
        if monitor.UserFriendlyName:
            name = "".join([chr(c) for c in monitor.UserFriendlyName if c > 0])
            monitor_names.append(name)
    return monitor_names

# 枚举显示设备，返回列表中每项为 (DeviceName, DeviceString)
def list_display_devices():
    devices = []
    i = 0
    while True:
        try:
            dev = win32api.EnumDisplayDevices(None, i)
            devices.append((dev.DeviceName, dev.DeviceString))
            i += 1
        except Exception:
            break
    return devices

# 构建 EDID 与设备的映射字典，采用模糊匹配
def get_edid_device_mapping():
    monitor_names = list_display_monitors()
    devices_list = list_display_devices()  # 每项为 (DeviceName, DeviceString)
    mapping = {}
    used_devices = set()
    for i, m in enumerate(monitor_names):
        found = False
        for dev_name, dev_str in devices_list:
            if dev_name in used_devices:
                continue
            if m.lower() in dev_str.lower():
                mapping[m] = dev_name
                used_devices.add(dev_name)
                found = True
                break
        if not found:
            if i < len(devices_list):
                mapping[m] = devices_list[i][0]
                used_devices.add(devices_list[i][0])
    return mapping

# 获取所有显示模式
def get_display_modes(display_device):
    modes = []
    i = 0
    while True:
        try:
            mode = win32api.EnumDisplaySettings(display_device, i)
            if mode.PelsWidth > 0 and mode.PelsHeight > 0 and mode.DisplayFrequency > 0:
                modes.append((mode.PelsWidth, mode.PelsHeight, mode.DisplayFrequency))
            i += 1
        except Exception:
            break
    return modes

# 切换显示分辨率
def change_display_resolution(device_name, width, height, refresh_rate):
    dm = win32api.EnumDisplaySettings(device_name, win32con.ENUM_CURRENT_SETTINGS)
    dm.PelsWidth = width
    dm.PelsHeight = height
    dm.DisplayFrequency = refresh_rate
    dm.Fields = win32con.DM_PELSWIDTH | win32con.DM_PELSHEIGHT | win32con.DM_DISPLAYFREQUENCY
    result = win32api.ChangeDisplaySettingsEx(device_name, dm)
    if result == win32con.DISP_CHANGE_SUCCESSFUL:
        write_log(f"切换分辨率成功: {width}x{height} @ {refresh_rate}Hz")
    else:
        write_log(f"分辨率更改失败，错误代码：{result}")
    return result


LOG_FILENAME = "自动分辨率日志.txt"
# 如果不存在就创建并写入 BOM，使 Windows 记事本能正确识别 UTF-8
if not os.path.exists(LOG_FILENAME):
    with open(LOG_FILENAME, "w", encoding="utf-8-sig") as f:
        f.write("")  # 仅创建文件并写入 BOM

def write_log(message):
    # 获取当前时间
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_message = f"[{current_time}] {message}"

    # 更新 Listbox（确保 log_listbox 已创建）
    try:
        log_listbox.insert(tk.END, log_message)
        # 自动滚动到末尾
        log_listbox.yview_moveto(1.0)
    except Exception:
        # 如果 GUI 还没创建，忽略
        pass

    # 将日志写入到文件，使用 utf-8（或 utf-8-sig），保证中文正常
    try:
        with open(LOG_FILENAME, "a", encoding="utf-8") as log_file:
            log_file.write(log_message + "\n")
    except Exception as e:
        print("写日志到文件失败:", e)

def open_camera(idx=1):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        log_event("错误", f"无法打开摄像头 {idx}")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, -4)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    return cap

# 显示器选择事件
def on_monitor_select(event):
    selected_index = get_selected_index()  # 获取选中的索引
    if selected_index is not None:
        update_resolution_listbox(selected_index)  # 更新分辨率列表框

# 分辨率选择事件
def on_resolution_select(event):
    selected_index = resolution_listbox.curselection()
    if selected_index:
        device, width, height, refresh_rate = valid_modes[selected_index[0]]
        resolution_str = f"{width}x{height} @ {refresh_rate}Hz"
        confirm = messagebox.askyesno("确认更改分辨率", f"您选择了分辨率: {resolution_str}\n是否要将显示器的分辨率更改为该设置？")
        if confirm:
            change_display_resolution(device, width, height, refresh_rate)

# 获取选择的显示器索引
def get_selected_index():
    selected_index = monitor_listbox.curselection()  # 获取选中的索引
    return selected_index[0] if selected_index else None  # 直接返回第一个索引值

# 更新分辨率列表框
def update_resolution_listbox(selected_index):
    resolution_listbox.delete(0, tk.END)  # 清空现有内容
    if selected_index is not None:
        selected_name = monitor_listbox.get(selected_index)  # 获取显示器名称
        selected_device = devices[selected_index]  # 获取设备名称
        resolutions = get_display_modes(selected_device)  # 获取显示模式

        displayed_resolutions = set()  # 用于检查和避免重复的分辨率
        for resolution in resolutions:
            width, height, refresh_rate = resolution
            resolution_str = f"{width}x{height} @ {refresh_rate}Hz"
            if resolution_str not in displayed_resolutions:
                displayed_resolutions.add(resolution_str)
                resolution_listbox.insert(tk.END, f"{selected_name}: {resolution_str}")  # 将分辨率显示在列表框

# 查找外部显示器音频设备
def find_monitor_audio():
    # 获取所有可用的音频设备
    device_list = sd.query_devices()
    external_devices = []
    numeric_devices = []

    # 定义多个外部显示器的音频设备关键字
    monitor_keywords = ["HDMI", "Display", "Monitor", "External", "DP", "英特尔(R)", "显示器音频", "NVIDIA", "AMD", "TV"]

    for device in device_list:
        # 筛选条件：设备名称包含多个关键字中的任何一个，并且有输出通道
        if device['max_output_channels'] > 0 and any(keyword in device['name'] for keyword in monitor_keywords):
            # 排除名称中包含 "Output" 的设备
            if "Output" in device['name']:
                continue

            # 如果设备名称中包含数字，则把它放到 numeric_devices 列表中
            if any(char.isdigit() for char in device['name']):
                numeric_devices.append(device['name'])
            else:
                external_devices.append(device['name'])

    # 将带有数字的设备放到列表的开头
    external_devices = numeric_devices + external_devices

    # 返回筛选后的外部音频设备列表
    return external_devices

def display_audio_device():
    # 获取所有外部音频设备
    devices = find_monitor_audio()

    # 清空下拉框中的所有选项
    device_combobox['values'] = []

    if devices:
        # 防止重复显示设备
        unique_devices = list(set(devices))  # 使用 set 去重
        # 设置下拉框的选项
        device_combobox['values'] = unique_devices
        if unique_devices:
            device_combobox.current(0)  # 默认选中第一个设备
    else:
        device_combobox.set("未找到外部显示器音频设备")  # 设置默认提示文字


# ================= 全局配置 =================
LOG_FILE = "日志.log"
VIDEO_DIR = "shiping"
AUDIO_PATH = "shengyin/atest_1KHz_0dB_L+R.mp3"
FONT_PATH = FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
# 用户自定义循环间隔（秒）
loop_interval_var = None  # 程序初始化时再绑定到 Tk 变量


# 摄像头检测参数
FPS = 15
RECORD_DURATION = 30        # 缓存总时长
TEST_DURATION = 20          # 每种分辨率下检测运行时长（秒）
BLACK_ALERT_THRESHOLD = 4
ALERT_DURATION = 1
BLACK_ALERT_PCT = 0.7         # 需要更多像素为暗才算黑（从 0.5 -> 0.7）
BLACK_ALERT_LB = 50           # 更低亮度阈值（从 60 -> 50）
# 分辨率循环线程控制
switch_event = threading.Event()

# ================ 日志 & 音频 ================
def log_event(event_type, message):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{event_type}] {message}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def play_audio():
    if os.path.exists(AUDIO_PATH):
        pygame.mixer.init(frequency=22050, size=-16, channels=2)
        pygame.mixer.music.load(AUDIO_PATH)
        pygame.mixer.music.play(loops=-1)
        log_event("系统", "音频开始播放")

def stop_audio():
    try:
        # 只有在 mixer 初始化并且有音乐在播放时才调用 stop
        if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            log_event("系统", "音频已停止")
    except Exception as e:
        # 捕获任何 pygame 错误，避免线程崩溃
        log_event("异常", f"stop_audio 错误: {e}")


# ============= 摄像头检测模块 =================
def get_camera_name(idx=0):
    devs = FilterGraph().get_input_devices()
    return devs[idx] if idx < len(devs) else f"Camera {idx}"

def draw_text(img, text, pos, font_size=30, color=(0,0,255)):
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return np.array(pil)

def save_video(frames, tag):
    def _save(buf):
        os.makedirs(VIDEO_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = f"{VIDEO_DIR}/{tag}_{ts}.avi"
        h, w = buf[0].shape[:2]
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"XVID"), FPS, (w,h))
        for f in buf: vw.write(f)
        vw.release()
        log_event(tag, f"视频已保存: {path}")
    threading.Thread(target=_save, args=(frames.copy(),), daemon=True).start()

def is_black(frame, pct=0.5, lb=60):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.sum(gray < lb) / gray.size > pct

def preprocess(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    b = cv2.GaussianBlur(g, (31,31), 0)
    return cv2.medianBlur(b, 5)

def compare(f1, f2):
    ga = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    gb = f2 if f2.ndim==2 else cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    if ga.shape != gb.shape:
        gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]))
    d = cv2.absdiff(ga, gb)
    _, th = cv2.threshold(d, 30, 255, cv2.THRESH_BINARY)
    return np.mean(th)


def _downsample(img, w=512):
    h = int(img.shape[0] * (w / img.shape[1]))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

import cv2
import numpy as np
from collections import deque

def detect_color_blocks(frame, k=8, small_w=256):
    """
    更保守的颜色聚类：使用较大的 k（避免把复杂画面合并成少数大簇）
    返回降序簇占比列表
    """
    try:
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (small_w, int(frame.shape[0]*small_w/frame.shape[1])), interpolation=cv2.INTER_LINEAR)
        Z = small.reshape((-1,3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(Z, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        cnts = np.bincount(labels.flatten(), minlength=k).astype(float)
        areas = (cnts / cnts.sum()).tolist()
        areas.sort(reverse=True)
        return areas
    except Exception:
        return []

def detect_horizontal_stripes(frame, down_h=256, diff_thresh_factor=1.0):
    """
    行均值差分检测横向条带，返回 (stripe_ratio, max_delta, mean_delta)
    """
    try:
        small = cv2.resize(frame, (int(frame.shape[1] * down_h / frame.shape[0]), down_h), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(float)
        row_mean = gray.mean(axis=1)
        diffs = np.abs(np.diff(row_mean))
        if diffs.size == 0:
            return 0.0, 0.0, 0.0
        mean = float(diffs.mean()); std = float(diffs.std()); mx = float(diffs.max())
        th = mean + diff_thresh_factor * std
        peaks = np.sum(diffs > th)
        return float(peaks) / float(len(row_mean)), mx, mean
    except Exception:
        return 0.0, 0.0, 0.0

def is_frame_glitch(prev_frame, cur_frame,
                    color_block_area_th=0.22,   # 单簇面积阈（原来可能太低）
                    color_block_count_th=4,     # top N 里需要达到的“大簇”数量
                    stripe_ratio_th=0.04,       # 横条占比阈（用于复核）
                    stripe_max_delta_th=25.0,   # 行差绝对阈
                    ssim_th=0.80):
    """
    更保守的单帧判定：
      - 使用 k=8 聚类（更细分颜色）
      - 要求 (颜色簇强 + 横条证据) 或 (SSIM 极低 + 横条中等) 才返回 True
    返回 (is_glitch, details_dict)
    """
    details = {}
    try:
        # 1) 颜色簇分析
        areas = detect_color_blocks(cur_frame, k=8, small_w=256)
        details['color_areas'] = areas
        top_big = sum(1 for a in areas[:color_block_count_th+1] if a >= color_block_area_th)
        details['top_big'] = top_big

        # color_blocks_flag 更严格：需要 top_big >= color_block_count_th 并且最大簇不超大到被误判
        color_blocks_flag = (top_big >= color_block_count_th)

        # 2) 横条检测
        stripe_ratio, stripe_max_delta, stripe_mean = detect_horizontal_stripes(cur_frame, down_h=256)
        details.update({'stripe_ratio': stripe_ratio, 'stripe_max_delta': stripe_max_delta, 'stripe_mean': stripe_mean})

        stripe_flag = (stripe_ratio >= stripe_ratio_th and stripe_max_delta >= stripe_max_delta_th)

        # 3) SSIM 低且伴随中等条带时也认为异常
        ssim_flag = False
        if prev_frame is not None:
            try:
                from skimage.metrics import structural_similarity as ssim
                g1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                g2 = cv2.cvtColor(cur_frame,  cv2.COLOR_BGR2GRAY)
                s = ssim(g1, g2)
                details['ssim'] = s
                if s < ssim_th and stripe_ratio > (stripe_ratio_th / 2):
                    ssim_flag = True
            except Exception:
                pass

        # 组合逻辑：更保守的判断
        # 触发条件（任一）：
        #  A. color_blocks_flag AND stripe_flag
        #  B. ssim_flag
        #  C. stripe_flag 且 color_blocks_flag（redundant with A）或 stripe_ratio 很高
        is_glitch = False
        if color_blocks_flag and stripe_flag:
            details['reason'] = f"color+stripe top_big={top_big}"
            is_glitch = True
        elif ssim_flag:
            details['reason'] = f"ssim_low s={details.get('ssim')}"
            is_glitch = True
        elif stripe_ratio >= (stripe_ratio_th * 1.5) and stripe_max_delta >= (stripe_max_delta_th * 1.2):
            details['reason'] = f"strong_stripes r={stripe_ratio:.3f} maxd={stripe_max_delta:.1f}"
            is_glitch = True

        return is_glitch, details
    except Exception as e:
        details['error'] = str(e)
        return False, details


import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def is_frame_artifact(prev_frame, cur_frame, ssim_th=0.90, artifact_ratio_th=0.35, block_size=64, var_th=100, edge_th=0.25):
    """
    花屏检测（SSIM + 块低方差+低边缘能量）：
    - prev_frame / cur_frame: BGR np.array（保证非 None 且尺寸相同）
    - 返回 (is_artifact:bool, ssim_val:float, artifact_ratio:float)
    """
    try:
        gp = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gc = cv2.cvtColor(cur_frame,  cv2.COLOR_BGR2GRAY)
    except Exception:
        return False, 1.0, 0.0

    # SSIM (0..1)
    try:
        s = ssim(gp, gc)
    except Exception:
        s = 1.0

    # reuse your detect_artifacts which computes low-texture-block ratio (0..1)
    try:
        ratio = detect_artifacts(cur_frame, bs=block_size, var_th=var_th, edge_th=edge_th)
    except Exception:
        # fallback: compute a quick low-variance-block ratio
        h, w = gp.shape
        cnt = tot = 0
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                blk = gp[y:y+block_size, x:x+block_size]
                if blk.size < block_size * block_size:
                    continue
                if np.var(blk) < var_th:
                    cnt += 1
                tot += 1
        ratio = (cnt / tot) if tot else 0.0

    is_art = (s < ssim_th) and (ratio > artifact_ratio_th)
    return is_art, s, ratio


def is_frame_flash(prev_frame, cur_frame, abs_bright_th=40, rel_bright_pct=0.25, diff_pct_th=0.30, bin_th=30):
    """
    闪屏/突变检测（亮度变动 + 二值化差占比）
    返回 (is_flash:bool, abs_delta:float, diff_pct:float)
    """
    try:
        g1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(cur_frame,  cv2.COLOR_BGR2GRAY)
    except Exception:
        return False, 0.0, 0.0

    m1 = float(np.mean(g1))
    m2 = float(np.mean(g2))
    abs_delta = abs(m2 - m1)
    rel_delta = abs_delta / (m1 + 1e-9)

    d = cv2.absdiff(g1, g2)
    _, th = cv2.threshold(d, bin_th, 255, cv2.THRESH_BINARY)
    diff_pct = float(np.mean(th > 0))

    is_flash = (abs_delta >= abs_bright_th) or (rel_delta >= rel_bright_pct) or (diff_pct >= diff_pct_th)
    return is_flash, abs_delta, diff_pct


def detect_artifacts(frame, bs=64, var_th=1000, edge_th=0.25):
    g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),(5,5),0)
    sx, sy = np.gradient(g.astype(float))
    em = np.hypot(sx, sy)
    h, w = g.shape; cnt = tot = 0
    for y in range(0, h, bs):
        for x in range(0, w, bs):
            blk = g[y:y+bs, x:x+bs]
            if blk.size < bs: continue
            if np.var(blk) < var_th and np.mean(em[y:y+bs, x:x+bs]) < edge_th:
                cnt += 1
            tot += 1
    return cnt/tot if tot else 0

def detect_jitter_orb(prev_gray, cur_gray, disp_th=1.0, pct_th=0.05):
    """
    用 ORB 特征匹配+RANSAC 估计平移。
    - disp_th: 单点平均位移阈值
    - pct_th: 匹配后有效点比例阈值
    """
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(cur_gray,  None)
    if des1 is None or des2 is None:
        return False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 10:
        return False

    # 过滤最优匹配
    matches = sorted(matches, key=lambda x: x.distance)[:50]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # 用 RANSAC 估计仿射，只取平移分量
    M, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3)
    if M is None:
        return False
    dx, dy = M[0,2], M[1,2]
    inliers = mask.flatten().sum()
    if inliers / len(matches) < pct_th:
        return False
    return np.hypot(dx, dy) > disp_th

from collections import deque

# 放在函数外或 show_camera_feed 开始处（只执行一次）
brightness_deque = deque(maxlen=5)   # 用最近 5 帧的中值作为基线
FLASH_PERSIST_FRAMES = 2             # 需要连续满足才报警
ABS_BRIGHT_TH = 60                   # 绝对亮度差阈值（可调）
DIFF_PCT_TH = 0.50                   # 差像素占比阈值（可调）
FLASH_COOLDOWN = 2.0                 # 报警后冷却 2 秒（可用已有 last_alarm）
# 在 show_camera_feed 开头或文件全局处初始化一次
glitch_hist = deque(maxlen=3)   # 滑动窗口长度（例如 3 帧）
GLITCH_CONFIRM_N = 2            # 在窗口内至少 N 帧为 True 才确认报警
flash_candidates_ts = deque()

def show_camera_feed():
    """
    摄像头检测模块（改良版）：
    - 稳健处理空帧；
    - 新增基于灰度平均亮度突变的闪屏(flicker)检测；
    - 在检测到闪屏/抖动/花屏/黑屏时，使用 root.after 调用 Tkinter messagebox（线程安全）。
    """
    global camera_ok_flag, last_camera_ok_flag

    first_check = True

    cap = open_camera(1)
    if not cap:
        write_log("错误", "无法打开摄像头1，无法进入 show_camera_feed")
        return

    window = get_camera_name(0)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 640, 480)

    # 缓存帧用来遇到黑屏/闪屏/花屏时录制录像
    buf = deque(maxlen=FPS * RECORD_DURATION)
    prev = None
    black_start = 0
    last_alarm = 0
    flash_cnt = 0
    art_cnt = 0
    frame_idx = 0
    jitter_cnt = 0
    last_art = 0
    alert_start = 0
    alert_img = None
    # ==== 闪屏（flicker）参数（局部） ====
    _flicker_count = 0
    last_flicker_time = 0.0
    FLICKER_BRIGHTNESS_TH = 10.0    # 可调，越小越灵敏
    COMPARE_PROP_TH = 0.02
    FLICKER_CONFIRM_FRAMES = 1
    FLICKER_COOLDOWN = 2.0

    # ==== 抖屏（jitter）持久化报警参数 ====
    jitter_start = None              # 抖动开始时间（None 表示当前不在抖动中）
    JITTER_ALERT_DURATION = 2.0      # 抖屏必须持续 >= 5 秒才提示
    JITTER_DISP_TH = 0.5             # ORB 位移阈值，按需调小
    JITTER_PCT_TH = 0.01             # ORB 匹配比例阈值，按需调小




    # 闪屏检测参数（局部变量，避免作用域问题）
    FLICKER_BRIGHTNESS_TH = 60.0   # 灰度平均值突变阈值（可调）
    # FLICKER_COOLDOWN = 2.0         # 间隔冷却（秒）
    # last_flicker_time = 0.0
    COMPARE_PROP_TH = 0.10         # compare 返回值除以 255 -> 比例阈值，0.02 表示 >2%像素变化
    FLICKER_CONFIRM_FRAMES = 4     # 需要连续多少帧满足条件才触发；1 非常灵敏，2 更稳健
    FLICKER_COOLDOWN = 2.0
    last_flicker_time = 0.0
    _flicker_count = 0             # 局部计数器（若你在函数外定义，请用 local）
    FLICKER_WINDOW_SEC = 1.0        # 在过去 1 秒窗口内统计变化
    FLICKER_DIFF_TH = 12.0          # 相邻帧亮度差阈值（比单帧突变更敏感/更小）
    FLICKER_MIN_CHANGES = 6        # 窗口内至少要有多少次大幅变化才考虑
    FLICKER_SIGN_RATIO = 0.6       # 在这些大幅变化里，正负交替比例需 >= 60%
    brightness_history = deque()   # 存 (timestamp, mean_brightness)
    
    # ====================================================================

    # 先播放监控音
    play_audio()
    log_event("系统", "摄像头检测线程已启动，绘制窗口以关闭结束监控")

    try:
        while True:
            # 检查窗口是否被用户关闭
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    log_event("系统", "用户关闭了摄像头监控窗口，show_camera_feed 退出")
                    switch_event.clear()
                    break
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                log_event("异常", tb)
                from tkinter import messagebox
                root.after(0, lambda: messagebox.showerror("监控线程出错", f"{e}\n\n详见日志"))

            # 响应外部请求（分辨率线程请检测一次）
            if check_camera_event.is_set():
                ret_chk, frame_chk = cap.read()
                if ret_chk and not is_black(frame_chk, pct=BLACK_ALERT_PCT, lb=BLACK_ALERT_LB):
                    camera_ok_flag = True
                else:
                    camera_ok_flag = False
                check_camera_event.clear()
                black_start = 0
                first_check = False

            # 读取帧
            ret, frame = cap.read()

            # 空帧稳健处理
            if not ret or frame is None or (hasattr(frame, "size") and frame.size == 0):
                log_event("异常", "读取到空帧，跳过本次循环（建议检查摄像头/驱动）")
                # 重试短暂等待，继续循环
                time.sleep(0.01)
                continue

            # 缓存帧
            buf.append(frame.copy())

            # 单帧检测（调用新的 is_frame_glitch）
            is_glitch, info = is_frame_glitch(prev, frame,
                                            color_block_area_th=0.22,
                                            color_block_count_th=3,
                                            stripe_ratio_th=0.04,
                                            stripe_max_delta_th=25.0,
                                            ssim_th=0.80)

            # 推入滑动历史（True/False）
            glitch_hist.append(1 if is_glitch else 0)

            # 只有当窗口里 >= GLITCH_CONFIRM_N 帧为 True 才真正报警
            if sum(glitch_hist) >= GLITCH_CONFIRM_N and time.time() - last_alarm > ALERT_DURATION:
                write_log(f"[系统] 检测到花屏/画面异常（确认，多帧）：{info}")
                log_event("花屏", f"{info}")
                try:
                    save_video(list(buf), "artifact_confirmed")
                except Exception:
                    pass
                alert_img = draw_text(np.zeros((200, 500, 3), np.uint8),
                                    "检测到花屏/画面异常（已确认）！", (30, 80),
                                    color=(0, 255, 255))
                alert_start = time.time()
                last_alarm = time.time()
                glitch_hist.clear()   # 报警后清空历史，避免重复



            # ---------- 替换：稳健且对短时突发友好的闪屏检测实现 ----------
            # （替换你原有的 “稳健化的闪屏检测” 部分，直至 brightness_deque.append(mean_cur) 行）

            # 参数（可调整）
            FLASH_SINGLE_ABS_TH = 40.0      # 单帧绝对亮度跳变阈（比原来小一些更易捕捉短时闪）
            FLASH_SINGLE_DIFF_PCT = 0.04    # 单帧二值化帧差占比阈（同时满足才认为有效）
            FLASH_PERSIST_WINDOW = 0.6      # 窗口秒数：在此窗口内累计若干闪屏候选视为短时闪烁
            FLASH_PERSIST_COUNT = 3         # 在窗口内至少累积多少次候选则报警（2 表示 2 次很快的突变）
            FLASH_MIN_INTERVAL = 1.5        # 两次闪屏报警之间的最短冷却时间（秒）
            # brightness_deque 用作基线样本（早已定义），我们仍用 median 作为基线

            now_ts = time.time()

            # 计算当前灰度平均亮度与基线（若基线未满则直接收集）
            gray_cur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_cur = float(np.mean(gray_cur))

            if len(brightness_deque) < brightness_deque.maxlen:
                # 尚未填满基线窗口：只收集样本，不判定（避免启动时误报）
                brightness_deque.append(mean_cur)
            else:
                baseline = float(np.median(brightness_deque))
                abs_delta = abs(mean_cur - baseline)

                # 计算帧差二值化占比（0..1）
                if prev is not None:
                    try:
                        g_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                        d = cv2.absdiff(g_prev, gray_cur)
                        _, th = cv2.threshold(d, 30, 255, cv2.THRESH_BINARY)
                        diff_pct = float(np.mean(th > 0))
                    except Exception:
                        diff_pct = 0.0
                else:
                    diff_pct = 0.0

                # 判断本帧是否为“闪屏候选”（需要亮度跳变 且 帧差占比）
                is_flash_candidate = (abs_delta >= FLASH_SINGLE_ABS_TH and diff_pct >= FLASH_SINGLE_DIFF_PCT)

                # 记录短窗口内的闪屏候选时间戳（使用全局/闭包变量 flash_candidates_ts 列表）
                # 请在函数外（或 show_camera_feed 顶部）定义： flash_candidates_ts = deque()
                try:
                    flash_candidates_ts
                except NameError:
                    flash_candidates_ts = deque()

                if is_flash_candidate:
                    flash_candidates_ts.append(now_ts)
                    # 记录 debug（可注释）
                    root.after(0, lambda s=abs_delta, p=diff_pct: write_log(f"[闪屏-调试] 候选 abs={s:.1f} diff_pct={p:.3f}"))
                # 清理过期项（保留 FLASH_PERSIST_WINDOW 秒内的时间戳）
                while flash_candidates_ts and (now_ts - flash_candidates_ts[0]) > FLASH_PERSIST_WINDOW:
                    flash_candidates_ts.popleft()

                # 单帧极大跳变也可直接报警（避免 diff_pct 受阈值影响漏判）
                SINGLE_HARD_ABS = FLASH_SINGLE_ABS_TH * 2.0
                single_hard = (abs_delta >= SINGLE_HARD_ABS)

                # 决策：如果单帧极强跳变 或 在短窗口内达到累计次数，则触发闪屏报警
                can_alert = False
                if single_hard and (now_ts - last_flicker_time) > FLASH_MIN_INTERVAL:
                    can_alert = True
                    reason = f"single_hard abs={abs_delta:.1f} diff_pct={diff_pct:.3f}"
                elif len(flash_candidates_ts) >= FLASH_PERSIST_COUNT and (now_ts - last_flicker_time) > FLASH_MIN_INTERVAL:
                    can_alert = True
                    reason = f"multi_candidate count={len(flash_candidates_ts)} abs={abs_delta:.1f} diff_pct={diff_pct:.3f}"

                if can_alert:
                    # 防止误把黑屏当闪屏：检查 prev/frame 是否均非黑屏
                    if not is_black(prev) and not is_black(frame):
                        last_flicker_time = now_ts
                        last_alarm = now_ts
                        write_log(f"[系统] 检测到闪屏（确认） {reason}")
                        try:
                            save_video(list(buf), "flash_confirmed")
                        except Exception:
                            pass
                        alert_img = draw_text(np.zeros((200,500,3), np.uint8),
                                            "检测到闪屏（确认）！", (30,80), color=(0,255,255))
                        alert_start = now_ts
                        # 弹窗（线程安全）
                        try:
                            root.after(0, lambda: messagebox.showwarning("闪屏报警", f"检测到摄像头画面闪屏，{reason}，已保存录像"))
                        except Exception as e:
                            log_event("异常", f"闪屏弹窗失败: {e}")
                    flash_candidates_ts.clear()   # 报警后清空候选

                # 最后将当前亮度加入基线（延后加入，避免本帧被基线吞掉）
                brightness_deque.append(mean_cur)

            # ---------- 替换结束 ----------



            # 黑屏监控（现有逻辑）
            if is_black(frame):
                if black_start == 0:
                    black_start = time.time()
                if time.time() - black_start > BLACK_ALERT_THRESHOLD and time.time() - last_alarm > ALERT_DURATION:
                    # 暂停分辨率循环（线程不退出）
                    if pause_event.is_set():
                        pause_event.clear()
                        write_log(f"[系统] 检测到持续黑屏 {time.time() - black_start:.1f}s，已暂停自动切换分辨率")

                    # 如果黑屏持续到达强制跳过阈值，则触发强制跳过事件：
                    if (time.time() - black_start) >= SKIP_ON_BLACK_DURATION:
                        # 记录并通知切换线程：强制跳过当前分辨率，继续下一个
                        write_log(f"[系统] 黑屏持续超过 {SKIP_ON_BLACK_DURATION:.0f}s，触发强制跳过当前分辨率并继续")
                        FORCE_SKIP_ON_PERSISTENT_BLACK.set()
                        # 允许切换线程继续（即使之前我们调用了 pause_event.clear()）
                        pause_event.set()

                        # ====== 关键：重置摄像头线程内部的黑屏计时与告警状态，
                        # 这样在切换到下一个分辨率后会从 0 秒重新开始计时 ======
                        # black_start = 0
                        alert_start = 0
                        alert_img = None
                        last_alarm = time.time()   # 防止立即又触发告警
                        # 一次性抑制下一次短时黑屏弹窗（由新分辨率立刻触发）
                        globals()['suppress_next_black_popup'] = True
                        # 若短时 cv2 窗口仍然存在，立即销毁它以避免残留
                        try:
                            cv2.destroyWindow("屏幕警告")
                        except Exception:
                            pass

                        # write_log("[系统] 已重置摄像头黑屏计时，下一分辨率将重新开始计时（并抑制一次立刻弹窗）")

                    last_alarm = time.time()
                    save_video(list(buf), "black_screen")
                    # 如果被标记为“抑制下一次弹窗”，则只做记录/保存，不显示短时屏幕警告窗口
                    if globals().get('suppress_next_black_popup', False):
                        globals()['suppress_next_black_popup'] = False   # 只抑制一次
                        # write_log("[系统] 已抑制本次短时黑屏提示（由强制跳过触发）")
                        # 不创建 alert_img，不设置 alert_start，也不 stop_audio()（如需可保留 stop_audio）
                    else:
                        # 正常行为：创建警告图像、停止音频并显示短时警告窗口
                        alert_img = draw_text(
                            np.zeros((200, 500, 3), np.uint8),
                            f"黑屏 {time.time() - black_start:.1f}s",
                            (50, 80)
                        )
                        alert_start = time.time()
                        stop_audio()
                    # 弹窗提示（线程安全）
                    try:
                        root.after(0, lambda m=messagebox: m.showwarning("黑屏报警", f"检测到持续黑屏 {time.time() - black_start:.1f}s"))
                    except Exception as e:
                        log_event("异常", f"黑屏弹窗失败: {e}")
            else:
                if black_start:
                    log_event("系统", f"黑屏恢复，持续 {time.time() - black_start:.1f}s")
                if camera_thread_active and not pause_event.is_set():
                    pause_event.set()
                    write_log("[系统] 摄像头恢复正常，自动切换分辨率已恢复")
                    black_start = 0
                    play_audio()

            # ===== 新版闪屏检测：亮度突变 OR 帧差比例（并支持连续帧确认） =====
            if prev is not None and hasattr(prev, "size") and prev.size > 0:
                try:
                    gray_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                    gray_cur  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                except Exception as e:
                    log_event("异常", f"闪屏检测灰度转换失败: {e}")
                else:
                    mean_prev = float(np.mean(gray_prev))
                    mean_cur  = float(np.mean(gray_cur))
                    brightness_diff = abs(mean_cur - mean_prev)

                    # 用 compare() 计算帧差的像素比例（0..1）
                    try:
                        diff_val = compare(prev, frame)          # compare 返回 0..255 的平均二值化差异
                        diff_prop = diff_val / 255.0
                    except Exception as e:
                        diff_prop = 0.0
                        log_event("异常", f"compare() 失败: {e}")

                    # 判断是否为闪屏（任一条件成立则视为闪屏帧）
                    is_flicker_frame = (brightness_diff >= FLICKER_BRIGHTNESS_TH) or (diff_prop >= COMPARE_PROP_TH)

                    if is_flicker_frame:
                        _flicker_count += 1
                    else:
                        _flicker_count = 0

                    now = time.time()
                    if _flicker_count >= FLICKER_CONFIRM_FRAMES and (now - last_flicker_time) >= FLICKER_COOLDOWN and (now - last_alarm) > ALERT_DURATION:
                        # 排除黑屏误判
                        if not is_black(prev) and not is_black(frame):
                            last_flicker_time = now
                            last_alarm = now
                            _flicker_count = 0
                            save_video(list(buf), "flicker")
                            log_event("闪屏", f"检测到闪屏 brightness_diff={brightness_diff:.1f} diff_prop={diff_prop:.4f}")

                            alert_img = draw_text(
                                np.zeros((200, 500, 3), np.uint8),
                                "检测到闪屏！",
                                (50, 80),
                                color=(0, 255, 255)
                            )
                            alert_start = time.time()
                            try:
                                root.after(0, lambda m=messagebox: m.showwarning("闪屏报警", "检测到摄像头画面闪屏，已保存录像"))
                            except Exception as e:
                                log_event("异常", f"闪屏弹窗失败: {e}")

# ====== 2) 再检测：抖屏 (jitter)，需持续 >= JITTER_ALERT_DURATION 才报警 ======
            # 使用连续持续时长判断（jitter_start），而非短窗口计数
            if prev is not None and hasattr(prev, "size") and prev.size > 0:
                try:
                    gray_prev_j = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                    gray_cur_j  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    detected = detect_jitter_orb(gray_prev_j, gray_cur_j, disp_th=JITTER_DISP_TH, pct_th=JITTER_PCT_TH)
                except Exception as e:
                    detected = False
                    log_event("异常", f"ORB 抖动检测异常: {e}")

                now = time.time()
                if detected:
                    # 如果刚开始检测到抖动，记录开始时间
                    if jitter_start is None:
                        jitter_start = now
                    # 如果抖动持续时间超过阈值并且 cooldown 允许，则报警
                    if jitter_start is not None and (now - jitter_start) >= JITTER_ALERT_DURATION and (now - last_alarm) > ALERT_DURATION:
                        last_alarm = now
                        last_jitter_alert = now    # 记录抖屏报警时间（用于抑制闪屏）
                        save_video(list(buf), "jitter")
                        log_event("抖动", f"检测到持续抖屏，持续 {(now - jitter_start):.1f}s (阈值 {JITTER_ALERT_DURATION}s)")
                        alert_img = draw_text(np.zeros((200, 500, 3), np.uint8),
                                              "检测到抖动！", (50, 80), color=(255, 0, 255))
                        alert_start = time.time()
                        try:
                            root.after(0, lambda m=messagebox: m.showwarning("抖动报警", "检测到持续抖屏，已保存录像"))
                        except Exception as e:
                            log_event("异常", f"抖动弹窗失败: {e}")
                        # 报警后重置 jitter_start，避免重复连续报警；下次再检测到抖动会重新计时
                        jitter_start = None
                else:
                    # 若本帧未检测到抖动，则重置抖动开始时间（要求连续抖动）
                    jitter_start = None
            # —— 抖动（ORB）/花屏（SSIM）检测维持你原有逻辑，但加上空帧保护与弹窗 —— 

            # 先做预处理（你原有）
            proc = preprocess(frame)

            # ORB 抖动（原来的两处报警合并并加弹窗）
            if jitter_cnt >= 1 and time.time() - last_alarm > ALERT_DURATION:
                log_event("抖动", "ORB 检测到抖动 平移>0.5px")
                last_alarm = time.time()
                save_video(list(buf), "jitter_orb")
                alert_img = draw_text(
                    np.zeros((200, 500, 3), np.uint8),
                    "检测到抖动！",
                    (50, 80),
                    color=(255, 0, 255)
                )
                alert_start = time.time()
                try:
                    root.after(0, lambda: messagebox.showwarning("抖动报警", "检测到摄像头画面抖动/闪烁，已保存录像"))
                except Exception as e:
                    log_event("异常", f"抖动弹窗失败: {e}")
                jitter_cnt = 0

            # 更新 prev（放到处理后，确保 prev 总是上一帧）
            prev = frame.copy()

            # 只在 frame_idx - last_art >= 10 时做花屏/更复杂检测（保留你的逻辑）
            frame_idx += 1
            if frame_idx - last_art >= 10:
                try:
                    gray_prev2 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                    gray_cur2  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sim_val = ssim(gray_prev2, gray_cur2)
                    ratio = detect_artifacts(frame)
                except Exception as e:
                    log_event("异常", f"花屏检测异常: {e}")
                    sim_val = 1.0
                    ratio = 0.0

                if sim_val < 0.9 and ratio > 0.5:
                    art_cnt += 1
                else:
                    art_cnt = 0

                if art_cnt >= 3 and time.time() - last_alarm > ALERT_DURATION:
                    log_event("花屏", f"SSIM={sim_val:.2f} ratio={ratio:.2f}")
                    last_alarm = time.time()
                    save_video(list(buf), "artifact")
                    alert_img = draw_text(
                        np.zeros((200, 500, 3), np.uint8),
                        "检测到花屏！",
                        (50, 80),
                        color=(0, 255, 255)
                    )
                    alert_start = time.time()
                    art_cnt = 0
                    last_art = frame_idx
                    try:
                        root.after(0, lambda: messagebox.showwarning("花屏报警", "检测到画面花屏/花屏样式异常，已保存录像"))
                    except Exception as e:
                        log_event("异常", f"花屏弹窗失败: {e}")

                # 在这里插入抖动监控（使用 prev 和 frame 做 ORB 检测）
                if prev is not None:
                    try:
                        gray_prev_orb = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                        gray_cur_orb  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        if detect_jitter_orb(gray_prev_orb, gray_cur_orb, disp_th=0.5, pct_th=0.01):
                            jitter_cnt += 1
                        else:
                            jitter_cnt = 0
                    except Exception as e:
                        log_event("异常", f"ORB 抖动检测异常: {e}")
                        jitter_cnt = 0

            # 屏幕警告窗口显示（OpenCV 窗口，仍保留）
            if alert_start:
                if time.time() - alert_start <= ALERT_DURATION:
                    try:
                        cv2.imshow("屏幕警告", alert_img)
                    except Exception:
                        pass
                else:
                    try:
                        cv2.destroyWindow("屏幕警告")
                    except:
                        pass
                    alert_start = 0

            # 显示实时画面
            cv2.imshow(window, frame)
            cv2.waitKey(1)

    finally:
        try:
            cap.release()
        except:
            pass
        cv2.destroyAllWindows()
        stop_audio()
        log_event("系统", "摄像头检测线程已结束")


# 循环切换分辨率
    write_log("[系统] 开始循环切换分辨率...")
    while switch_event.is_set():
        for mode in valid_modes:
            if not switch_event.is_set():
                break

            device, width, height, refresh_rate = mode
            change_display_resolution(device, width, height, refresh_rate)

            # 切完分辨率后，请求 show_camera_feed 立刻“检测一次黑屏/正常”
            camera_ok_flag = None
            check_camera_event.set()
            # 等待 show_camera_feed 把 camera_ok_flag 设置完毕
            while check_camera_event.is_set() and switch_event.is_set():
                time.sleep(0.1)

            # 此时 show_camera_feed 已写日志，也把 camera_ok_flag 设置好
            if camera_ok_flag is False:
                # 若摄像头线程发出了“强制跳过当前分辨率”的信号，则不再无限等待，直接跳到下一个分辨率
                if FORCE_SKIP_ON_PERSISTENT_BLACK.is_set():
                    write_log("[系统] 强制跳过当前分辨率（长时间黑屏）并继续下一个分辨率")
                    # 清除事件，避免影响后续分辨率
                    FORCE_SKIP_ON_PERSISTENT_BLACK.clear()
                    # 清理 camera_ok_flag，确保下一次分辨率检测是干净状态
                    camera_ok_flag = None
                    # 直接跳过当前分辨率，进入下一个 for-loop 项
                    continue


                # 否则，按原逻辑等待画面恢复（每秒重新请求检测）
                while switch_event.is_set():
                    # 等待 1 秒，再次请求 show_camera_feed 检测
                    time.sleep(1)
                    camera_ok_flag = None
                    check_camera_event.set()
                    while check_camera_event.is_set() and switch_event.is_set():
                        time.sleep(0.1)
                    if camera_ok_flag:
                        # 恢复正常了，写日志后跳出等待
                        write_log("系统", "摄像头恢复正常，继续切换分辨率")
                        break

            # 间隔 10 秒再切下一个分辨率
            time.sleep(10)

def stop_switching():
    write_log("循环切换已停止。")
    switch_event.clear()

# ========== 启动/停止复合逻辑 ==========
def start_loop_combined(res_list):
    """
    点击“启动分辨率循环”时，先让用户选择：
      - 如果点击“是”：设置 camera_thread_active=True，启动摄像头监控线程，并启动分辨率循环线程。
      - 如果点击“否”：设置 camera_thread_active=False，仅启动分辨率循环线程，跳过摄像头检测。
    """
    global camera_thread_active

    if not res_list:
        messagebox.showwarning("提示", "当前没有可用的分辨率，无法启动循环切换。")
        return

    use_cam = messagebox.askyesno("画面检测", "是否同时进行摄像头画面监测？")
    if use_cam:
        camera_thread_active = True
        switch_event.set()
        # 先启动摄像头监控线程
        threading.Thread(target=show_camera_feed, daemon=True).start()
        # 再启动分辨率循环线程，并传入 res_list
        threading.Thread(target=lambda: start_loop_switching(res_list), daemon=True).start()
    else:
        camera_thread_active = False
        # 仅启动分辨率循环线程
        switch_event.set()
        threading.Thread(target=lambda: start_loop_switching(res_list), daemon=True).start()



def stop_loop():
    switch_event.clear()

# ================== 修改后的 start_loop_switching ==================
# 在全局配置区域，加上切换后检测前的“容忍期”（秒）
POST_SWITCH_GRACE = 5

def start_loop_switching(modes):
    """
    传入一个分辨率模式列表 modes，循环切换其中的每一种：
      modes 是 [(device1, width1, height1, hz1), ...]
    当 camera_thread_active=True 时：
      - 每次切换后先等待一个短暂的“容忍期”，再触发摄像头检测；
      - 若黑屏持续超过容忍期，则暂停循环，直到恢复正常后才继续。
    """
    global camera_ok_flag
    write_log("[系统] 开始循环切换分辨率...")
    first_cycle = True

    while switch_event.is_set():
        for device, width, height, refresh_rate in modes:
            if not switch_event.is_set():
                break

            # —— 如果被“暂停”了，就一直等 —— 
            pause_event.wait()

            # —— 真正执行本次分辨率切换 —— 
            change_display_resolution(device, width, height, refresh_rate)

            # 切换后短暂容忍，再触发摄像头检测（仅当启用摄像头监测时）
            if camera_thread_active:
                time.sleep(POST_SWITCH_GRACE)
                camera_ok_flag = None
                check_camera_event.set()
                # 等待摄像头线程处理完检测请求
                while check_camera_event.is_set() and switch_event.is_set():
                    time.sleep(0.1)

                # --- 写一次检测结果日志 ---
                if camera_ok_flag:
                    write_log("[系统] 切换分辨率后检测：分辨率切换成功，摄像头出图正常")
                else:
                    write_log("[系统] 切换分辨率后检测：分辨率切换成功，摄像头监测黑屏")

                    # 再次容忍并做一次确认检测（避免首轮误判）
                    camera_ok_flag = None
                    check_camera_event.set()
                    while check_camera_event.is_set() and switch_event.is_set():
                        time.sleep(0.1)

                    if not first_cycle and camera_ok_flag is False:
                        write_log("[系统] 检测到持续黑屏，暂停自动切换分辨率……")
                        pause_event.clear()
            else:
                # 未启用摄像头检测，直接继续切换（写个日志便于调试）
                write_log("[系统] 未启用摄像头监测，继续切换分辨率")


            first_cycle = False

            # —— 间隔等待，可随时响应停止 —— 
            interval = loop_interval_var.get()
            for _ in range(interval):
                if not switch_event.is_set():
                    break
                time.sleep(1)

    write_log("[系统] 分辨率循环线程已结束。")


# 控制面板操作按钮
def refresh_page():
    write_log("页面已刷新。")
    # 清空有效分辨率列表
    valid_modes.clear()

    # 重新获取显示器信息
    monitor_names.clear()
    monitor_names.extend(list_display_monitors())
    devices.clear()
    devices.extend(list_display_devices())

    # 清空并更新显示器名称列表框
    monitor_listbox.delete(0, tk.END)
    for name in monitor_names:
        monitor_listbox.insert(tk.END, name)
    
    # 强制刷新显示器列表框
    monitor_listbox.update_idletasks()

    # 清空并更新音频设备下拉框
    device_combobox['values'] = []  # 清空音频设备
    device_combobox.set('')  # 设置为空
    audio_devices = find_monitor_audio()  # 重新查找音频设备

    if audio_devices:
        device_combobox['values'] = list(set(audio_devices))  # 去重后插入设备列表
        device_combobox.current(0)  # 默认选择第一个设备
    else:
        device_combobox.set("未找到外部显示器音频设备")

    # 强制刷新音频设备下拉框
    device_combobox.update_idletasks()

    # 清空并更新分辨率列表框
    resolution_listbox.delete(0, tk.END)

    # 如果有选中的显示器，重新更新分辨率列表框
    selected_index = get_selected_index()
    if selected_index is not None:
        update_resolution_listbox(selected_index)
    
    # 强制刷新分辨率列表框
    resolution_listbox.update_idletasks()


def restart_pc():
    write_log("系统正在重启...")
    time.sleep(2)
    os.system("shutdown /r /f /t 0")

def sleep_pc():
    write_log("系统正在进入睡眠状态...")
    ctypes.windll.user32.LockWorkStation()

def hibernate_pc():
    write_log("系统正在进入休眠状态...")
    os.system("shutdown /h")
    
# 笔记本模式：选择外接显示器，使用 \\.\DISPLAY2
def set_notebook():
    log_listbox.insert(tk.END, "选择笔记本模式\n")
    resolution_listbox.delete(0, tk.END)
    valid_modes.clear()

    monitor_names = list_display_monitors()
    log_listbox.insert(tk.END, f"检测到的显示器名称（EDID）：{monitor_names}\n")
    
    # 强制获取 \\.\DISPLAY1 的显示器名称（EDID）
    device_1 = r'\\.\DISPLAY1'  # 使用 DISPLAY1 设备
    
    # 获取与 \\.\DISPLAY1 对应的显示器名称（EDID）
    monitor_name_1 = None
    for edid, dev in edid_to_device.items():
        if dev == device_1:
            monitor_name_1 = edid  # 获取与 \\.\DISPLAY1 对应的显示器名称（EDID）
            break
    
    if not monitor_name_1:
        log_listbox.insert(tk.END, f"没有找到与 {device_1} 对应的显示器名称\n")
        return
    
    # 获取 \\.\DISPLAY2 的显示模式
    device_2 = r'\\.\DISPLAY2'  # 使用 DISPLAY2 设备
    modes = get_display_modes(device_2)
    
    if not modes:
        log_listbox.insert(tk.END, f"设备 {device_2} 没有返回任何显示模式。\n")
    else:
        # 记录所有显示模式
        # log_listbox.insert(tk.END, f"获取到的显示模式: {modes}\n")

        # 用集合来避免重复的分辨率
        seen_resolutions = set()  # 用于跟踪已经显示过的分辨率
        filtered_modes = []

        for mode in modes:
            width, height, refresh_rate = mode
            # 筛选 4K、2K 和 1080P 分辨率，并只保留 240Hz、144Hz、120Hz、60Hz 和 30Hz
            if (width, height) in [(3840, 2160), (2560, 1440), (3440, 1440), (1920, 1080), (1920, 1200), (7680, 4320), (7680, 2160), (5120, 1440), (3840, 1600)]:
                if refresh_rate in [240, 144, 120, 60, 30]:
                    # 检查分辨率是否已经出现过
                    resolution_key = (width, height, refresh_rate)
                    if resolution_key not in seen_resolutions:
                        filtered_modes.append((width, height, refresh_rate))
                        seen_resolutions.add(resolution_key)

        # 按分辨率面积（宽 × 高）和刷新率从大到小排序
        filtered_modes.sort(key=lambda x: (x[0] * x[1], x[2]), reverse=True)

        # 显示符合条件的分辨率模式
        for mode in filtered_modes:
            width, height, refresh_rate = mode
            res_str = f"{width}x{height} @ {refresh_rate}Hz"
            valid_modes.append((device_2, width, height, refresh_rate))
            resolution_listbox.insert(tk.END, f"{monitor_name_1}: {res_str}")
    
    if not valid_modes:
        log_listbox.insert(tk.END, "未找到符合条件的显示模式。\n")


# 台式机模式：选择主显示器（第一个设备）
def set_desktop():
    log_listbox.insert(tk.END, "选择台式机模式\n")
    resolution_listbox.delete(0, tk.END)
    valid_modes.clear()

    devices_list = list_display_devices()
    if devices_list:
        desktop_device = devices_list[0][0]
        log_listbox.insert(tk.END, f"主显示器设备: {desktop_device}\n")
        
        modes = get_display_modes(desktop_device)
        
        # 用集合来避免重复的分辨率
        seen_resolutions = set()  # 用于跟踪已经显示过的分辨率
        
        for mode in modes:
            width, height, refresh_rate = mode
            # 筛选 120Hz、60Hz 和 30Hz 的分辨率
            if refresh_rate in [120, 60, 30]:
                resolution_key = (width, height, refresh_rate)
                # 检查分辨率是否已经出现过
                if resolution_key not in seen_resolutions:
                    # 检查是否是需要的分辨率
                    if (width, height) in [(7680, 4320), (7680, 2160),(5120, 1440), (3840, 2160), (2560, 1440), (3440, 1440), (1920, 1080)]:
                        valid_modes.append((desktop_device, width, height, refresh_rate))
                        res_str = f"{width}x{height} @ {refresh_rate}Hz"
                        resolution_listbox.insert(tk.END, f"主显示器: {res_str}")
                        seen_resolutions.add(resolution_key)
    else:
        log_listbox.insert(tk.END, "没有检测到显示器设备\n")
    
    if not valid_modes:
        log_listbox.insert(tk.END, "未找到符合条件的显示模式。\n")


# 多显模式：
# 根据用户选择的显示器EDID名称，直接将第一个和第二个外接显示器分别指定为 '\\.\DISPLAY2' 和 '\\.\DISPLAY3'
# 只输出 4K（3840×2160）、2K（2560×1440 或 3440×1440）及 1080P（1920×1080，仅保留 60Hz 和 30Hz）模式，
# 并按分辨率面积及刷新率从大到小排序
# 改进的 set_multi_display 函数
def set_multi_display():
    selected_index = monitor_listbox.curselection()
    if not selected_index:
        log_listbox.insert(tk.END, "请先选择一个显示器EDID名称。\n")
        return

    monitor_index = selected_index[0]
    edid_name = monitor_listbox.get(monitor_index)
    
    # 根据选择的索引固定使用对应的设备名
    if monitor_index == 0:
        device = r'\\.\DISPLAY2'
    elif monitor_index == 1:
        device = r'\\.\DISPLAY3'
    elif monitor_index == 2:
        device = r'\\.\DISPLAY1'
    else:
        log_listbox.insert(tk.END, "多显模式仅支持三个外接显示器（即 '\\.\\DISPLAY2'、'\\.\DISPLAY3' 和 '\\.\DISPLAY1'）。\n")
        return

    log_listbox.insert(tk.END, f"获取设备 {edid_name} 的显示模式\n")
    resolution_listbox.delete(0, tk.END)
    valid_modes.clear()
    
    # 获取显示模式
    modes = get_display_modes(device)
    
    if not modes:
        log_listbox.insert(tk.END, f"设备 {edid_name} 没有返回任何有效的显示模式，强制设置外部分辨率。\n")
        modes = [
            (3840, 2160, 60),  # 4K 60Hz
            (2560, 1440, 60),  # 2K 60Hz
            (1920, 1080, 60),  # 1080p 60Hz
            (1920, 1080, 30),  # 1080p 30Hz
            (720, 576, 50),    # 720x576 50Hz
            (720, 480, 60),    # 720x480 60Hz
        ]
    
    # 用集合来避免重复的分辨率
    seen_resolutions = set()  # 用于跟踪已经显示过的分辨率
    filtered_modes = []

    for mode in modes:
        width, height, refresh_rate = mode
        # 筛选 240Hz、120Hz、60Hz 和 30Hz 的分辨率
        if refresh_rate in [240, 120, 60, 30]:
            resolution_key = (width, height, refresh_rate)
            # 检查分辨率是否已经出现过
            if resolution_key not in seen_resolutions:
                # 筛选符合条件的分辨率（4K、2K、1080P、720p）
                if (width, height) == (3840, 2160) or (width, height) in [(2560, 1440), (3440, 1440)] or (width, height) == (1920, 1080) or (width, height) == (720, 576) or (width, height) == (720, 480):
                    filtered_modes.append((device, width, height, refresh_rate))
                    seen_resolutions.add(resolution_key)

    # 按分辨率面积（宽 × 高）和刷新率从大到小排序
    filtered_modes.sort(key=lambda x: (x[1] * x[2], x[3]), reverse=True)
    
    if filtered_modes:
        for mode in filtered_modes:
            _, width, height, refresh_rate = mode
            res_str = f"{width}x{height} @ {refresh_rate}Hz"
            valid_modes.append(mode)
            resolution_listbox.insert(tk.END, f"{edid_name}: {res_str}")
    else:
        log_listbox.insert(tk.END, f"设备 {edid_name} 没有符合条件的显示模式。\n")

# -------------------- 窗口界面设置 --------------------
root = tk.Tk()
root.title("屏幕分辨率管理")
#root.geometry("900x800")
# ——— 新增：自动居中窗口 ———
root.update_idletasks()
screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()
win_w, win_h = 900, 800
x = (screen_w - win_w) // 2
y = (screen_h - win_h) // 2
root.geometry(f"{win_w}x{win_h}+{x}+{y}")
# —————————————————————————————————
# root.resizable(False, False)
root.resizable(True, True)

def alert_black(duration):
    # 由摄像头线程调用
    root.after(0, lambda: messagebox.showwarning("黑屏报警", f"检测到持续 {duration:.1f}s 黑屏"))
    
# 获取EDID名称和设备信息，并建立映射
monitor_names = list_display_monitors()
devices = list_display_devices()  # 每项为 (DeviceName, DeviceString)
valid_modes = []  # 保存当前显示模式列表
edid_to_device = get_edid_device_mapping()  # {EDID_name: DeviceName}

# 日志中输出初始映射关系
initial_log = "初始EDID与设备映射：\n"
for edid, dev in edid_to_device.items():
    initial_log += f"{edid} -> {dev}\n"

# 显示 EDID 名称列表
monitor_frame = Frame(root)
# monitor_frame.pack(pady=10)
# monitor_frame.pack(pady=10, fill="both", expand=True)
monitor_frame.pack(pady=4, fill="x")   # 把 expand=True 去掉，pady 改小
monitor_label = Label(monitor_frame, text="显示器名称 (EDID):", font=("微软雅黑", 12))
monitor_label.pack()
monitor_listbox = Listbox(monitor_frame, width=50, height=4)
monitor_listbox.pack()
for name in monitor_names:
    monitor_listbox.insert(tk.END, name)
monitor_listbox.bind('<<ListboxSelect>>', on_monitor_select)

# 音频设备下拉框
audio_frame = Frame(root)
audio_frame.pack(pady=2)
audio_label = Label(audio_frame, text="外部显示器音频名称:", font=("微软雅黑", 12))
audio_label.pack()

device_combobox = ttk.Combobox(audio_frame, width=50)
device_combobox.pack()

# 启动时自动检测音频设备
display_audio_device()

# ================== 新增：外部显示器音频监测 ==================
# 全局控制变量
audio_monitor_event = threading.Event()   # 停止标志（置位表示停止）
audio_monitor_thread = None
audio_monitor_active = False
# 音频静音窗口相关全局
silence_window = None
silence_label_var = None
silence_current_device = None

# 确保告警冷却变量被初始化
_audio_last_silence_alert = 0.0

# 参数（可按需调整）
AUDIO_SR = 44100                   # 采样率
AUDIO_CHUNK_SEC = 0.2              # 每次读取时长（秒）
AUDIO_SILENCE_RMS = 1e-4           # 低于该 RMS 视为“近似静音”
AUDIO_SILENCE_DURATION = 5.0       # 连续静音多少秒算“静音报警”
AUDIO_STUTTER_DROP_PCT = 0.3       # 短期内低于 RMS 阈值块的比例超过该值视为卡顿
AUDIO_STUTTER_WINDOW_SEC = 3.0     # 用于判断卡顿的滑动窗口长度（秒）
AUDIO_ALERT_COOLDOWN = 5           # 报警冷却（秒），避免频繁弹窗
# 新增：判断是否“有播放活动”的阈值与窗口
PLAYBACK_ACTIVITY_RMS = 1e-3         # 若最近块中有 rms > 该值，则认为正在播放（按需微调）
PLAYBACK_ACTIVITY_WINDOW_SEC = 3.0   # 查看最近多少秒来判定“是否有播放”
PLAYBACK_WARMUP_SEC = 1.0            # 启动检测后给些稳态时间，避免刚启动误判
# 去抖与计时用的全局变量（放在文件开头参数区）
last_shown_silence = -1
_last_sound_ts = time.time()

# 内部状态
_audio_rms_deque = deque()          # 存放最近的 rms 值（时间戳, rms）
_audio_last_silence_alert = 0
_audio_last_stutter_alert = 0

def find_device_index_by_name(name):
    """按设备名称查找 sounddevice 的设备索引（先精确再模糊匹配）"""
    devs = sd.query_devices()
    for i, d in enumerate(devs):
        if d and 'name' in d and d['name'] == name:
            return i, d
    for i, d in enumerate(devs):
        if d and 'name' in d and name.lower() in d['name'].lower():
            return i, d
    return None, None

def _audio_alert_popup(title, msg):
    """主线程弹窗+写日志（安全地从后台线程调用）"""
    try:
        write_log(f"[音频] {title}: {msg}")
        root.after(0, lambda: messagebox.showwarning(title, msg))
    except Exception as e:
        log_event("异常", f"_audio_alert_popup 异常: {e}")

def show_silence_window(duration_sec, device_name):
    global silence_window, silence_label_var, silence_current_device
    if silence_window is not None:
        silence_current_device = device_name
        try:
            silence_label_var.set(f"检测到持续静音 {int(duration_sec)} 秒\n设备: {device_name}")
        except Exception:
            pass
        return

    silence_window = tk.Toplevel(root)
    silence_window.title("外部显示器无声音")
    w, h = 360, 120
    try:
        rx = root.winfo_rootx(); ry = root.winfo_rooty(); rw = root.winfo_width() or 900
        silence_window.geometry(f"{w}x{h}+{rx + rw//3}+{ry + 100}")
    except Exception:
        silence_window.geometry(f"{w}x{h}")
    silence_window.resizable(False, False)

    def _on_close():
        close_silence_window()
    silence_window.protocol("WM_DELETE_WINDOW", _on_close)

    silence_current_device = device_name
    silence_label_var = tk.StringVar()
    silence_label_var.set(f"检测到持续静音 {int(duration_sec)} 秒\n设备: {device_name}")

    lbl = tk.Label(silence_window, textvariable=silence_label_var, font=("微软雅黑", 12), justify="center")
    lbl.pack(padx=10, pady=(12,6))
    btn = tk.Button(silence_window, text="关闭", width=10, command=_on_close)
    btn.pack(pady=(0,10))


def update_silence_window(duration_sec, device_name=None):
    global silence_label_var, silence_current_device
    if silence_window is None:
        return
    if device_name:
        silence_current_device = device_name
    try:
        silence_label_var.set(f"检测到持续静音 {int(duration_sec)} 秒\n设备: {silence_current_device}")
    except Exception:
        pass


def close_silence_window():
    global silence_window, silence_label_var, silence_current_device
    try:
        if silence_window is not None:
            silence_window.destroy()
    except Exception:
        pass
    silence_window = None
    silence_label_var = None
    silence_current_device = None


def audio_monitor_worker(device_name):
    """
    改良版 audio_monitor_worker：
    - 兼容新/旧 sounddevice WASAPI loopback 调用方式；
    - 优先按名称解析设备，若设备为纯输出(in=0)则尝试查找 loopback 条目；
    - 找不到 loopback 则回退到 Stereo Mix / VB-Cable（如存在）；
    - 使用 callback+queue 读取数据，主循环计算 RMS 并报警。
    """
    global last_shown_silence, _last_sound_ts
    # global _audio_rms_deque, _audio_last_silence_alert, _audio_last_stutter_alert, audio_monitor_active

    audio_monitor_event.clear()
    audio_monitor_active = True

    # 初始化线程内状态
    playback_seen = False
    last_activity_ts = 0.0
    start_time = time.time()

    chunk = int(AUDIO_SR * AUDIO_CHUNK_SEC)

    # 安全获取设备列表 helper
    def get_devices():
        try:
            return sd.query_devices()
        except Exception:
            return []

    # 按名字查索引（精确->子串->关键词）
    def find_device_index_by_name_prefer_exact(name):
        if not name:
            return None, None
        target = (name or "").strip().lower()
        devs = get_devices()
        for i, d in enumerate(devs):
            if (d.get('name') or '').strip().lower() == target:
                return i, d
        for i, d in enumerate(devs):
            if target in (d.get('name') or '').lower():
                return i, d
        keywords = target.split()
        for i, d in enumerate(devs):
            n = (d.get('name') or '').lower()
            if all(k in n for k in keywords):
                return i, d
        return None, None

    # 查找 loopback 条目（名字含 'loopback' 或设备名里含原始名字且有 input 通道）
    def find_loopback_device_for_name(name_substr=None):
        devs = get_devices()
        # 首先找带 loopback 的条目
        for i, d in enumerate(devs):
            n = (d.get('name') or '').lower()
            if 'loopback' in n and d.get('max_input_channels', 0) > 0:
                return i, d
        # 再找名字包含目标名且有 input 通道
        if name_substr:
            for i, d in enumerate(devs):
                n = (d.get('name') or '').lower()
                if name_substr.lower() in n and d.get('max_input_channels', 0) > 0:
                    return i, d
        return None, None

    # Stereo Mix 回退查找
    def find_stereo_mix_device_local():
        devs = get_devices()
        for ii, dd in enumerate(devs):
            name = (dd.get('name') or "").lower()
            if dd.get('max_input_channels', 0) > 0 and 'stereo' in name and 'mix' in name:
                return ii, dd
        for ii, dd in enumerate(devs):
            name = (dd.get('name') or "").lower()
            if dd.get('max_input_channels', 0) > 0 and ('mix' in name or '立体声' in name or 'wave out' in name or 'stereo' in name):
                return ii, dd
        return None, None

    # 测试能否用 blocking 模式打开（用于探测）
    def try_open_blocking_by_index(idx, ch, use_wasapi, device_name_hint=None):
        try:
            if use_wasapi:
                # 尝试两种策略：
                # 1) 若 sd.WasapiSettings 支持 loopback 参数，直接使用
                try:
                    was = sd.WasapiSettings(loopback=True)
                    s = sd.InputStream(device=idx, channels=ch,
                                       samplerate=AUDIO_SR, blocksize=chunk,
                                       dtype='float32', extra_settings=was)
                    s.start(); s.stop(); s.close()
                    return True, None
                except TypeError:
                    # 新版 sounddevice 不支持 loopback 参数：尝试查找 (loopback) 设备并直接打开该 index
                    loop_idx, _ = find_loopback_device_for_name(device_name_hint)
                    if loop_idx is not None:
                        s = sd.InputStream(device=loop_idx, channels=ch,
                                           samplerate=AUDIO_SR, blocksize=chunk,
                                           dtype='float32')
                        s.start(); s.stop(); s.close()
                        return True, ("used_loopback_index", loop_idx)
                    else:
                        return False, "wasapi_loopback_not_supported_and_no_loopback_device"
                except Exception as e:
                    return False, e
            else:
                s = sd.InputStream(device=idx, channels=ch,
                                   samplerate=AUDIO_SR, blocksize=chunk,
                                   dtype='float32')
                s.start(); s.stop(); s.close()
                return True, None
        except Exception as e:
            return False, e

    # ========== 开始解析设备 ==========
    # 先按传入 device_name 找索引
    try:
        dev_idx, dev_info = find_device_index_by_name_prefer_exact(device_name)
    except Exception:
        dev_idx, dev_info = None, None

    tried = []
    stream_device_index = None
    stream_channels = 1
    resolved_device_name = None
    wasapi_available = True
    try:
        _ = sd.WasapiSettings()  # 仅检测能否构造 WasapiSettings (不传 loopback)
    except Exception:
        wasapi_available = False

    # 优先尝试用户指定的设备（如果找到）
    if dev_idx is not None and dev_info is not None:
        dev_max_ch = int(dev_info.get('max_output_channels', 1) or 1)
        tried.append((dev_idx, dev_max_ch))
        ok = False; err = None
        # 如果该设备本身有 input 通道，可以直接尝试打开（无需 loopback）
        if dev_info.get('max_input_channels', 0) > 0:
            ok, err = try_open_blocking_by_index(dev_idx, min(2, dev_info.get('max_input_channels', 1)), False, device_name)
            if ok:
                stream_device_index = dev_idx
                stream_channels = min(2, dev_info.get('max_input_channels', 1))
                resolved_device_name = dev_info.get('name')
        else:
            # 设备为输出（in==0），尝试用 WASAPI loopback（若可用）
            if wasapi_available:
                ok, err = try_open_blocking_by_index(dev_idx, dev_max_ch, True, device_name)
                if ok:
                    # 如果 try_open_blocking_by_index 返回用 loopback index，处理返回值
                    if isinstance(err, tuple) and err[0] == "used_loopback_index":
                        stream_device_index = err[1]
                        stream_channels = dev_max_ch
                        resolved_device_name = sd.query_devices()[stream_device_index].get('name')
                    else:
                        stream_device_index = dev_idx
                        stream_channels = dev_max_ch
                        resolved_device_name = dev_info.get('name')
            # 若以上都不行，尝试直接以输出设备打开（可能失败，但在某些系统可行）
            if stream_device_index is None:
                ok, err = try_open_blocking_by_index(dev_idx, dev_max_ch, False, device_name)
                if ok:
                    stream_device_index = dev_idx
                    stream_channels = dev_max_ch
                    resolved_device_name = dev_info.get('name')

    # 如果还没找到，扫描系统输出设备（优先非 HDMI）
    if stream_device_index is None:
        all_devs = get_devices()
        candidates = []
        for i, d in enumerate(all_devs):
            if d.get('max_output_channels', 0) > 0:
                name = (d.get('name') or "").lower()
                priority = 1 if ('monitor' in name or 'hdmi' in name or 'display' in name) else 0
                candidates.append((priority, i, d))
        candidates.sort(key=lambda x: x[0])
        for priority, i, d in candidates:
            dev_max_ch = int(d.get('max_output_channels', 1) or 1)
            tried.append((i, dev_max_ch))
            # 若设备本身有 input 通道，直接尝试
            if d.get('max_input_channels', 0) > 0:
                ok, err = try_open_blocking_by_index(i, min(2, d.get('max_input_channels', 1)), False, d.get('name'))
                if ok:
                    stream_device_index = i
                    stream_channels = min(2, d.get('max_input_channels', 1))
                    resolved_device_name = d.get('name')
                    break
            else:
                # 尝试以 wasapi loopback 方式（若可）
                if wasapi_available:
                    ok, err = try_open_blocking_by_index(i, dev_max_ch, True, d.get('name'))
                    if ok:
                        if isinstance(err, tuple) and err[0] == "used_loopback_index":
                            stream_device_index = err[1]
                            stream_channels = dev_max_ch
                            resolved_device_name = sd.query_devices()[stream_device_index].get('name')
                        else:
                            stream_device_index = i
                            stream_channels = dev_max_ch
                            resolved_device_name = d.get('name')
                        break
                # 尝试普通打开（不常见）
                ok, err = try_open_blocking_by_index(i, dev_max_ch, False, d.get('name'))
                if ok:
                    stream_device_index = i
                    stream_channels = dev_max_ch
                    resolved_device_name = d.get('name')
                    break

    # 回退：优先查找 VB-Cable / Stereo Mix 输入设备
    if stream_device_index is None:
        # 优先找 vb-cable-like (名字含 'cable')
        devs = get_devices()
        for ii, dd in enumerate(devs):
            n = (dd.get('name') or '').lower()
            if 'cable' in n and dd.get('max_input_channels', 0) > 0:
                stream_device_index = ii
                stream_channels = min(2, dd.get('max_input_channels', 1))
                resolved_device_name = dd.get('name')
                log_event("音频", f"回退到虚拟线设备: {resolved_device_name} (index={ii})")
                break
        # 再找 Stereo Mix
        if stream_device_index is None:
            sm_idx, sm_info = find_stereo_mix_device_local()
            if sm_idx is not None:
                stream_device_index = sm_idx
                stream_channels = min(2, max(1, int(sm_info.get('max_input_channels', 1) or 1)))
                resolved_device_name = sm_info.get('name')
                log_event("音频", f"回退到录音设备: {resolved_device_name} (index={sm_idx})")

    if stream_device_index is None:
        _audio_alert_popup(
            "音频监测启动失败",
            (f"无法为设备 '{device_name}' 打开回环录音（尝试过索引: {tried}）。\n"
             "建议：启用 Stereo Mix、或安装 VB-Cable、或选择非 HDMI 的扬声器输出。")
        )
        audio_monitor_active = False
        return

    # 最终用于创建流的 device 参数：优先使用解析到的设备名称字符串（更稳健）
    try:
        dev_info_dbg = sd.query_devices()[stream_device_index]
        device_for_stream = dev_info_dbg.get('name') or stream_device_index
        # 写日志确认实际使用的设备
        root.after(0, lambda m=f"[音频] 使用监测设备: index={stream_device_index}, name='{device_for_stream}', in:{dev_info_dbg.get('max_input_channels')} out:{dev_info_dbg.get('max_output_channels')}": write_log(m))
    except Exception:
        device_for_stream = stream_device_index
        root.after(0, lambda: write_log(f"[音频] 使用监测设备 index={stream_device_index}"))

    # 创建回调队列与回调
    q = queue.Queue(maxsize=60)
    def callback(indata, frames, time_info, status):
        try:
            if status:
                log_event("音频", f"callback status: {status}")
            q.put(indata.copy(), block=False)
        except queue.Full:
            try:
                _ = q.get_nowait()
                q.put(indata.copy(), block=False)
            except Exception:
                pass

    # 创建 InputStream（如果需要 extraSettings: 先尝试构造 WasapiSettings(loopback=True)，若不支持则直接用 device_for_stream）
    try:
        use_extra = False
        extra = None
        # 判断是否需要/能使用 WasapiSettings(loopback=True)
        # 如果 resolved_device_name 含 'loopback'，直接用 device_for_stream(index/name) 不加 extra
        if wasapi_available:
            try:
                # 试探性构造（新版本可能不接受 loopback kw）
                extra_try = sd.WasapiSettings(loopback=True)
                extra = extra_try
                use_extra = True
            except TypeError:
                # 新版本不接受 loopback kw：不抛，使用 loopback-index如果上面找到过
                loop_idx, loop_info = find_loopback_device_for_name(device_name)
                if loop_idx is not None:
                    device_for_stream = loop_idx
                    use_extra = False
                else:
                    use_extra = False
            except Exception:
                use_extra = False

        if use_extra and extra is not None:
            stream_cb = sd.InputStream(device=device_for_stream,
                                       channels=stream_channels,
                                       samplerate=AUDIO_SR,
                                       blocksize=chunk,
                                       dtype='float32',
                                       callback=callback,
                                       extra_settings=extra)
        else:
            stream_cb = sd.InputStream(device=device_for_stream,
                                       channels=stream_channels,
                                       samplerate=AUDIO_SR,
                                       blocksize=chunk,
                                       dtype='float32',
                                       callback=callback)
        stream_cb.start()
        log_event("音频", f"音频回调流已启动: index={stream_device_index}, channels={stream_channels}, wasapi_extra_used={use_extra}")
    except Exception as e:
        # 记录原因并提示回退
        log_event("音频", f"创建回调流失败: {e}")
        # 若未尝试过 stereo mix，尝试回退
        sm_idx, sm_info = find_stereo_mix_device_local()
        if sm_idx is not None and sm_idx != stream_device_index:
            try:
                device_for_stream = sm_idx
                stream_cb = sd.InputStream(device=device_for_stream,
                                           channels=min(2, sm_info.get('max_input_channels',1)),
                                           samplerate=AUDIO_SR,
                                           blocksize=chunk,
                                           dtype='float32',
                                           callback=callback)
                stream_cb.start()
                stream_device_index = sm_idx
                stream_channels = min(2, sm_info.get('max_input_channels',1))
                resolved_device_name = sm_info.get('name')
                log_event("音频", f"回退并成功使用 Stereo Mix: index={sm_idx}, name={resolved_device_name}")
            except Exception as e2:
                _audio_alert_popup("音频监测启动失败", f"创建回调流失败(回退 Stereo Mix 也失败): {e2}")
                audio_monitor_active = False
                return
        else:
            _audio_alert_popup("音频监测启动失败", f"创建回调流失败: {e}")
            audio_monitor_active = False
            return

    # ===== 启动时短暂自校准噪声基线（放在 stream_cb.start() 后，进入主循环前） =====
    CALIB_SECONDS = 2.0
    CALIB_SAMPLE_COUNT = max(3, int(CALIB_SECONDS / AUDIO_CHUNK_SEC))
    baseline_samples = []
    for _ in range(CALIB_SAMPLE_COUNT):
        try:
            block = q.get(timeout=1.0)
        except queue.Empty:
            baseline_samples.append(0.0)
            continue
        try:
            arr = np.array(block, dtype=np.float32)
            if arr.ndim > 1:
                sample_rms = float(np.sqrt(np.mean(np.square(arr))))
            else:
                sample_rms = float(np.sqrt(np.mean(np.square(arr))))
        except Exception:
            sample_rms = 0.0
        baseline_samples.append(sample_rms)

    baseline_noise = float(np.median(baseline_samples)) if baseline_samples else 0.0
    ABSOLUTE_MIN_ACTIVITY = 1e-6
    ACTIVITY_FACTOR = 1.5
    activity_threshold = max(baseline_noise * ACTIVITY_FACTOR, ABSOLUTE_MIN_ACTIVITY)
    try:
        root.after(0, lambda: write_log(f"[音频] 自校准: baseline_noise={baseline_noise:.6e}, activity_threshold={activity_threshold:.6e}"))
    except Exception:
        print(f"[音频] 自校准: baseline_noise={baseline_noise:.6e}, activity_threshold={activity_threshold:.6e}")
    # ===== 校准结束 =====

    # ===== 主循环：从队列取数据，计算 RMS 并执行静音/卡顿判定 =====
    try:
        while not audio_monitor_event.is_set():
            try:
                data_block = q.get(timeout=1.0)
                has_data = True
            except queue.Empty:
                data_block = None
                has_data = False

            ts = time.time()
            if not has_data:
                rms = 0.0
            else:
                try:
                    arr = np.array(data_block, dtype=np.float32)
                    rms = float(np.sqrt(np.mean(np.square(arr))))
                except Exception:
                    rms = 0.0

            # 滑动队列记录
            _audio_rms_deque.append((ts, rms))
            while _audio_rms_deque and (_audio_rms_deque[0][0] < ts - AUDIO_STUTTER_WINDOW_SEC - 1.0):
                _audio_rms_deque.popleft()

            # 计算短窗口内最大值
            activity_window_start = ts - PLAYBACK_ACTIVITY_WINDOW_SEC
            recent_vals = [r for t, r in _audio_rms_deque if t >= activity_window_start]
            recent_max = max(recent_vals) if recent_vals else 0.0

            # 备用静音阈值（避免过分依赖单一常量）
            silence_threshold_dynamic = max(AUDIO_SILENCE_RMS, (globals().get('baseline_noise', 0.0)) * 0.6)
            playback_fallback_threshold = max((globals().get('activity_threshold', activity_threshold)) * 0.8,
                                            silence_threshold_dynamic * 3.0, 1e-7)

            # 判定播放（若短窗口内出现显著音量）
            if recent_max > activity_threshold or recent_max > playback_fallback_threshold or rms > playback_fallback_threshold:
                last_activity_ts = ts
                playback_seen = True
                _last_sound_ts = ts   # <- 这里记录“最后一次有声音”的时间戳

            # 用时间戳计算从最后一次检测到有声音到现在的静音时长（更稳定）
            cont_silent_seconds = max(0.0, ts - _last_sound_ts) if playback_seen else 0.0

            # 调试日志（线程安全）
            debug_line = (f"[音频-调试] t={time.strftime('%H:%M:%S', time.localtime(ts))} "
                        f"rms={rms:.6f} recent_max={recent_max:.6f} "
                        f"baseline={globals().get('baseline_noise',0.0):.6e} activity_th={activity_threshold:.6e} "
                        f"fallback_th={playback_fallback_threshold:.6e} silence_th={silence_threshold_dynamic:.6e} "
                        f"playback_seen={playback_seen} cont_silent={cont_silent_seconds:.2f}s")
            try:
                root.after(0, lambda ln=debug_line: write_log(ln))
            except Exception:
                print(debug_line)

            # 单一静音窗口逻辑（线程安全 via root.after）
            # 使用全局 AUDIO_SILENCE_DURATION 作为阈值（你可以调整这个常量）
            if playback_seen and cont_silent_seconds >= AUDIO_SILENCE_DURATION:
                sec_int = int(cont_silent_seconds)
                # 仅在整数秒变化时更新 UI（去抖）
                if sec_int != globals().get('last_shown_silence', -1):
                    globals()['last_shown_silence'] = sec_int
                    root.after(0, lambda s=cont_silent_seconds, d=device_name: (
                        show_silence_window(s, d) if silence_window is None else update_silence_window(s, d)
                    ))
            else:
                # 一旦检测到“有声音恢复”，自动关闭提示窗口
                globals()['last_shown_silence'] = -1
                if silence_window is not None:
                    root.after(0, close_silence_window)



            # 卡顿判定
            # if playback_seen and (time.time() - start_time) > PLAYBACK_WARMUP_SEC:
            #     window_start_ts = ts - AUDIO_STUTTER_WINDOW_SEC
            #     vals = [r for t, r in _audio_rms_deque if t >= window_start_ts]
            #     if vals:
            #         low_count = sum(1 for v in vals if v < (AUDIO_SILENCE_RMS * 2))
            #         drop_pct = low_count / len(vals)
            #         if drop_pct >= AUDIO_STUTTER_DROP_PCT and (ts - (_audio_last_stutter_alert or 0)) > AUDIO_ALERT_COOLDOWN:
            #             _audio_last_stutter_alert = ts
            #             root.after(0, lambda: _audio_alert_popup("外部显示器声音卡顿", f"检测到短期内声音中断/抖动（{drop_pct*100:.0f}%）设备: {device_name}"))

            # time.sleep(0.005)

    finally:
        try:
            stream_cb.stop()
            stream_cb.close()
        except Exception:
            pass
        audio_monitor_active = False
        log_event("系统", f"音频监测线程已结束: {device_name}")


def start_audio_monitoring():
    """GUI 调用：开始监测当前下拉框选中的外部设备"""
    global audio_monitor_thread, audio_monitor_event, audio_monitor_active
    if audio_monitor_active:
        messagebox.showinfo("提示", "音频监测已在运行")
        return
    dev = device_combobox.get()
    if not dev or "未找到" in dev:
        messagebox.showwarning("提示", "请先在下拉框选择外部显示器音频设备")
        return
    audio_monitor_event.clear()
    audio_monitor_thread = threading.Thread(target=audio_monitor_worker, args=(dev,), daemon=True)
    audio_monitor_thread.start()
    write_log(f"[系统] 音频监测已请求启动: {dev}")

def stop_audio_monitoring():
    """GUI 调用：停止音频监测"""
    global audio_monitor_event, audio_monitor_thread
    if not audio_monitor_active and (audio_monitor_thread is None or not audio_monitor_thread.is_alive()):
        write_log("[系统] 音频监测未运行")
        return
    audio_monitor_event.set()
    write_log("[系统] 请求停止音频监测（等待线程结束）")

btn_frame_audio = Frame(audio_frame)
btn_frame_audio.pack(pady=5)

Button(btn_frame_audio, text="启动音频监测", command=start_audio_monitoring, font=("微软雅黑", 10)).pack(side=tk.LEFT, padx=5)
Button(btn_frame_audio, text="停止音频监测", command=stop_audio_monitoring, font=("微软雅黑", 10)).pack(side=tk.LEFT, padx=5)


# 按钮区域
button_frame = Frame(root)
button_frame.pack(pady=10)


Button(button_frame, text="笔记本", command=set_notebook, font=("微软雅黑", 12)).pack(side=tk.LEFT, padx=10)
Button(button_frame, text="台式机", command=set_desktop, font=("微软雅黑", 12)).pack(side=tk.LEFT, padx=10)
Button(button_frame, text="多显", command=set_multi_display, font=("微软雅黑", 12)).pack(side=tk.LEFT, padx=10)

# 分辨率列表区域
resolution_frame = Frame(root)
resolution_frame.pack(pady=10)
resolution_label = Label(resolution_frame, text="显示器分辨率:", font=("微软雅黑", 12))
resolution_label.pack()
resolution_listbox = Listbox(resolution_frame, width=50, height=10)
resolution_listbox.pack()
resolution_listbox.bind('<<ListboxSelect>>', on_resolution_select)

# 日志显示区域
log_frame = Frame(root)
log_frame.pack(pady=10)
log_label = Label(log_frame, text="日志:", font=("微软雅黑", 12))
log_label.pack()
# log_listbox = Listbox(log_frame, width=100, height=10)
log_listbox = Listbox(log_frame, width=100, height=10, font=("微软雅黑", 10))
log_listbox.pack()
log_listbox.insert(tk.END, initial_log)

# 按钮区域
button_frame = Frame(root)
button_frame.pack(pady=10)

# ——— 新增：循环间隔输入框 ———
interval_frame = Frame(button_frame)
interval_frame.pack(side=tk.LEFT, padx=5)

tk.Label(interval_frame, text="用户选择循环时间(s):", font=("微软雅黑", 12)).pack(side=tk.LEFT)
loop_interval_var = tk.IntVar(value=10)
tk.Spinbox(
    interval_frame,
    from_=1, to=3600,
    textvariable=loop_interval_var,
    width=5,
    font=("微软雅黑", 12)
).pack(side=tk.LEFT)

Button(button_frame,
       text="启动分辨率循环",
       command=lambda: start_loop_combined(valid_modes),
       font=("微软雅黑", 12)
).pack(side=tk.LEFT, padx=10)

Button(button_frame, text="停止分辨率循环", command=stop_switching, font=("微软雅黑", 12)).pack(side=tk.LEFT, padx=10)
Button(button_frame, text="刷新", command=refresh_page, font=("微软雅黑", 12)).pack(side=tk.LEFT, padx=10)
Button(button_frame, text="重启计算机", command=restart_pc, font=("微软雅黑", 12)).pack(side=tk.LEFT, padx=10)
Button(button_frame, text="睡眠", command=sleep_pc, font=("微软雅黑", 12)).pack(side=tk.LEFT, padx=10)
Button(button_frame, text="休眠", command=hibernate_pc, font=("微软雅黑", 12)).pack(side=tk.LEFT, padx=10)

root.mainloop()
