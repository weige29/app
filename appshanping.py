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


# 写入日志函数
# def write_log(message):
#     # 获取当前时间
#     current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#     log_message = f"[{current_time}] {message}"
    
#     # 更新Listbox
#     log_listbox.insert(tk.END, log_message)
    
#     # 将日志写入到文件
#     with open("自动分辨率日志.txt", "a", encoding="utf-8") as log_file:
#         log_file.write(log_message + "\n")
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


# ================= 全局变量，用于线程间“立刻检测一次摄像头黑屏/正常” =================
check_camera_event = threading.Event()
camera_ok_flag = None
last_camera_ok_flag = None   # 记录上次检测结果，初始 None

def show_camera_feed():
    """
    摄像头检测模块：一直打开一个 VideoCapture(1)，不断读帧并在需要时告警。
    同时它会响应 check_camera_event，如果这个事件被置位，就立刻做一次“读一帧检测黑屏/正常”，并把结果放到 camera_ok_flag 里。
    """
    global camera_ok_flag, last_camera_ok_flag

     # ========== 新增：第一次检测摄像头状态时，不写黑屏日志 ========== 
    first_check = True
    # ============================================================

    # 只在程序启动时打一遍 log，之后不要在这个函数里反复写“切分辨率成功”之类的日志
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
    # ← 在这里初始化抖动计数器
    jitter_cnt = 0
    last_art = 0
    alert_start = 0
    alert_img = None
    # 抖动检测（改良版）状态与参数
    JITTER_WINDOW_SEC = 1.0        # 在过去 1 秒内统计抖动次数
    JITTER_MIN_COUNT = 2          # 窗口内至少出现多少次判为抖动（可调）
    JITTER_DISP_TH = 0.5          # detect_jitter_orb 的位移阈值，减小可提高灵敏度（示例 0.5 -> 0.3）
    JITTER_PCT_TH = 0.01          # detect_jitter_orb 的匹配比例阈值（示例 0.01 -> 1%）
    jitter_times = deque()        # 存存检测到抖动的时间戳

    # 先播放监控音
    play_audio()
    log_event("系统", "摄像头检测线程已启动，绘制窗口以关闭结束监控")

    try:
        while True:
            # ========== 检查摄像头窗口是否被用户关闭 ==========
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    log_event("系统", "用户关闭了摄像头监控窗口，show_camera_feed 退出")
                    # 在此处清除 switch_event，让分辨率循环也停止
                    switch_event.clear()
                    break
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                log_event("异常", tb)
                # 在主线程里弹窗，保证能看到
                from tkinter import messagebox
                messagebox.showerror("监控线程出错", f"{e}\n\n详见日志")

            # —— 检测是否收到了分辨率线程的“请检测一次摄像头状态”请求 —— 
            if check_camera_event.is_set():
                ret_chk, frame_chk = cap.read()
                if ret_chk and not is_black(frame_chk, pct=0.5, lb=60):
                    camera_ok_flag = True
                else:
                    camera_ok_flag = False
                    # ========== 修改点：只有当不是第一次检测时才写“切换后正常”日志 ==========
                #     if not first_check:
                #         write_log("[系统] 切换分辨率后检测：摄像头出图正常")
                #     # =======================================================
                # else:
                #     camera_ok_flag = False
                #     # ========== 修改点：只有当不是第一次检测，才写“切换后黑屏”日志 ==========
                #     if not first_check:
                #         write_log("[系统] 切换分辨率后检测：摄像头黑屏")
                check_camera_event.clear()
                 # —— 切换分辨率后重置黑屏累计时间 —— 
                black_start = 0
                    # =======================================================
                first_check = False                    # 标记：已经做过第一次检测
                # check_camera_event.clear()

            # —— 正常的黑屏 / 闪屏 / 花屏监控逻辑（与之前完全一样，只是不会再写“切分辨率成功”那几条）——
            ret, frame = cap.read()
            if not ret:
                log_event("错误", "无法读取到视频帧，show_camera_feed 退出")
                break
            buf.append(frame.copy())
            frame_idx += 1

            # —— 黑屏监控
            if is_black(frame):
                if black_start == 0:
                    black_start = time.time()
                if time.time() - black_start > BLACK_ALERT_THRESHOLD and time.time() - last_alarm > ALERT_DURATION:
                    # log_event("黑屏", f"持续 {time.time() - black_start:.1f}s")

                    # 暂停分辨率循环（线程不退出）
                    if pause_event.is_set():
                        pause_event.clear()
                        write_log(f"[系统] 检测到持续黑屏 {time.time() - black_start:.1f}s，已暂停自动切换分辨率")
                        log_event("黑屏", f"持续 {time.time() - black_start:.1f}s")
                    last_alarm = time.time()
                    save_video(list(buf), "black_screen")
                    alert_img = draw_text(
                        np.zeros((200, 500, 3), np.uint8),
                        f"黑屏 {time.time() - black_start:.1f}s",
                        (50, 80)
                    )
                    alert_start = time.time()
                    stop_audio()
            else:
                if black_start:
                    # log_event("系统", f"黑屏恢复，持续 {time.time() - black_start:.1f}s")
                    log_event("系统", f"黑屏恢复，持续 {time.time() - black_start:.1f}s")
            # 画面恢复后，重新启动分辨率循环（如果之前是开启检测模式）
                    # if camera_thread_active and not switch_event.is_set():
                    #     switch_event.set()
                    #     write_log("[系统] 摄像头恢复正常，自动切换分辨率已恢复")
                    # 画面恢复，解除挂起，让切换线程继续
                if camera_thread_active and not pause_event.is_set():
                    pause_event.set()
                    write_log("[系统] 摄像头恢复正常，自动切换分辨率已恢复")
                    black_start = 0
                    play_audio()

            # —— 闪屏监控
            proc = preprocess(frame)
            # 连续 1 帧就报警
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
                    jitter_cnt = 0

            # —— 更新 prev 和 frame_idx —— 
            prev = frame.copy()
            frame_idx += 1

            # —— 花屏监控
            if frame_idx - last_art >= 10:
                gray_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                gray_cur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sim_val = ssim(gray_prev, gray_cur)
                ratio = detect_artifacts(frame)
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

                 # —— 在这里插入抖动监控 ——  
                if prev is not None:
                    gray_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                    gray_cur  = cv2.cvtColor(frame,   cv2.COLOR_BGR2GRAY)
                    if detect_jitter_orb(gray_prev, gray_cur, disp_th=0.5, pct_th=0.01):
                        jitter_cnt += 1
                    else:
                        jitter_cnt = 0

                    if jitter_cnt >= 1 and time.time() - last_alarm > ALERT_DURATION:
                        log_event("抖动", f"检测到连续抖动，mag_th=2.0 pct_th=0.1")
                        last_alarm = time.time()
                        save_video(list(buf), "jitter")
                        alert_img = draw_text(
                            np.zeros((200, 500, 3), np.uint8),
                            "检测到抖动！",
                            (50, 80),
                            color=(255, 0, 255)
                        )
                        alert_start = time.time()
                        jitter_cnt = 0

            # —— 显示“屏幕警告”窗口，并在超过 2 秒后自动关闭它 —— 
            if alert_start:
                if time.time() - alert_start <= ALERT_DURATION:
                    cv2.imshow("屏幕警告", alert_img)
                else:
                    # 超过 ALERT_DURATION（2 秒）之后，主动关闭“屏幕警告”窗口
                    try:
                        cv2.destroyWindow("屏幕警告")
                    except:
                        pass
                    alert_start = 0  # 重置，避免反复 destroy

            # —— 显示摄像头画面
            cv2.imshow(window, frame)
            cv2.waitKey(1)

    finally:
        cap.release()
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
                # 如果是黑屏，就一直等待 show_camera_feed 把黑屏恢复为正常
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

# def start_loop_switching():
#     switch_event.set()
#     threading.Thread(target=start_loop_switching, daemon=True).start()

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

            # 切换后短暂容忍，再触发摄像头检测
            # if camera_thread_active:
            #     time.sleep(POST_SWITCH_GRACE)
            #     camera_ok_flag = None
            #     check_camera_event.set()
            #     while check_camera_event.is_set():
            #         time.sleep(0.1)

            # # === 这里写一次切换后检测日志 ===
            # if camera_ok_flag:
            #     write_log("[系统] 切换分辨率后检测：摄像头出图正常")
            # else:
            #     write_log("[系统] 切换分辨率后检测：摄像头黑屏")

            #     # —— 容忍期后才真正检测 —— 
            #     camera_ok_flag = None
            #     check_camera_event.set()
            #     while check_camera_event.is_set() and switch_event.is_set():
            #         time.sleep(0.1)

            #     # 如果不是首轮且检测到黑屏，就暂停下一次切换
            #     if not first_cycle and camera_ok_flag is False:
            #         write_log("[系统] 检测到持续黑屏，暂停自动切换分辨率……")
            #         pause_event.clear()  # 进入“挂起”状态
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
                    write_log("[系统] 切换分辨率后检测：摄像头出图正常")
                else:
                    write_log("[系统] 切换分辨率后检测：摄像头黑屏")

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
win_w, win_h = 900, 900
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
monitor_frame.pack(pady=10, fill="both", expand=True)
monitor_label = Label(monitor_frame, text="显示器名称 (EDID):", font=("微软雅黑", 12))
monitor_label.pack()
monitor_listbox = Listbox(monitor_frame, width=50, height=5)
monitor_listbox.pack()
for name in monitor_names:
    monitor_listbox.insert(tk.END, name)
monitor_listbox.bind('<<ListboxSelect>>', on_monitor_select)

# 音频设备下拉框
audio_frame = Frame(root)
audio_frame.pack(pady=10)
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
    global _audio_rms_deque, _audio_last_silence_alert, _audio_last_stutter_alert, audio_monitor_active

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

            # 播放活动判定
            activity_window_start = ts - PLAYBACK_ACTIVITY_WINDOW_SEC
            recent_vals = [r for t, r in _audio_rms_deque if t >= activity_window_start]
            recent_max = max(recent_vals) if recent_vals else 0.0
            if recent_max > activity_threshold:
                last_activity_ts = ts
                playback_seen = True

            # 连续静音计算
            cont_silent_seconds = 0.0
            for i in range(len(_audio_rms_deque)-1, -1, -1):
                t_i, r_i = _audio_rms_deque[i]
                if r_i < AUDIO_SILENCE_RMS:
                    cont_silent_seconds += AUDIO_CHUNK_SEC
                else:
                    break

            # 调试日志
            debug_line = f"[音频-调试] t={time.strftime('%H:%M:%S', time.localtime(ts))} rms={rms:.6f} recent_max={recent_max:.6f} threshold={activity_threshold:.6e} playback_seen={playback_seen} cont_silent={cont_silent_seconds:.2f}s"
            try:
                root.after(0, lambda ln=debug_line: write_log(ln))
            except Exception:
                print(debug_line)

            # 无声音报警（仅在曾播放过后触发）
            # 新的：使用单一持续更新窗口显示静音时长（线程安全 via root.after）
            SILENCE_ALERT_SECONDS = AUDIO_SILENCE_DURATION if 'AUDIO_SILENCE_DURATION' in globals() else 5.0

            if playback_seen and (time.time() - start_time) > PLAYBACK_WARMUP_SEC:
                if cont_silent_seconds >= SILENCE_ALERT_SECONDS:
                    # 控制日志频率（避免每个循环都写）
                    if (_audio_last_silence_alert is None) or (ts - _audio_last_silence_alert) > 1.0:
                        _audio_last_silence_alert = ts
                        root.after(0, lambda s=cont_silent_seconds, d=device_name: write_log(f"[音频] 检测到持续静音 {int(s)}s，显示静音窗口，设备: {d}"))
                    # 在主线程创建或更新静音窗口
                    root.after(0, lambda s=cont_silent_seconds, d=device_name: show_silence_window(s, d))
                else:
                    # 小于阈值时若窗口存在则关闭（声音恢复）
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

# def audio_monitor_worker(device_name):
#     """
#     回调+队列版的音频监测线程（已改良）：
#       - 按设备名优先解析索引（精确->模糊），并在打开流前把“实际使用设备 index/name”写日志；
#       - 优先尝试 WASAPI loopback（若可用），否则回退到 Stereo Mix（录音设备）；
#       - 使用 callback+queue 读取块，主循环计算 RMS 做播放/静音/卡顿判定。
#     """
#     global _audio_rms_deque, _audio_last_silence_alert, _audio_last_stutter_alert, audio_monitor_active

#     audio_monitor_event.clear()
#     audio_monitor_active = True

#     # 函数内播放/活动状态初始化
#     playback_seen = False
#     last_activity_ts = 0.0
#     start_time = time.time()

#     chunk = int(AUDIO_SR * AUDIO_CHUNK_SEC)

#     # 内部：按名称查索引（优先精确匹配再模糊匹配）
#     def find_device_index_by_name_prefer_exact(name):
#         if not name:
#             return None, None
#         target = (name or "").strip().lower()
#         devs = sd.query_devices()
#         # 精确匹配
#         for i, d in enumerate(devs):
#             if (d.get('name') or '').strip().lower() == target:
#                 return i, d
#         # 子串匹配
#         for i, d in enumerate(devs):
#             if target in (d.get('name') or '').lower():
#                 return i, d
#         # 逐词匹配
#         keywords = target.split()
#         for i, d in enumerate(devs):
#             n = (d.get('name') or '').lower()
#             if all(k in n for k in keywords):
#                 return i, d
#         return None, None

#     # 先尝试按传入 device_name 解析索引（如果有）
#     try:
#         dev_idx, dev_info = find_device_index_by_name_prefer_exact(device_name)
#     except Exception:
#         dev_idx, dev_info = None, None

#     # ------ 探测能否用 WASAPI loopback ------
#     wasapi_available = True
#     try:
#         _ = sd.WasapiSettings(loopback=True)
#     except Exception:
#         wasapi_available = False

#     # 尝试以索引打开（blocking probe）
#     def try_open_blocking(dev, ch, use_wasapi):
#         try:
#             if isinstance(dev, str):
#                 dev_param = dev
#             else:
#                 dev_param = int(dev)
#             if use_wasapi:
#                 was = sd.WasapiSettings(loopback=True)
#                 s = sd.InputStream(device=dev_param, channels=ch,
#                                    samplerate=AUDIO_SR, blocksize=chunk,
#                                    dtype='float32', extra_settings=was)
#             else:
#                 s = sd.InputStream(device=dev_param, channels=ch,
#                                    samplerate=AUDIO_SR, blocksize=chunk,
#                                    dtype='float32')
#             s.start(); s.stop(); s.close()
#             return True, None
#         except Exception as e:
#             return False, e

#     tried = []
#     stream_device_index = None
#     stream_channels = 1
#     resolved_device_name = None  # 最终用于创建流时的 name 或 index

#     # 1) 如果通过 device_name 找到 dev_idx，优先尝试它（并尝试 2/1 通道降级）
#     if dev_idx is not None:
#         dev_max_ch = int(dev_info.get('max_output_channels', 1) or 1)
#         tried.append((dev_idx, dev_max_ch))
#         ok = False; err = None
#         if wasapi_available:
#             ok, err = try_open_blocking(dev_idx, dev_max_ch, True)
#             if not ok:
#                 for ch in (2, 1):
#                     if ch != dev_max_ch:
#                         tried.append((dev_idx, ch))
#                         ok, err = try_open_blocking(dev_idx, ch, True)
#                         if ok:
#                             break
#         else:
#             ok, err = try_open_blocking(dev_idx, dev_max_ch, False)
#             if not ok:
#                 for ch in (2, 1):
#                     if ch != dev_max_ch:
#                         tried.append((dev_idx, ch))
#                         ok, err = try_open_blocking(dev_idx, ch, False)
#                         if ok:
#                             break
#         if ok:
#             stream_device_index = dev_idx
#             stream_channels = dev_max_ch
#             resolved_device_name = dev_info.get('name')

#     # 2) 如果还没找到，扫描系统输出设备（优先非 HDMI/monitor）
#     if stream_device_index is None:
#         all_devs = sd.query_devices()
#         # 选出有输出通道的设备，按 priority 排序（避免优先选 HDMI）
#         candidates = []
#         for i, d in enumerate(all_devs):
#             if d.get('max_output_channels', 0) > 0:
#                 name = (d.get('name') or "").lower()
#                 # priority 0 is preferred (non-monitor), 1 is less preferred
#                 priority = 1 if ('monitor' in name or 'hdmi' in name or 'display' in name) else 0
#                 candidates.append((priority, i, d))
#         candidates.sort(key=lambda x: x[0])
#         for priority, i, d in candidates:
#             dev_max_ch = int(d.get('max_output_channels', 1) or 1)
#             tried.append((i, dev_max_ch))
#             ok, err = try_open_blocking(i, dev_max_ch, wasapi_available)
#             if ok:
#                 stream_device_index = i
#                 stream_channels = dev_max_ch
#                 resolved_device_name = d.get('name')
#                 break
#             for ch in (2, 1):
#                 if ch != dev_max_ch:
#                     tried.append((i, ch))
#                     ok, err = try_open_blocking(i, ch, wasapi_available)
#                     if ok:
#                         stream_device_index = i
#                         stream_channels = ch
#                         resolved_device_name = d.get('name')
#                         break
#             if stream_device_index is not None:
#                 break

#     # 3) 回退：尝试 Stereo Mix / 录音设备
#     def find_stereo_mix_device_local():
#         devs = sd.query_devices()
#         for ii, dd in enumerate(devs):
#             name = (dd.get('name') or "").lower()
#             if dd.get('max_input_channels', 0) > 0 and 'stereo' in name and 'mix' in name:
#                 return ii, dd
#         for ii, dd in enumerate(devs):
#             name = (dd.get('name') or "").lower()
#             if dd.get('max_input_channels', 0) > 0 and ('mix' in name or '立体声' in name or 'wave out' in name or 'stereo' in name):
#                 return ii, dd
#         return None, None

#     if stream_device_index is None:
#         sm_idx, sm_info = find_stereo_mix_device_local()
#         if sm_idx is not None:
#             stream_device_index = sm_idx
#             stream_channels = min(2, max(1, int(sm_info.get('max_input_channels', 1) or 1)))
#             resolved_device_name = sm_info.get('name')
#             log_event("音频", f"回退到录音设备: {resolved_device_name} (index={sm_idx})")
#         else:
#             _audio_alert_popup(
#                 "音频监测启动失败",
#                 (f"无法为设备 '{device_name}' 打开回环录音（尝试过: {tried}）。\n"
#                  "建议：启用 Stereo Mix、或安装 VB-Cable、或选择非 HDMI 的扬声器输出。")
#             )
#             audio_monitor_active = False
#             return

#     # 在打开流前再次按名称解析（如果用户指定了 device_name，优先按名称打开，避免索引漂移）
#     try:
#         if device_name:
#             idx_by_name, info_by_name = find_device_index_by_name_prefer_exact(device_name)
#             if idx_by_name is not None:
#                 # 使用解析到的索引/name（覆盖 stream_device_index），方便用户显式选择设备
#                 stream_device_index = idx_by_name
#                 stream_channels = int(info_by_name.get('max_output_channels', stream_channels) or stream_channels)
#                 resolved_device_name = info_by_name.get('name')
#     except Exception:
#         pass

#     # 最终用于创建流的 device 参数（用 name 比 index 更稳健）
#     device_for_stream = resolved_device_name if resolved_device_name is not None else stream_device_index

#     # 写出“实际使用的设备信息”日志（线程安全）
#     try:
#         dev_info_dbg = sd.query_devices()[stream_device_index]
#         dbg_msg = f"[音频] 使用监测设备: index={stream_device_index}, name='{dev_info_dbg.get('name')}', in:{dev_info_dbg.get('max_input_channels')} out:{dev_info_dbg.get('max_output_channels')}"
#         root.after(0, lambda m=dbg_msg: write_log(m))
#     except Exception:
#         try:
#             root.after(0, lambda m=f"[音频] 使用监测设备: {device_for_stream}": write_log(m))
#         except Exception:
#             pass

#     # ----- 创建回调流并 start() -----
#     q = queue.Queue(maxsize=30)

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

#     try:
#         if wasapi_available and (isinstance(device_for_stream, int) or ('hdmi' in (str(device_for_stream).lower()) or 'monitor' in (str(device_for_stream).lower()) or 'display' in (str(device_for_stream).lower()))):
#             was = sd.WasapiSettings(loopback=True)
#             stream_cb = sd.InputStream(device=device_for_stream,
#                                        channels=stream_channels,
#                                        samplerate=AUDIO_SR,
#                                        blocksize=chunk,
#                                        dtype='float32',
#                                        callback=callback,
#                                        extra_settings=was)
#         else:
#             stream_cb = sd.InputStream(device=device_for_stream,
#                                        channels=stream_channels,
#                                        samplerate=AUDIO_SR,
#                                        blocksize=chunk,
#                                        dtype='float32',
#                                        callback=callback)
#         stream_cb.start()
#         log_event("音频", f"音频回调流已启动: index={stream_device_index}, channels={stream_channels}, wasapi={wasapi_available}")
#     except Exception as e:
#         _audio_alert_popup("音频监测启动失败", f"创建回调流失败: {e}")
#         audio_monitor_active = False
#         return

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

#     if baseline_samples:
#         baseline_noise = float(np.median(baseline_samples))
#     else:
#         baseline_noise = 0.0

#     ABSOLUTE_MIN_ACTIVITY = 1e-6
#     ACTIVITY_FACTOR = 1.5
#     activity_threshold = max(baseline_noise * ACTIVITY_FACTOR, ABSOLUTE_MIN_ACTIVITY)

#     try:
#         root.after(0, lambda: write_log(f"[音频] 自校准: baseline_noise={baseline_noise:.6e}, activity_threshold={activity_threshold:.6e}"))
#     except Exception:
#         print(f"[音频] 自校准: baseline_noise={baseline_noise:.6e}, activity_threshold={activity_threshold:.6e}")
#     # ===== 校准结束 =====

#     # ===== 主循环：从队列取数据，计算 RMS 并执行静音/卡顿判定（替换现有重复/出错的实现） =====
#     try:
#         while not audio_monitor_event.is_set():
#             # 1) 获取数据块（回调队列），超时视为静音块
#             try:
#                 data_block = q.get(timeout=1.0)
#                 has_data = True
#             except queue.Empty:
#                 data_block = None
#                 has_data = False

#             # 2) 计算本块 RMS
#             if not has_data:
#                 rms = 0.0
#                 ts = time.time()
#             else:
#                 try:
#                     arr = np.array(data_block, dtype=np.float32)
#                     if arr.ndim > 1:
#                         rms = float(np.sqrt(np.mean(np.square(arr))))
#                     else:
#                         rms = float(np.sqrt(np.mean(np.square(arr))))
#                 except Exception:
#                     rms = 0.0
#                 ts = time.time()

#             # 3) 记录到滑动队列
#             _audio_rms_deque.append((ts, rms))
#             # 丢弃过旧项（保持窗口）
#             while _audio_rms_deque and (_audio_rms_deque[0][0] < ts - AUDIO_STUTTER_WINDOW_SEC - 1.0):
#                 _audio_rms_deque.popleft()

#             # ===== 播放活动判定（任意一块超过阈值则认为曾播放过） =====
#             activity_window_start = ts - PLAYBACK_ACTIVITY_WINDOW_SEC
#             recent_vals = [r for t, r in _audio_rms_deque if t >= activity_window_start]
#             recent_max = max(recent_vals) if recent_vals else 0.0
#             if recent_max > activity_threshold:
#                 last_activity_ts = ts
#                 playback_seen = True

#             # ===== 连续静音时长计算 =====
#             cont_silent_seconds = 0.0
#             for i in range(len(_audio_rms_deque)-1, -1, -1):
#                 t_i, r_i = _audio_rms_deque[i]
#                 if r_i < AUDIO_SILENCE_RMS:
#                     cont_silent_seconds += AUDIO_CHUNK_SEC
#                 else:
#                     break

#             # ===== 调试日志（线程安全地通过 root.after 写 GUI） =====
#             debug_line = f"[音频-调试] t={time.strftime('%H:%M:%S', time.localtime(ts))} rms={rms:.6f} recent_max={recent_max:.6f} threshold={activity_threshold:.6e} playback_seen={playback_seen} cont_silent={cont_silent_seconds:.2f}s"
#             try:
#                 root.after(0, lambda ln=debug_line: write_log(ln))
#             except Exception:
#                 # 如果 root 尚未可用，退回到控制台打印
#                 print(debug_line)

#             # ===== 只有当曾经播放过（playback_seen）且过了 warmup 时间才触发“无声音”报警 =====
#             if playback_seen and (time.time() - start_time) > PLAYBACK_WARMUP_SEC:
#                 if cont_silent_seconds >= AUDIO_SILENCE_DURATION and (ts - (_audio_last_silence_alert or 0)) > AUDIO_ALERT_COOLDOWN:
#                     _audio_last_silence_alert = ts
#                     _audio_alert_popup("外部显示器无声音", f"检测到持续静音 {cont_silent_seconds:.1f}s，设备: {device_name}")
#                     # 弹窗后复位 playback_seen，避免重复弹窗，直到下次检测到播放行为
#                     playback_seen = False

#             # ===== 卡顿判定（仅在曾经播放过时生效） =====
#             if playback_seen and (time.time() - start_time) > PLAYBACK_WARMUP_SEC:
#                 window_start_ts = ts - AUDIO_STUTTER_WINDOW_SEC
#                 vals = [r for t, r in _audio_rms_deque if t >= window_start_ts]
#                 if vals:
#                     low_count = sum(1 for v in vals if v < (AUDIO_SILENCE_RMS * 2))
#                     drop_pct = low_count / len(vals)
#                     if drop_pct >= AUDIO_STUTTER_DROP_PCT and (ts - (_audio_last_stutter_alert or 0)) > AUDIO_ALERT_COOLDOWN:
#                         _audio_last_stutter_alert = ts
#                         _audio_alert_popup("外部显示器声音卡顿", f"检测到短期内声音中断/抖动（{drop_pct*100:.0f}%）设备: {device_name}")

#             # 小睡以降低 CPU 占用（队列 get 已阻塞，通常不会频繁到这里）
#             time.sleep(0.005)

#     finally:
#         try:
#             stream_cb.stop()
#             stream_cb.close()
#         except Exception:
#             pass
#         audio_monitor_active = False
#         log_event("系统", f"音频监测线程已结束: {device_name}")


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
