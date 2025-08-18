# camera_monitor.py
import cv2
import numpy as np
import time
import threading
from collections import deque
from PIL import Image, ImageDraw, ImageFont
from pygrabber.dshow_graph import FilterGraph
from skimage.metrics import structural_similarity as ssim
from logger import write_log, log_event
from config import FPS, RECORD_DURATION, ALERT_DURATION, BLACK_ALERT_THRESHOLD, BLACK_ALERT_PCT, BLACK_ALERT_LB, FONT_PATH, VIDEO_DIR
from audio_utils import play_audio, stop_audio
import os
import state  # 使用 state.xxx 访问共享状态（事件、flag 等）

# GUI root（兼容接口），但 OpenCV 弹窗作为主报警方式，不依赖 root
root = None

# ===== 工具函数 =====
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
        try:
            os.makedirs(VIDEO_DIR, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = f"{VIDEO_DIR}/{tag}_{ts}.avi"
            h, w = buf[0].shape[:2]
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"XVID"), FPS, (w,h))
            for f in buf:
                vw.write(f)
            vw.release()
            log_event(tag, f"视频已保存: {path}")
        except Exception as e:
            log_event("异常", f"save_video 失败: {e}")
    threading.Thread(target=_save, args=(frames.copy(),), daemon=True).start()

def open_camera(idx=1):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        log_event("错误", f"无法打开摄像头 {idx}")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    return cap

def is_black(frame, pct=BLACK_ALERT_PCT, lb=BLACK_ALERT_LB):
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

def detect_jitter_orb(prev_gray, cur_gray, disp_th=1.0, pct_th=0.05):
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(cur_gray,  None)
    if des1 is None or des2 is None:
        return False
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 10:
        return False
    matches = sorted(matches, key=lambda x: x.distance)[:50]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    M, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3)
    if M is None:
        return False
    dx, dy = M[0,2], M[1,2]
    inliers = int(mask.flatten().sum()) if mask is not None else 0
    if inliers == 0 or (inliers / len(matches)) < pct_th:
        return False
    return np.hypot(dx, dy) > disp_th

def detect_artifacts(frame, bs=64, var_th=1000, edge_th=0.25):
    g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5,5), 0)
    sx, sy = np.gradient(g.astype(float))
    em = np.hypot(sx, sy)
    h, w = g.shape
    cnt = tot = 0
    for y in range(0, h, bs):
        for x in range(0, w, bs):
            blk = g[y:y+bs, x:x+bs]
            if blk.size < bs * bs:
                continue
            if np.var(blk) < var_th and np.mean(em[y:y+bs, x:x+bs]) < edge_th:
                cnt += 1
            tot += 1
    return (cnt / tot) if tot else 0.0

# ===== 主检测循环 =====
def show_camera_feed():
    """
    摄像头检测主循环（使用 OpenCV 小窗口作为报警弹窗）
    """
    cap = open_camera(1)
    if not cap:
        write_log("错误", "无法打开摄像头1，无法进入 show_camera_feed")
        return

    window = get_camera_name(0)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 640, 480)

    buf = deque(maxlen=FPS * RECORD_DURATION)
    prev = None
    black_start = 0
    last_alarm = 0
    frame_idx = 0
    jitter_cnt = 0
    art_cnt = 0
    last_art = 0
    alert_start = 0
    alert_img = None

    # 闪屏参数
    _flicker_count = 0
    last_flicker_time = 0.0
    FLICKER_BRIGHTNESS_TH = 60.0
    COMPARE_PROP_TH = 0.10
    FLICKER_CONFIRM_FRAMES = 6
    FLICKER_COOLDOWN = 2.0

    # 抖动参数
    jitter_start = None
    JITTER_ALERT_DURATION = 2.0
    JITTER_DISP_TH = 0.5
    JITTER_PCT_TH = 0.01

    # 播放提示音
    play_audio()
    log_event("系统", "摄像头检测线程已启动")

    try:
        while True:
            # 检查窗口是否被用户关闭
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    log_event("系统", "用户关闭了摄像头监控窗口，退出 show_camera_feed")
                    state.switch_event.clear()
                    break
            except Exception as e:
                log_event("异常", f"检测窗口属性读取异常: {e}")

            # 响应外部检测请求（切分辨率线程会 set 这个 event）
            if state.check_camera_event.is_set():
                # 读取一帧用于检测（避免使用旧缓冲帧）
                ret_chk, frame_chk = cap.read()
                if ret_chk and not is_black(frame_chk, pct=BLACK_ALERT_PCT, lb=BLACK_ALERT_LB):
                    state.camera_ok_flag = True
                else:
                    state.camera_ok_flag = False
                state.check_camera_event.clear()
                black_start = 0

            # 读取帧
            ret, frame = cap.read()
            if not ret or frame is None or (hasattr(frame, "size") and frame.size == 0):
                log_event("异常", "读取到空帧，跳过本次循环")
                time.sleep(0.01)
                continue

            # 缓存帧
            buf.append(frame.copy())

            # 黑屏检测
            if is_black(frame, pct=BLACK_ALERT_PCT, lb=BLACK_ALERT_LB):
                if black_start == 0:
                    black_start = time.time()
                if time.time() - black_start > BLACK_ALERT_THRESHOLD and time.time() - last_alarm > ALERT_DURATION:
                    if state.pause_event.is_set():
                        state.pause_event.clear()
                        write_log(f"[系统] 检测到持续黑屏 {time.time() - black_start:.1f}s，已暂停自动切换分辨率")
                        log_event("黑屏", f"持续 {time.time() - black_start:.1f}s")
                    last_alarm = time.time()
                    save_video(list(buf), "black_screen")
                    # 构造警告图像并显示（OpenCV 窗口）
                    alert_img = draw_text(np.zeros((200, 500, 3), np.uint8), f"黑屏 {time.time() - black_start:.1f}s", (50, 80))
                    alert_start = time.time()
                    stop_audio()
                    log_event("黑屏", "显示 OpenCV 警告窗口")
            else:
                if black_start:
                    log_event("系统", f"黑屏恢复，持续 {time.time() - black_start:.1f}s")
                # 若启用了摄像头检测且处于暂停，则恢复
                if state.camera_thread_active and not state.pause_event.is_set():
                    state.pause_event.set()
                    write_log("[系统] 摄像头恢复正常，自动切换分辨率已恢复")
                    black_start = 0
                    play_audio()

            # 闪屏检测（基于亮度突变或帧差）
            if prev is not None and hasattr(prev, "size") and prev.size > 0:
                try:
                    gray_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                    gray_cur  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    mean_prev = float(np.mean(gray_prev))
                    mean_cur  = float(np.mean(gray_cur))
                    brightness_diff = abs(mean_cur - mean_prev)
                    diff_val = compare(prev, frame)
                    diff_prop = diff_val / 255.0
                except Exception as e:
                    brightness_diff = 0.0
                    diff_prop = 0.0
                    log_event("异常", f"闪屏/差异计算失败: {e}")
                else:
                    is_flicker_frame = (brightness_diff >= FLICKER_BRIGHTNESS_TH) or (diff_prop >= COMPARE_PROP_TH)
                    if is_flicker_frame:
                        _flicker_count += 1
                    else:
                        _flicker_count = 0

                    now = time.time()
                    if _flicker_count >= FLICKER_CONFIRM_FRAMES and (now - last_flicker_time) >= FLICKER_COOLDOWN and (now - last_alarm) > ALERT_DURATION:
                        if not is_black(prev) and not is_black(frame):
                            last_flicker_time = now
                            last_alarm = now
                            _flicker_count = 0
                            save_video(list(buf), "flicker")
                            alert_img = draw_text(np.zeros((200, 500, 3), np.uint8), "检测到闪屏！", (50, 80), color=(0, 255, 255))
                            alert_start = time.time()
                            log_event("闪屏", f"检测到闪屏 brightness_diff={brightness_diff:.1f} diff_prop={diff_prop:.4f}")

            # 抖动检测（持续时间判断）
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
                    if jitter_start is None:
                        jitter_start = now
                    if jitter_start is not None and (now - jitter_start) >= JITTER_ALERT_DURATION and (now - last_alarm) > ALERT_DURATION:
                        last_alarm = now
                        save_video(list(buf), "jitter")
                        alert_img = draw_text(np.zeros((200, 500, 3), np.uint8), "检测到抖动！", (50, 80), color=(255, 0, 255))
                        alert_start = time.time()
                        log_event("抖动", f"检测到持续抖屏，持续 {(now - jitter_start):.1f}s")
                        jitter_start = None
                else:
                    jitter_start = None

            # 预处理（占位）
            proc = preprocess(frame)

            # ORB 连续计数的报警（兼容旧逻辑）
            if jitter_cnt >= 1 and time.time() - last_alarm > ALERT_DURATION:
                last_alarm = time.time()
                save_video(list(buf), "jitter_orb")
                alert_img = draw_text(np.zeros((200, 500, 3), np.uint8), "检测到抖动！", (50,80), color=(255,0,255))
                alert_start = time.time()
                log_event("抖动", "ORB 检测到抖动 平移>0.5px")
                jitter_cnt = 0

            # 更新 prev
            prev = frame.copy()

            # 花屏/纹理异常检测（间隔检测）
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
                    last_alarm = time.time()
                    save_video(list(buf), "artifact")
                    alert_img = draw_text(np.zeros((200, 500, 3), np.uint8), "检测到花屏！", (50, 80), color=(0, 255, 255))
                    alert_start = time.time()
                    log_event("花屏", f"SSIM={sim_val:.2f} ratio={ratio:.2f}")
                    art_cnt = 0
                    last_art = frame_idx

                # 继续 ORB 计数
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

            # 如果有 alert_img 且在弹窗持续期内，显示 OpenCV 的警告窗口
            if alert_start:
                if time.time() - alert_start <= ALERT_DURATION:
                    try:
                        cv2.imshow("屏幕警告", alert_img)
                    except Exception:
                        pass
                else:
                    try:
                        cv2.destroyWindow("屏幕警告")
                    except Exception:
                        pass
                    alert_start = 0
                    alert_img = None

            # 显示主摄像头画面
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


# 兼容外部调用接口（保留 root 参数）
def start_camera_monitor(r):
    """
    兼容接口：外部传入 Tk root 后启动摄像头监控（当前实现不依赖 root）。
    建议以线程方式调用，例如：
        threading.Thread(target=lambda: start_camera_monitor(root), daemon=True).start()
    """
    global root
    root = r
    show_camera_feed()
