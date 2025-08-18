# audio_utils.py
import os
import pygame
import sounddevice as sd
from logger import log_event
from config import AUDIO_PATH

def find_monitor_audio():
    device_list = sd.query_devices()
    external_devices = []
    numeric_devices = []
    monitor_keywords = ["HDMI", "Display", "Monitor", "External", "DP", "英特尔(R)", "显示器音频", "NVIDIA", "AMD", "TV"]

    for device in device_list:
        if device['max_output_channels'] > 0 and any(keyword in device['name'] for keyword in monitor_keywords):
            if "Output" in device['name']:
                continue
            if any(char.isdigit() for char in device['name']):
                numeric_devices.append(device['name'])
            else:
                external_devices.append(device['name'])

    external_devices = numeric_devices + external_devices
    return external_devices

def play_audio():
    if os.path.exists(AUDIO_PATH):
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2)
            pygame.mixer.music.load(AUDIO_PATH)
            pygame.mixer.music.play(loops=-1)
            log_event("系统", "音频开始播放")
        except Exception as e:
            log_event("异常", f"play_audio 错误: {e}")

def stop_audio():
    try:
        if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            log_event("系统", "音频已停止")
    except Exception as e:
        log_event("异常", f"stop_audio 错误: {e}")
