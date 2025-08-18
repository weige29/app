# display_utils.py
import wmi
import win32api
import win32con
from typing import List, Tuple
from logger import write_log, log_event

def list_display_monitors() -> List[str]:
    """通过 WMI 获取 EDID 名称列表"""
    w = wmi.WMI(namespace='wmi')
    monitors = w.WmiMonitorID()
    monitor_names = []
    for monitor in monitors:
        if monitor.UserFriendlyName:
            name = "".join([chr(c) for c in monitor.UserFriendlyName if c > 0])
            monitor_names.append(name)
    return monitor_names

def list_display_devices() -> List[Tuple[str,str]]:
    """枚举显示设备，返回 (DeviceName, DeviceString) 列表"""
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

def get_edid_device_mapping():
    """构建 EDID->DeviceName 的映射（尝试模糊匹配）"""
    monitor_names = list_display_monitors()
    devices_list = list_display_devices()
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

def get_display_modes(display_device):
    """返回 (width, height, refresh) 列表"""
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

def change_display_resolution(device_name, width, height, refresh_rate):
    """切换分辨率"""
    try:
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
    except Exception as e:
        log_event("异常", f"change_display_resolution 异常: {e}")
        return None
