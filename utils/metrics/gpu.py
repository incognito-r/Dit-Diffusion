from pynvml import *

def init_nvml():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)  # GPU 0
    return handle

def get_gpu_memory(handle):
    info = nvmlDeviceGetMemoryInfo(handle)
    used = info.used / (1024 ** 2)  # MiB
    total = info.total / (1024 ** 2)  # MiB
    return used, total

def gpu_info(handle, threshold_mb=11900):
    used, total = get_gpu_memory(handle)
    if used >= threshold_mb:
       return f"🚨GPU usage:{used:.0f} > {threshold_mb} Mib)"
   
    return f"{used:.0f} / {total:.0f} MiB"