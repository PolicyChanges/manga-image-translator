import os
import gc
import sys
import torch
import pynvml
import time

import cv2
import math
import numpy as np
import queue
import threading

#   from basicsr.utils.download_util import load_file_from_url
from torch.nn import functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from colorama import Fore, Back, Style, init
# Initialize colorama and auto-reset colors after every print statement
init(autoreset=True)

def dump_gpu_diagnostic(virtual_device_idx: int = 0):
    """Queries NVIDIA driver via NVML to log exact process memory allocations."""
    print("\n" + "="*20 + " CUDA OOM DIAGNOSTIC DUMP " + "="*20)
    
    try:
        pynvml.nvmlInit()
        
        # 1. Map PyTorch virtual index to physical index using environment variables
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible.strip():
            physical_indices = [int(x.strip()) for x in cuda_visible.split(",") if x.strip().isdigit()]
            physical_idx = physical_indices[virtual_device_idx]
        else:
            physical_idx = virtual_device_idx

        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_idx)
        device_name = pynvml.nvmlDeviceGetName(handle)
        print(f"Device: cuda:{virtual_device_idx} -> Physical GPU ID {physical_idx} ({device_name})")

        # 2. Extract hardware memory status
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"Hardware State -> Total: {mem_info.total / 1024**2:.1f}MB | Used: {mem_info.used / 1024**2:.1f}MB | Free: {mem_info.free / 1024**2:.1f}MB")

        # 3. Scan and list all active processes on this specific GPU
        print("\nRunning GPU Processes:")
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        if not processes:
            print("  No active compute processes found.")
        for proc in processes:
            try:
                # Get the process name from its PID
                proc_name = pynvml.nvmlSystemGetProcessName(proc.pid)
            except pynvml.NVMLError:
                proc_name = "Unknown Process"
            
            # NVML reports process memory in bytes; convert to MB
            proc_mem = proc.usedGpuMemory / (1024**2) if proc.usedGpuMemory else 0
            is_self = " (Current Script)" if proc.pid == os.getpid() else ""
            print(f"  - PID: {proc.pid:<6} | Memory: {proc_mem:>7.1f} MB | Name: {proc_name}{is_self}")

        pynvml.nvmlShutdown()

    except pynvml.NVMLError as err:
        print(f"Failed to query NVML: {err}")
    
    # 4. Dump PyTorch's internal allocator state
    print("\nPyTorch Allocator State:")
    print(torch.cuda.memory_summary(device=virtual_device_idx, abbreviated=True))
    print("="*66 + "\n")

"""
        min_free_gb=1.5
        recurse_idx=0
        poll_interval=.5
        import time
        while True:
            #if recurse_idx > recurse_count:
            #    dump_gpu_diagnostic()
            #    exit(1)
            # Get free and total memory in bytes
            free_bytes, total_bytes = torch.cuda.mem_get_info('cuda:0')
            free_gb = free_bytes / (1024 ** 3)
            
            if free_gb >= min_free_gb:
                print(f"Success: {free_gb:.2f} GB free. Proceeding to load model.")
                break
            else:  
                print(f"LOW MEMEORY WARNING: Low Memory: Only {free_gb:.2f} GB...Try {recurse_idx}... Waiting {poll_interval}s...")
            #gc.collect()
            torch.cuda.empty_cache()
            time.sleep(poll_interval) 
            recurse_idx=recurse_idx+1  
"""
async def wait_for_gpu_memory(device_idx: int, recurse_idx: int, min_free_gb: float = 4.0, poll_interval: int = 1, recurse_count: int = 5):
    return
    """Blocks execution until the target GPU has enough free memory."""
    #print(Fore.RED + "WARNING!!!!!!!!: Checking memory on cuda:") #{device_idx}...")
    
    while True:
        #if recurse_idx > recurse_count:
        #    dump_gpu_diagnostic()
        #    exit(1)
        # Get free and total memory in bytes
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
        free_gb = free_bytes / (1024 ** 3)
        
        if free_gb >= min_free_gb:
            print(f"Success: {free_gb:.2f} GB free. Proceeding to load model.")
            break
        else:  
            print(f"LOW MEMEORY WARNING: Low Memory: Only {free_gb:.2f} GB...Try {recurse_idx}... Waiting {poll_interval}s...")
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(poll_interval) 
            recurse_idx=recurse_idx+1    
