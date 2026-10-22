from typing import List
from PIL import Image

from .common import CommonUpscaler, OfflineUpscaler
from .waifu2x import Waifu2xUpscaler
from .esrgan import ESRGANUpscaler
from .esrgan_pytorch import ESRGANUpscalerPytorch
from .real_esrgan_pytorch import RealESRGANUpscalerPytorch 
from ..config import Upscaler

UPSCALERS = {
    Upscaler.waifu2x: Waifu2xUpscaler,
    Upscaler.esrgan: ESRGANUpscaler,
    Upscaler.upscler4xultrasharp: ESRGANUpscalerPytorch,
}
upscaler_cache = {}

def get_upscaler(key: Upscaler, *args, **kwargs) -> CommonUpscaler:
    if key not in UPSCALERS:
        raise ValueError(f'Could not find upscaler for: "{key}". Choose from the following: %s' % ','.join(UPSCALERS))
    if not upscaler_cache.get(key):
        upscaler = UPSCALERS[key]
        upscaler_cache[key] = upscaler(*args, **kwargs)
    return upscaler_cache[key]

async def prepare(upscaler_key: Upscaler):
    upscaler = get_upscaler(upscaler_key)
    if isinstance(upscaler, OfflineUpscaler):
        await upscaler.download()

async def dispatch(upscaler_key: Upscaler, image_batch: List[Image.Image], upscale_ratio: int, device: str = 'cpu') -> List[Image.Image]:
    if upscale_ratio == 1:
        return image_batch
    upscaler = get_upscaler(upscaler_key)
    import torch
    '''import time
    import gc
    recurse_idx=0
    poll_interval=1
    min_free_gb=6.0
    if isinstance(upscaler, OfflineUpscaler):
        while True:
            #if recurse_idx > recurse_count:
            #    dump_gpu_diagnostic()
            #    exit(1)
            # Get free and total memory in bytes
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            free_gb = free_bytes / (1024 ** 3)
            if recurse_idx > 6:
                min_free_gb=min_free_gb-1.0
            if free_gb >= min_free_gb:
                print(f"Success: {free_gb:.2f} GB free. Proceeding to load model.")
                break
            else:  
                print(f"LOW MEMEORY WARNING: Low Memory: Only {free_gb:.2f} GB...Try {recurse_idx}... Waiting {poll_interval}s...")
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(poll_interval) 
                recurse_idx=recurse_idx+1    '''
    torch.cuda.empty_cache()    
    await upscaler.load(device)
    return await upscaler.upscale(image_batch, upscale_ratio)

async def unload(upscaler_key: Upscaler):
    upscaler_cache.pop(upscaler_key, None)
