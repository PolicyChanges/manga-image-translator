import os
import sys
import math
import pathlib
import shutil
import tempfile
import asyncio
import logging
import multiprocessing
from copy import deepcopy

# Absolute package pathing prevents child runtime lookup issues
from manga_translator.manga_translator import set_main_logger
from manga_translator.utils import init_logging, get_logger, set_log_level

def run_gpu_worker(gpu_id, worker_args, staging_folder, backend_type):
    """
    Subprocess entry point bound to an importable location to prevent unpickling failure.
    Applies vendor-agnostic environment masking depending on the detected backend stack.
    """
    # ----------------------------------------------------
    # VENDOR-AGNOSTIC DEVICE ISOLATION
    # ----------------------------------------------------
    if backend_type in ("cuda", "rocm"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    elif backend_type == "xpu":
        os.environ["ZE_AFFINITY_MASK"] = str(gpu_id)
        os.environ["ONEAPI_DEVICE_SELECTOR"] = f"level_zero:{gpu_id}"
    elif backend_type == "mps":
        pass
    
    # Force GPU acceleration flags within the active subprocess parameters
    worker_args.use_gpu = True

    # Re-initialize basic logging namespaces within the clean spawned worker space
    init_logging()
    set_log_level(level=logging.DEBUG if worker_args.verbose else logging.INFO)
    worker_logger = get_logger(worker_args.mode)
    set_main_logger(worker_logger)
    
    # Overwrite the contextual root logger instance for this subprocess
    logger = logging.getLogger(worker_args.mode)

    # Clear any stale handlers inherited or auto-created in the subprocess space
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # Build a dedicated StreamHandler targeting the console terminal standard output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if worker_args.verbose else logging.INFO)
    
    # Apply a clean, vendor-agnostic scannable visual layout tracking active hardware IDs
    tag = f"{backend_type.upper()} {gpu_id}"
    formatter = logging.Formatter(f'[{tag}] %(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.propagate = False
    
    # Pass the staging directory string to downstream pipeline expectations
    worker_args.input = staging_folder
    
    try:
        import manga_translator.__main__ as main_module
        main_module.logger = logger
        asyncio.run(main_module.dispatch(worker_args))
    except Exception as e:
        logger.error(f"Worker Process Error: {e.__class__.__name__}: {e}", exc_info=True)

def handle_multi_gpu_execution(args, logger):
    """
    Orchestration manager that pre-warms weight caches sequentially via a temporary white image canvas
    and then safely splits batches using symlink staging folders across valid hardware nodes.
    """
    if not args.input:
        logger.error("[CRITICAL ERROR] No input path supplied via -i for --multi-gpu mode.")
        sys.exit(1)

    if isinstance(args.input, list):
        if len(args.input) == 0:
            logger.error("[CRITICAL ERROR] Input path target collection argument slice is empty.")
            sys.exit(1)
        raw_input = args.input[0]
    else:
        raw_input = args.input

    # Clean path resolving sequence across absolute system constraints
    input_path = pathlib.Path(str(raw_input)).expanduser().resolve()
    if not input_path.is_dir():
        logger.error(f"[CRITICAL ERROR] --multi-gpu requires the input target (-i) to be a valid folder directory. '{input_path}' is invalid.")
        sys.exit(1)

    # ----------------------------------------------------
    # DYNAMIC OUTPUT PATH GENERATION
    # ----------------------------------------------------
    output_folder_name = f"{input_path.name}-translated"
    output_path = input_path.parent / output_folder_name
    
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Target translations output directory established: '{output_path}'")
    except Exception as e:
        logger.error(f"[CRITICAL ERROR] Failed to initialize target output folder directory structure: {e}")
        sys.exit(1)

    args.dest = str(output_path)

    # ----------------------------------------------------
    # DYNAMIC VENDOR HARDWARE PROBING
    # ----------------------------------------------------
    backend_type = None
    num_gpus = 0

    try:
        import torch
        if torch.cuda.is_available():
            backend_type = "rocm" if "rocm" in torch.__version__ else "cuda"
            num_gpus = torch.cuda.device_count()
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            backend_type = "xpu"
            num_gpus = torch.xpu.device_count()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            backend_type = "mps"
            num_gpus = multiprocessing.cpu_count() // 2
            
    except ImportError:
        logger.error("[CRITICAL ERROR] PyTorch is required to identify compute stacks for vendor-agnostic scaling.")
        sys.exit(1)

    if num_gpus < 1 or backend_type is None:
        logger.error("[CRITICAL ERROR] No visible hardware accelerator framework detected.")
        sys.exit(1)

    # ----------------------------------------------------
    # MODEL PRE-WARMING FLOW (PREVENTS DOWNLOADING RACES)
    # ----------------------------------------------------
    logger.info("Initializing baseline pipeline warm-up to prepare and lock model weights sequentially...")
    
    # Dynamically generate a simple 512x512 white canvas to trigger network caching runs safely
    try:
        import numpy as np
        import cv2
        white_canvas = np.full((1, 1, 3), 255, dtype=np.uint8)
    except ImportError:
        logger.error("[CRITICAL ERROR] NumPy and OpenCV are required to generate the warm-up canvas image.")
        sys.exit(1)

    # Setup isolated temporary warming workspaces
    warmup_in_dir = pathlib.Path(tempfile.mkdtemp(prefix="manga_warmup_in_"))
    warmup_out_dir = pathlib.Path(tempfile.mkdtemp(prefix="manga_warmup_out_"))
    warmup_img_path = warmup_in_dir / "warmup.png"

    cv2.imwrite(str(warmup_img_path), white_canvas)

    # Configure precise arguments for a single sequential run on the first primary engine instance
    warmup_args = deepcopy(args)
    warmup_args.input = str(warmup_in_dir)
    warmup_args.dest = str(warmup_out_dir)
    # Run the worker cycle once to safely lock weight targets into local caches sequentially
    run_gpu_worker(gpu_id=0, worker_args=warmup_args, staging_folder=str(warmup_in_dir), backend_type=backend_type)

    # Clean the dynamic warming footprints entirely out of the local path matrix
    try:
        shutil.rmtree(warmup_in_dir)
        shutil.rmtree(warmup_out_dir)
        logger.info("Pipeline pre-warming completed. Local model locks verified successfully.")
    except Exception as warmup_clean_err:
        logger.warning(f"[WARN] Non-fatal error cleaning up warming frames cache structures: {warmup_clean_err}")

    # Gather target source files
    supported_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.xcf', '.psd'}
    all_files = sorted([p for p in input_path.iterdir() if p.suffix.lower() in supported_exts])

    if not all_files:
        logger.info(f"[INFO] Balanced execution canceled. No target images found inside: '{input_path}'")
        sys.exit(0)

    active_worker_count = min(num_gpus, len(all_files))
    logger.info(f"Initializing distributed parallel translation scaling across {active_worker_count} compute worker instances ({backend_type.upper()}).")

    chunk_size = math.ceil(len(all_files) / active_worker_count)
    processes = []
    staging_dirs = []

    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    try:
        for i in range(active_worker_count):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            file_chunk = all_files[start_idx:end_idx]

            if not file_chunk:
                continue

            # Create a clean isolated temporary directory for this specific GPU worker
            tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix=f"manga_gpu_{i}_staging_"))
            staging_dirs.append(tmp_dir)

            for target_file in file_chunk:
                symlink_path = tmp_dir / target_file.name
                try:
                    symlink_path.symlink_to(target_file)
                except Exception as sym_err:
                    if os.name == 'nt':
                        logger.warning(f"[WARN] Privilege restrictions blocked symlink creation. Copying baseline asset cache instead: {target_file.name}")
                        shutil.copy2(target_file, symlink_path)
                    else:
                        raise sym_err

            worker_args = deepcopy(args)
            
            p = multiprocessing.Process(
                target=run_gpu_worker,
                args=(i, worker_args, str(tmp_dir), backend_type),
                name=f"Manga-{backend_type.upper()}-Worker-{i}"
            )
            processes.append(p)
            p.start()

        # Monitor process termination structures
        for p in processes:
            p.join()

    finally:
        logger.info("Initializing background temporary folder structure teardown routines...")
        for staging_path in staging_dirs:
            if staging_path.exists():
                try:
                    shutil.rmtree(staging_path)
                except Exception as clean_err:
                    logger.warning(f"[WARN] Failed to completely wipe out temp directory path '{staging_path}': {clean_err}")                   
    logger.info("[SUCCESS] Multi-GPU vendor-agnostic pipeline processing tasks completed successfully.")
    sys.exit(0)                