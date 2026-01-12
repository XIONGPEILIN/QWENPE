#!/usr/bin/env python
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
import multiprocessing as mp
from queue import Empty

# Configuration
SCALES = [1.0, 2.0, 4.0, 6.0]
CKPT_NAME = "2011-ste-28000"
BASE_COMPARE_DIR = Path("compare")
JSON_PATH = "dataset_qwen_pe_top1000.json"
GT_BASE_DIR = "/export/ssd2/xiong-p/qwenpe/pico-banana-400k-subject_driven/openimages"
EVAL_SCRIPT = "evaluate_metrics.py"

# Use fewer cards to avoid OOM
AVAILABLE_GPUS = [0] 

def prepare_flat_directory(cfg_int, src_dir):
    """Prepares a flat image directory from nested results."""
    if not src_dir.exists():
        print(f"[Skip] Source directory not found: {src_dir}")
        return None
        
    flat_dir = Path(f"temp_flat_results_cfg{cfg_int}")
    if flat_dir.exists():
        shutil.rmtree(flat_dir)
    flat_dir.mkdir(parents=True)
    
    count = 0
    for sample_folder in src_dir.iterdir():
        if not sample_folder.is_dir(): continue
        json_file = sample_folder / "sample.json"
        output_img = sample_folder / "output.png"
        
        if json_file.exists() and output_img.exists():
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                orig_rel_path = data['edit_image'][0]
                filename = os.path.basename(orig_rel_path)
                shutil.copy(output_img, flat_dir / filename)
                count += 1
            except Exception as e:
                print(f"Error processing {sample_folder}: {e}")
    
    print(f"[CFG {cfg_int}] Collected {count} images into {flat_dir}")
    return flat_dir if count > 0 else None

def eval_worker(gpu_queue, task_queue):
    """Picks up a task and runs it on an available GPU."""
    while True:
        try:
            task = task_queue.get(timeout=1)
        except Empty:
            break

        cfg, flat_dir = task
        gpu_id = gpu_queue.get()
        
        print(f"\n>>> [GPU {gpu_id}] Starting evaluation for CFG {cfg}...")
        try:
            output_csv = f"evaluation_results_cfg{int(cfg)}.csv"
            # Using current python executable
            cmd = [
                sys.executable, EVAL_SCRIPT,
                "--json_path", JSON_PATH,
                "--pred_dir", str(flat_dir),
                "--gt_base_dir", GT_BASE_DIR,
                "--output_csv", output_csv,
            ]
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            # Execute and wait
            subprocess.run(cmd, env=env, check=True)
            print(f"<<< [GPU {gpu_id}] Finished evaluation for CFG {cfg}")
        except Exception as e:
            print(f"!!! [GPU {gpu_id}] Error evaluating CFG {cfg}: {e}")
        finally:
            # Release GPU back to queue
            gpu_queue.put(gpu_id)

def main():
    # Ensure EVAL_SCRIPT exists
    if not os.path.exists(EVAL_SCRIPT):
        print(f"Error: {EVAL_SCRIPT} not found.")
        return

    # 1. Sequential data preparation (fast CPU task)
    tasks = []
    print("Preparing image directories for evaluation...")
    for cfg in SCALES:
        cfg_int = int(cfg)
        src_dir = BASE_COMPARE_DIR / f"{CKPT_NAME}_cfg{cfg_int}"
        flat_dir = prepare_flat_directory(cfg_int, src_dir)
        if flat_dir:
            tasks.append((cfg, flat_dir))
    
    if not tasks:
        print("No valid evaluation tasks found (check 'compare' directory).")
        return

    # 2. Parallel processing across 8 GPUs
    mp.set_start_method('spawn', force=True)
    
    task_queue = mp.Queue()
    for t in tasks:
        task_queue.put(t)
        
    gpu_queue = mp.Queue()
    for g in AVAILABLE_GPUS:
        gpu_queue.put(g)
        
    # Start as many workers as needed (up to number of GPUs or tasks)
    num_workers = min(len(tasks), len(AVAILABLE_GPUS))
    print(f"Spawning {num_workers} parallel evaluation workers on GPUs {AVAILABLE_GPUS[:num_workers]}...")
    
    workers = []
    for _ in range(num_workers):
        p = mp.Process(target=eval_worker, args=(gpu_queue, task_queue))
        p.start()
        workers.append(p)
        
    for p in workers:
        p.join()
        
    print("\nAll CFG evaluations completed successfully.")

if __name__ == "__main__":
    main()