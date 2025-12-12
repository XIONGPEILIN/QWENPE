import torch
import time
import os

def run_gpu_test():
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your GPU installation.")
        return

    print(f"CUDA is available! Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Set the visible device (optional, but good for testing a specific GPU)
    # If the user wants to test on a specific GPU, they can uncomment and change this
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1" # For example, to test GPU 1

    try:
        # Create a large tensor on the GPU
        print("Attempting to allocate a large tensor on GPU...")
        # Allocate 4GB of memory on GPU
        # Adjust size based on available memory and desired load
        tensor_size_gb = 4
        # Need to be careful here, as some GPUs might have little free memory
        # If the target GPU is 100% busy with 1MiB memory, allocating 4GB might fail.
        # Let's try a smaller tensor first to see if it even works.
        # Let's try 1GB, or even 500MB if 1GB fails. Start with a size that is likely to fit.
        # Max GPU memory for RTX PRO 6000 is ~97GB, A100 is 80GB.
        # 1GB should be fine if there's any memory left.
        
        # If the GPU shows 1MiB usage for a 100% utilized GPU, that's very suspicious.
        # It means the GPU is busy, but not allocating much memory via PyTorch
        # or the reporting is broken. Let's try a modest allocation.
        
        # Let's try to allocate 1GB on each available GPU one by one if possible.
        # Or just pick GPU 0 by default, and if it's already 100% utilized,
        # the test might fail.
        
        # The user's goal is to test if code can run on a 100% utilized GPU.
        # This means, I should try to run *something* on one of the 100% utilized GPUs.
        # From the nvidia-smi output, GPU 1 and 2 are 100% utilized, but with 1MiB memory.
        # Let's target GPU 1 (index 1) for this test, assuming it's the target.
        # If CUDA_VISIBLE_DEVICES is not set, PyTorch will use device 0 by default.
        # To test the 100% utilized GPU, we should explicitly set it.
        
        # Let's use torch.device for explicit device selection.
        target_gpu_idx = 1 # Target GPU 1, which is 100% utilized
        if target_gpu_idx < torch.cuda.device_count():
            device = torch.device(f"cuda:{target_gpu_idx}")
            print(f"Testing on GPU {target_gpu_idx}: {torch.cuda.get_device_name(target_gpu_idx)}")
            
            # Try to allocate a reasonably sized tensor
            # A 100MB tensor should be enough to show up in nvidia-smi if it works
            tensor_size_gb = 10
            num_elements = tensor_size_gb * 1024 * 1024 * 1024 // 4 # Approximately 10GB of float32
            
            x = torch.randn(num_elements, device=device)
            print(f"Allocated {x.element_size() * x.numel() / (1024**2):.2f} MB on GPU {target_gpu_idx}.")
            
            # Perform a simple operation to ensure GPU is active
            print("Performing a simple operation on GPU...")
            y = x * 2 + 1
            z = y.sum()
            
            print(f"Operation completed. Result sum: {z.item():.2f}")
            
            # Keep the tensor in memory for a short duration to observe in nvidia-smi
            print("Keeping GPU active for 10 seconds. Check nvidia-smi now.")
            time.sleep(10)
            
            del x, y, z
            torch.cuda.empty_cache()
            print("Tensor deallocated and CUDA cache emptied.")
            
        else:
            print(f"Target GPU {target_gpu_idx} not found. Only {torch.cuda.device_count()} GPUs available.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("This might indicate that the GPU is fully utilized and cannot allocate more resources, or there's another issue.")

if __name__ == "__main__":
    run_gpu_test()
