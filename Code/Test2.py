import os

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


import torch

def test_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available. Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        x = torch.rand(1000, 1000).to(device)
        y = torch.matmul(x, x)
        print(y)
    else:
        print("CUDA is not available.")

test_gpu()