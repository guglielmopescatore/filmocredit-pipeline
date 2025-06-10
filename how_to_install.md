IF YOU HAVE NVIDIA GPU (you must install NVIDIA driver and CUDA 12.6 https://developer.nvidia.com/cuda-12-6-0-download-archive):


    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

    pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/



IF YOU HAVE CPU ONLY:

    pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/




pip install paddleocr==3.0.0

pip install -r requirements.txt