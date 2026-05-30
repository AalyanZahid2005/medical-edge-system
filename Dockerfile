# Base image optimized for Jetson Nano TensorRT & PyTorch
FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

WORKDIR /app

# FIX: Delete the expired Kitware repository so apt-get doesn't crash
# FIX: Nuke Kitware from all source lists
RUN rm -f /etc/apt/sources.list.d/*kitware*.list && sed -i '/kitware/d' /etc/apt/sources.list

# Install system dependencies for OpenCV and DICOM networking
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-pip \
    python3-scipy \
    python3-sklearn \
    && rm -rf /var/lib/apt/lists/*

# Install C++ compilers required to build ONNX from source
RUN apt-get update && apt-get install -y protobuf-compiler libprotobuf-dev && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN pip3 install --default-timeout=1000 -r requirements.txt

RUN pip3 install --no-cache-dir pynetdicom==1.5.7

# Copy all project files and models
COPY . .

# Expose API and Admin ports
EXPOSE 5000 8080 4242

# Default command (overridden by docker-compose)
CMD ["python3", "flask_api_full.py", "--no-stub"]

