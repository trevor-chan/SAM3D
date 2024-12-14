# FROM python:3.11

# WORKDIR /app
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     libatlas-base-dev \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     python3-tk \
#     xvfb \
#     x11vnc \
#     && rm -rf /var/lib/apt/lists/*

# ENV DISPLAY=:0.0
# RUN mkdir -p ~/.vnc && x11vnc -storepasswd 1234 ~/.vnc/passwd && chmod 600 ~/.vnc/passwd
# EXPOSE 5900


# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# # RUN python -m pip list
# COPY . .

# RUN echo '#!/bin/bash\n' \
#          'Xvfb :0 -screen 0 1024x768x16 &\n' \
#          'x11vnc -forever -usepw -display :0.0 &' > /app/start.sh && chmod +x /app/start.sh

# CMD ["bash", "/app/start.sh"]

FROM nvidia/cuda:11.8.0-base-ubuntu20.04

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-tk \
    python3-pip \
    xvfb \
    x11vnc \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GPU support
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Set the display environment
ENV DISPLAY=:0.0

# Install Python dependencies
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Expose ports (5900 for VNC)
EXPOSE 5900

# Copy application files
COPY . .

# Create a startup script to initialize Xvfb and x11vnc
RUN echo '#!/bin/bash\n' \
         'Xvfb :0 -screen 0 1024x768x16 &\n' \
         'x11vnc -forever -usepw -display :0.0 &' > /app/start.sh && chmod +x /app/start.sh

# Set the startup command
CMD ["bash", "/app/start.sh"]
