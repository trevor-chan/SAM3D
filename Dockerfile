# FROM python:3.11

# RUN pip install "poetry==1.8.4"
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     libgl1-mesa-glx \
#     curl
# WORKDIR /app
# COPY pyproject.toml poetry.lock ./
# RUN poetry install --no-dev && poetry show
# RUN python -m pip list
# COPY . .

# RUN pip install --no-cache-dir wheel
# RUN mkdir -p /wheels
# RUN pip wheel PyQt5==5.15.11 PyQt5_sip==12.15 -w /wheels
# RUN pip install --no-cache-dir PyQt5==5.15.11
    # qtbase5-dev \
    # qtchooser \
    # qttools5-dev \
    # qttools5-dev-tools \

FROM python:3.11

WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-tk \
    xvfb \
    x11vnc \
    && rm -rf /var/lib/apt/lists/*

ENV DISPLAY=:0.0
RUN mkdir -p ~/.vnc && x11vnc -storepasswd 1234 ~/.vnc/passwd && chmod 600 ~/.vnc/passwd
EXPOSE 5900


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# RUN python -m pip list
COPY . .

RUN echo '#!/bin/bash\n' \
         'Xvfb :0 -screen 0 1024x768x16 &\n' \
         'x11vnc -forever -usepw -display :0.0 &' > /app/start.sh && chmod +x /app/start.sh

CMD ["bash", "/app/start.sh"]