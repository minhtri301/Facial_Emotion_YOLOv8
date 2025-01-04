## Hướng dẫn cài đặt

request python --version from 8 - 12

## Setup
  1. Tạo môi trường ảo Python:
     
    python -m venv venv
     
    venv\Scripts\activate
     
  2. Cài đặt các thư viện từ requirements.txt:
     
    pip install -r requirements.txt
  
  3. PyTorch dành cho CPU
     
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

## Run
  
    python realtime_emotion.py
