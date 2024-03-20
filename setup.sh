apt update
mkdir /content

# git clone Retrieval-based-Voice-Conversion-WebUI
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git

# Isolates Vocal 
# PC
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#Linux
# pip3 install torch torchvision torchaudio
pip install pathvalidate
pip install yt_dlp
pip install samplerate==0.1.0
pip install librosa==0.9.1
pip install git+https://github.com/ytdl-org/ytdl-nightly.git@2023.08.07
####
pip install numba==0.48.0
pip install resampy==0.2.2
## February 2024 patch
#apt-get install python3.8
#apt-get install python3.8-distutils
#apt-get install python3.8 pip
pip install librosa==0.9.1
pip install numpy==1.19.5
pip install numba==0.55.0
pip install tqdm
#pip install torch==1.13.1
pip install yt_dlp
pip install samplerate==0.1.0
pip install git+https://github.com/ytdl-org/ytdl-nightly.git@2023.08.07
pip install opencv-python
pip install pathvalidate
pip install pydub

# Inference setup
pip install torchcrepe

# Training Voicemodel
cd /content
apt install -qq -y build-essential
apt install -qq -y python3-dev
apt install -qq -y ffmpeg
apt install -qq -y aria2
pip install --upgrade pip setuptools wheel faiss-gpu fairseq ffmpeg ffmpeg-python praat-parselmouth pyworld numpy==1.23.5 numba==0.56.4 librosa==0.9.2 tensorboard
git clone -b pr-optimization --single-branch https://github.com/alexlnkp/Mangio-RVC-Tweaks.git
mv /content/Mangio-RVC-Tweaks /content/Mangio-RVC-Fork
git clone https://github.com/maxrmorrison/torchcrepe.git
mv torchcrepe/torchcrepe Mangio-RVC-Fork/
rm -rf torchcrepe

#@title Download Pretrained Models
#Didn't ask.

#V1
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/v1/f0G32k.pth -d /content/Mangio-RVC-Fork/pretrained -o f0G32k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/v1/f0D32k.pth -d /content/Mangio-RVC-Fork/pretrained -o f0D32k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/v1/f0G40k.pth -d /content/Mangio-RVC-Fork/pretrained -o f0G40k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/v1/f0D40k.pth -d /content/Mangio-RVC-Fork/pretrained -o f0D40k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/v1/f0G48k.pth -d /content/Mangio-RVC-Fork/pretrained -o f0G48k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/v1/f0D48k.pth -d /content/Mangio-RVC-Fork/pretrained -o f0D48k.pth

#V2
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/f0G32k.pth -d /content/Mangio-RVC-Fork/pretrained_v2 -o f0G32k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/f0D32k.pth -d /content/Mangio-RVC-Fork/pretrained_v2 -o f0D32k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/f0G40k.pth -d /content/Mangio-RVC-Fork/pretrained_v2 -o f0G40k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/f0D40k.pth -d /content/Mangio-RVC-Fork/pretrained_v2 -o f0D40k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/f0G48k.pth -d /content/Mangio-RVC-Fork/pretrained_v2 -o f0G48k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/f0D48k.pth -d /content/Mangio-RVC-Fork/pretrained_v2 -o f0D48k.pth

#OV2 pretrains
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/poiqazwsx/Ov2Super32kfix/resolve/main/f0Ov2Super32kG.pth -d /content/Mangio-RVC-Fork/pretrained_v2 -o f0G32k_OV2.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/poiqazwsx/Ov2Super32kfix/resolve/main/f0Ov2Super32kD.pth -d /content/Mangio-RVC-Fork/pretrained_v2 -o f0D32k_OV2.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ORVC/Ov2Super/resolve/main/f0Ov2Super40kG.pth?download=true -d /content/Mangio-RVC-Fork/pretrained_v2 -o f0G40k_OV2.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ORVC/Ov2Super/resolve/main/f0Ov2Super40kD.pth?download=true -d /content/Mangio-RVC-Fork/pretrained_v2 -o f0D40k_OV2.pth
#TEMP UNTIL NEW PRETRAINS ARE OUT
#!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/f0G48k.pth -d /content/Mangio-RVC-Fork/pretrained_v2 -o f0G48k_OV2.pth
#!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/f0D48k.pth -d /content/Mangio-RVC-Fork/pretrained_v2 -o f0D48k_OV2.pth

#RIN_E3 pretrains
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/MUSTAR/RIN_E3/resolve/main/RIN_E3_G.pth -d /content/Mangio-RVC-Fork/pretrained_v2 -o f0G40k_RIN_E3.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/MUSTAR/RIN_E3/resolve/main/RIN_E3_D.pth -d /content/Mangio-RVC-Fork/pretrained_v2 -o f0D40k_RIN_E3.pth

#Hubert/RMVPE
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/hubert_base.pt -d /content/Mangio-RVC-Fork -o hubert_base.pt
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/rmvpe.pt -d /content/Mangio-RVC-Fork -o rmvpe.pt

#FM JSONs
rm -rf /content/Mangio-RVC-Fork/configs/32k.json
rm -rf /content/Mangio-RVC-Fork/configs/40k.json
rm -rf /content/Mangio-RVC-Fork/configs/48k.json
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/32k.json -d /content/Mangio-RVC-Fork/configs -o 32k.json
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/40k.json -d /content/Mangio-RVC-Fork/configs -o 40k.json
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/48k.json -d /content/Mangio-RVC-Fork/configs -o 48k.json