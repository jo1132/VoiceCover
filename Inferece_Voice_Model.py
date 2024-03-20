import os,sys,pdb,torch
os.chdir('/root/VoiceCover')
now_dir = os.getcwd()
sys.path.append(now_dir)
import argparse
import glob
import sys
import torch
from multiprocessing import cpu_count
import ffmpeg
import numpy as np


def load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()

now_dir=os.getcwd()
sys.path.append(now_dir)
sys.path.append(os.path.join(now_dir,"Retrieval-based-Voice-Conversion-WebUI"))

from vc_infer_pipeline import VC
from lib.infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono, SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono
from fairseq import checkpoint_utils
from scipy.io import wavfile
import Config_Infer

hubert_model=None
def load_hubert():
    global hubert_model
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(["/root/VoiceCover/Retrieval-based-Voice-Conversion-WebUI/hubert_base.pt"],suffix="",)
    hubert_model = models[0]
    hubert_model = hubert_model.to(device)
    if(is_half):hubert_model = hubert_model.half()
    else:hubert_model = hubert_model.float()
    hubert_model.eval()

def vc_single(sid,input_audio,f0_up_key,f0_file,f0_method,file_index,index_rate,filter_radius=3,resample_sr=48000,rms_mix_rate=0.25, protect=0.33):
    global tgt_sr,net_g,vc,hubert_model
    if input_audio is None:return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    audio=load_audio(input_audio,16000)
    times = [0, 0, 0]
    if(hubert_model==None):load_hubert()
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version")
    audio_opt=vc.pipeline(hubert_model,net_g,sid,audio,input_audio,times,f0_up_key,f0_method,file_index,index_rate,if_f0,filter_radius=filter_radius,tgt_sr=tgt_sr,resample_sr=resample_sr,rms_mix_rate=rms_mix_rate,version=version,protect=protect,f0_file=f0_file, crepe_hop_length=128)
    # print(times)
    return audio_opt


def get_vc(model_path, device_, is_half_):
    global n_spk,tgt_sr,net_g,vc,cpt,device,is_half
    device = device_
    is_half = is_half_
    config = Config_Infer.Inference_config()
    print("loading pth %s"%model_path)
    cpt = torch.load(model_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3]=cpt["weight"]["emb_g.weight"].shape[0]#n_spk
    if_f0=cpt.get("f0",1)
    version=cpt.get("version", "v2")
    if(if_f0==1):
        if version == "v1":
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=is_half)
    else:
        if version == "v1":
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))  # 不加这一行清不干净，真奇葩
    net_g.eval().to(device)
    if (is_half):net_g = net_g.half()
    else:net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk=cpt["config"][-3]

import os
import sys
now_dir=os.getcwd()
sys.path.append(now_dir)
sys.path.append(os.path.join(now_dir,"Retrieval-based-Voice-Conversion-WebUI"))

from IPython.display import Audio
from scipy.io.wavfile import write as write_wav

semantic_path = "semantic_output/pytorch_model.bin" # set to None if you don't want to use finetuned semantic
coarse_path = "coarse_output/pytorch_model.bin" # set to None if you don't want to use finetuned coarse
fine_path = "fine_output/pytorch_model.bin" # set to None if you don't want to use finetuned fine
use_rvc = True # Set to False to use bark without RVC
rvc_name = 'KanyeV2_Redux_40khz'
rvc_path = f"Retrieval-based-Voice-Conversion-WebUI/weights/{rvc_name}.pth"
index_path = f"Retrieval-based-Voice-Conversion-WebUI/logs/{rvc_name}/added_IVF256_Flat_nprobe_1_{rvc_name}_v2.index"
device="cuda:0"
is_half=True
SAMPLE_RATE = 24_000

#from rvc_infer import get_vc, vc_single
get_vc(rvc_path, device, is_half)

infer_data_path = '/content/dataset_Infer'
for file in os.listdir(infer_data_path):
    if 'Vocals.wav' in file:
        filepath = os.path.join(infer_data_path, file)
        
index_rate = 0.75
f0up_key = -6
filter_radius = 3
rms_mix_rate = 0.25
protect = 0.33
resample_sr = SAMPLE_RATE
f0method = "harvest" #harvest or pm
try:
    audio_array = vc_single(0,filepath,f0up_key,None,f0method,index_path,index_rate, filter_radius=filter_radius, resample_sr=resample_sr, rms_mix_rate=rms_mix_rate, protect=protect)
except:
    audio_array = vc_single(0,filepath,f0up_key,None,'pm',index_path,index_rate, filter_radius=filter_radius, resample_sr=resample_sr, rms_mix_rate=rms_mix_rate, protect=protect)

save_file_path = os.path.join(infer_data_path, f'{rvc_name}_{os.path.split(filepath)[0]}')
write_wav(filepath, SAMPLE_RATE, audio_array)

#Audio(audio_array, rate=SAMPLE_RATE)