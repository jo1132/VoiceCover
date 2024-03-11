###########################################
###################### Setting Current Path
###########################################
import time
start_time = time.time()

import Config
config = Config.Voice_Model_Training_config()
experiment_name = config.experiment_name #@param {type:"string"}
print('experiment_name:', experiment_name)
dataset = config.dataset  #@param {type:"string"}
print('dataset:', dataset)

#@title Clone Repositories
import os
firsttry = True
os.chdir('/content/Mangio-RVC-Fork')
now_dir = os.getcwd()
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "weights"), exist_ok=True)
os.makedirs('/content/rvcDisconnected', exist_ok=True)

#########################################
###################### GPU Check
#########################################
#@title GPU Check
import torch

ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

if torch.cuda.is_available() or ngpu != 0:
  for i in range(ngpu):
    gpu_name = torch.cuda.get_device_name(i)
    if any(
        value in gpu_name.upper()
        for value in ["10", "16", "20", "30", "40", "A2", "A3", "A4", "P4", "A50", "500", "A60", "70", "80", "90", "M4", "T4", "TITAN"]
    ):
      if_gpu_ok = True
      print("Compatible GPU detected: %s" % gpu_name)
      gpu_infos.append("%s\t%s" % (i, gpu_name))
      mem.append(int(torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024 + 0.4))

if if_gpu_ok and len(gpu_infos) > 0:
  gpu_info = "\n".join(gpu_infos)

else:
  raise Exception("No GPU detected; training cannot continue. Please change your runtime type to a GPU.")
gpus = "-".join(i[0] for i in gpu_infos)



#############################################
###################### Set Training Variables
#############################################
#@title Set Training Variables
now_dir = "/content/Mangio-RVC-Fork"

pretrain_type = config.pretrain_type
path_to_training_folder =config.path_to_training_folder
model_architecture = config.model_architecture
target_sample_rate = config.target_sample_rate
cpu_threads = config.cpu_threads
speaker_id = config.speaker_id
pitch_extraction_algorithm = config.pitch_extraction_algorithm
crepe_hop_length = config.crepe_hop_length
pitch_guidance = config.pitch_guidance

#cpu_threads = !nproc
cpu_threads = int(cpu_threads)

exp_dir = f"{now_dir}/logs/{experiment_name}"

assert crepe_hop_length!=None, "You need to input something for crepe_hop_length, silly."
assert crepe_hop_length>0, "Hop length must be more than 0."
assert crepe_hop_length<=512, "Save frequency must be less than 512."

if pretrain_type!="original" and model_architecture!="v2":
  model_architecture="v2"
  print("The new pretrains only support RVC v2 at this time. Your settings have been automatically adjusted.")

#TEMPORARY UNTIL SIMPLCUP 48K IS RELEASED
if pretrain_type!="original" and target_sample_rate=="48k":
  target_sample_rate="40k"
  print("The new pretrains only support 40k sample rate and lower at this time. Your settings have been automatically adjusted.")
if pretrain_type=="RIN_E3" and target_sample_rate!="40k":
  target_sample_rate="40k"
  print("RIN_E3 only supports 40k sample rate at this time. Your settings have been automatically adjusted.")


if(experiment_name == "experiment_name"):
  print("Warning: Your experiment name should be changed to the name of your dataset.")


#############################################
########################## Preprocessing Data
#############################################
import os
import shutil
import zipfile

directories=[]

def sanitize_directory(directory):
  for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path):
      if filename == ".DS_Store" or filename.startswith("._") or not filename.endswith(('.wav', '.flac', '.mp3', '.ogg', '.m4a')):
        os.remove(file_path)
    elif os.path.isdir(file_path):
      #Get rid of the MACOSX directory just so it doesn't mess with renaming later
      if(filename == "__MACOSX"):
        shutil.rmtree(file_path)
        continue
      #Append the directory to directories for future dataset check, then recurse.
      directories.append(file_path)
      sanitize_directory(file_path)

dataset_path = '/content/dataset/' + dataset
final_directory = '/content/datasets'
temp_directory = '/content/temp_dataset'

if os.path.exists(final_directory):
  print("Dataset folder already found. Wiping...")
  shutil.rmtree(final_directory)
if os.path.exists(temp_directory):
  print("Temporary folder already found. Wiping...")
  shutil.rmtree(temp_directory)

if not os.path.exists(dataset_path):
  raise Exception(f'I can\'t find {dataset} in {os.path.dirname(dataset_path)}.')

os.makedirs(final_directory, exist_ok=True)
os.makedirs(temp_directory, exist_ok=True)


#Oops.
os.system("unzip -d " + temp_directory + " -B " + dataset_path)
print("Sanitizing...")
sanitize_directory(temp_directory)

if(len(directories) == 0):
  #If there's no directories, we're dealing with a ZIP of just audio files.
  #Move everything to /dataset/experiment_name/.
  print("Dataset Type: Audio Files (Single Speaker)")
  expDir=os.path.join(final_directory, experiment_name)
  os.makedirs(expDir, exist_ok=True)
  for r, _, f in os.walk(temp_directory):
    for name in f:
      print("cp {}/{} {}".format(temp_directory, name, expDir))
      os.system("cp {}/{} {}".format(temp_directory, name, expDir))
elif(len(directories) == 1):
  #If there's only one directory, we're dealing with a single speaker.
  #Rename the folder to experiment_name and move it to /dataset/.
  print("Dataset Type: Single Speaker")
  fi = os.path.join(temp_directory, experiment_name)
  os.rename(directories[0], fi)
  shutil.move(fi, final_directory)

else:
  #If anything else, we're dealing with multispeaker.
  #Move all folders to /dataset/ indiscriminately.
  print("Dataset Type: Multispeaker")
  for fi in directories:
    shutil.move(fi, final_directory)

shutil.rmtree(temp_directory)

print("Dataset imported.")


###############################################################
########################## Preprocessing and Feature Extraction
###############################################################
#@title Preprocessing and Feature Extraction

import os
import subprocess

assert cpu_threads>0, "CPU threads not allocated correctly."

sr = int(target_sample_rate.rstrip('k'))*1000
pttf = path_to_training_folder + experiment_name
os.makedirs(f"{exp_dir}", exist_ok=True)

cmd = f"python trainset_preprocess_pipeline_print.py \"{pttf}\" {sr} {cpu_threads} \"{exp_dir}\" 1"
print(cmd)
os.system(cmd)

gpuList = gpus.split("-")
cmd = f"python extract_f0_print.py \"{exp_dir}\" {cpu_threads} {pitch_extraction_algorithm} {crepe_hop_length}"
print(cmd)
os.system(cmd)

leng = len(gpus)
cmd = f"python extract_feature_print.py \"device\" {leng} 0 0 \"{exp_dir}\" {model_architecture}"
print(cmd)
os.system(cmd)


#########################################
########################## Index Training
#########################################
#@title Index Training
#@markdown Ensure that Feature Extraction has run successfully before running this cell.

#@markdown Use this option if you wish to save the two extra files generated by index training to your Google Drive. (Only the added index is normally needed.)
save_extra_files_to_drive = False #@param {type:"boolean"}

#Oh dear lord why is this baked into infer-web I hate this
import os
import sys
import traceback
import numpy as np
import faiss

#from sklearn.cluster import MiniBatchKMeans

exp_dir = "%s/logs/%s" % (now_dir, experiment_name)
os.makedirs(exp_dir, exist_ok=True)
feature_dir = (
    "%s/3_feature256" % (exp_dir)
    if model_architecture == "v1"
    else "%s/3_feature768" % (exp_dir)
)
print(feature_dir)
if not os.path.exists(feature_dir):
  raise Exception("No features exist for this model yet. Did you run Feature Extraction?")
listdir_res = list(os.listdir(feature_dir))
if len(listdir_res) == 0:
  raise Exception("No features exist for this model yet. Did you run Feature Extraction?")

try:
  from sklearn.cluster import MiniBatchKMeans
except:
  print("Due to a bug with Colab, we will need to reinstall Numpy real quick. Give me a sec!")
  os.system('pip install -U numpy')
  print("Numpy reinstalled. Please restart the runtime, and then re-run the \"Set Training Variables\" cell to continue.")
  sys.exit()
else:
  print("Proper Numpy version detected.")

infos=[]
npys=[]
for name in sorted(listdir_res):
  phone = np.load("%s/%s" % (feature_dir, name))
  npys.append(phone)
big_npy = np.concatenate(npys, 0)
big_npy_idx = np.arange(big_npy.shape[0])
np.random.shuffle(big_npy_idx)
if big_npy.shape[0] > 2e5:
  print("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
  try:
    big_npy = (
        MiniBatchKMeans(
            n_clusters=10000,
            verbose=True,
            batch_size=256,
            compute_labels = False,
            init="random"
        )
        .fit(big_npy)
        .cluster_centers_

    )
  except:
    info = traceback.format_exc()
    print(info)

np.save("%s/total_fea.npy" % exp_dir, big_npy)
n_ivf = min(int(16*np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
print("%s,%s" % (big_npy.shape, n_ivf))
index = faiss.index_factory(256 if model_architecture == "v1" else 768, "IVF%s,Flat" % n_ivf)
print("Training index...")
index_ivf = faiss.extract_index_ivf(index)
index_ivf.nprobe = 1
index.train(big_npy)
faiss.write_index(
    index,
    "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index" % (exp_dir, n_ivf, index_ivf.nprobe, experiment_name, model_architecture)
)
print("Adding...")
batch_size_add = 8192
for i in range(0, big_npy.shape[0], batch_size_add):
  index.add(big_npy[i:i+batch_size_add])
faiss.write_index(
    index,
    "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
    % (exp_dir, n_ivf, index_ivf.nprobe, experiment_name, model_architecture)
)

npr = index_ivf.nprobe

print("Saving files to Drive...")
DATASET_PATH_DRIVE = "/content/rvcDisconnected/" + experiment_name
if(not os.path.exists(DATASET_PATH_DRIVE)):
    os.makedirs(DATASET_PATH_DRIVE)
DATASET_PATH_COLAB = "/content/Mangio-RVC-Fork/logs/" + experiment_name
if(save_extra_files_to_drive):
  os.system("cp {}/total_fea.npy {}".format(DATASET_PATH_COLAB, DATASET_PATH_DRIVE))
  os.system("cp {}/trained_IVF{}_Flat_nprobe_{}_{}_{}.index {}".format(DATASET_PATH_COLAB, n_ivf, npr, experiment_name, model_architecture, DATASET_PATH_DRIVE))
os.system("cp {}/added_IVF{}_Flat_nprobe_{}_{}_{}.index {}".format(DATASET_PATH_COLAB, n_ivf, npr, experiment_name, model_architecture, DATASET_PATH_DRIVE))

print("All done! Your index file has completed training.")
try:
  firsttry
except:
  print("If you had to restart the runtime, disconnect and delete the runtime in order to continue. (Restarting the runtime again will not work.)")


###################################
########################## Training
###################################
import os
import math
from random import shuffle

#@title Training
save_frequency = config.save_frequency
total_epochs = config.total_epochs
batch_size = config.batch_size
save_only_latest_ckpt = config.save_only_latest_ckpt
cache_all_training_sets = config.cache_all_training_sets
save_small_final_model = config.save_small_final_model
#@markdown The automatically calculated log interval is known to be very inaccurate and can cause delays between an epoch finishing and Tensorboard writes. If you would like, you can manually define a log interval here.
use_manual_stepToEpoch = config.use_manual_stepToEpoch
manual_stepToEpoch = config.manual_stepToEpoch

assert save_frequency!=None, "You need to input something for save_frequency, silly."
assert save_frequency>0, "Save frequency must be more than 0."
if(save_frequency>50):print(f"...A save frequency of {save_frequency}? A bit high, but... alright then.")
assert total_epochs!=None, "You need to input something for total_epochs, silly."
assert total_epochs>0, "Total epochs must be more than 0."
if(total_epochs>10000):print(f"...A total epoch count of of {total_epochs}? This is going to overtrain, but... alright then.")
assert batch_size!=None, "You need to input something for batch_size, silly."
assert batch_size>0, "Batch size must be more than 0."
assert batch_size<=40, "Batch size must be less than 40. (I'd reccomend a value between 6 and 12 for Colab.)"

pretrained_base = "pretrained/" if model_architecture == "v1" else "pretrained_v2/"
unpt = f"_{pretrain_type}" if pretrain_type!="original" else ""

pretrainedD = f"{pretrained_base}f0D{target_sample_rate}{unpt}.pth"
pretrainedG = f"{pretrained_base}f0G{target_sample_rate}{unpt}.pth"

#Log interval
log_interval = 1
liFolderPath = os.path.join(exp_dir, "1_16k_wavs")
if(os.path.exists(liFolderPath) and os.path.isdir(liFolderPath)):
  wav_files = [f for f in os.listdir(liFolderPath) if f.endswith(".wav")]
  if wav_files:
    sample_size = len(wav_files)
    log_interval = math.ceil(sample_size / batch_size)
    if log_interval > 1:
      log_interval += 1

if log_interval > 250 and not use_manual_stepToEpoch:
  print(f"That's a big dataset you got there. Log interval normalized to 200 steps from {log_interval} steps.")
  log_interval = 200

if use_manual_stepToEpoch:
  log_interval = manual_stepToEpoch

#Create Python command
cmd = "python train_nsf_sim_cache_sid_load_pretrain.py -e \"%s\" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s -li %s" % (
    experiment_name,
    target_sample_rate,
    1,
    batch_size,
    0,
    total_epochs,
    save_frequency,
    "-pg %s" % pretrainedG if pretrainedG != "" else "\b",
    "-pd %s" % pretrainedD if pretrainedD != "" else "\b",
    1 if save_only_latest_ckpt else 0,
    1 if cache_all_training_sets else 0,
    1 if save_small_final_model else 0,
    model_architecture,
    log_interval,
)
print(cmd)

#Create mute filelist
gt_wavs_dir = f"{exp_dir}/0_gt_wavs"
feature_dir = (
  f"{exp_dir}/3_feature256"
  if model_architecture == "v1"
  else f"{exp_dir}/3_feature768"
)
f0_dir = f"{exp_dir}/2a_f0"
f0nsf_dir = f"{exp_dir}/2b-f0nsf"
names = (
  set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
  & set([name.split(".")[0] for name in os.listdir(feature_dir)])
  & set([name.split(".")[0] for name in os.listdir(f0_dir)])
  & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
)
opt = []
for name in names:
  opt.append(
    "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
    % (
      gt_wavs_dir.replace("\\", "\\\\"),
      name,
      feature_dir.replace("\\", "\\\\"),
      name,
      f0_dir.replace("\\", "\\\\"),
      name,
      f0nsf_dir.replace("\\", "\\\\"),
      name,
      speaker_id,
    )
  )
fea_dim = 256 if model_architecture == "v1" else 768
for _ in range(2):
  opt.append(
      f"{now_dir}/logs/mute/0_gt_wavs/mute{target_sample_rate}.wav|{now_dir}/logs/mute/3_feature{fea_dim}/mute.npy|{now_dir}/logs/mute/2a_f0/mute.wav.npy|{now_dir}/logs/mute/2b-f0nsf/mute.wav.npy|{speaker_id}"
  )
shuffle(opt)
with open(f"{exp_dir}/filelist.txt", "w") as f:
  f.write("\n".join(opt))
print("Mute filelist written. Best of luck training!")

os.chdir('/content/Mangio-RVC-Fork')
os.system(cmd)
#os.system("cd /content/Mangio-RVC-Fork")
#os.system("load_ext tensorboard")
#os.system("tensorboard --logdir /content/Mangio-RVC-Fork/logs")

print('total processtime:', round(time.time() - start_time, 2))