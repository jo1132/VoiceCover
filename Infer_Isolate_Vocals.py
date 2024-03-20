#########################################
###################### Data path define
#########################################
import Config_Infer
config = Config_Infer.Isolate_Vocals_config()
input = config.input_path

##########################################
###################### Setting Environment
##########################################
if os.path.isdir('/content/VocalRemover5-COLAB_arch/separated'):
    print('separated 폴더를 정리합니다.')
    folder_path = '/content/VocalRemover5-COLAB_arch/separated'
    for file in os.listdir(folder_path):
        os.remove(os.path.join(folder_path, file))

if os.path.isdir('/content/dataset_Infer'):
    print('dataset_Infer 폴더를 정리합니다.')
    folder_path = '/content/dataset_Infer'
    for file in os.listdir(folder_path):
        os.remove(os.path.join(folder_path, file))

os.makedirs('/content', exist_ok=True)
os.chdir('/content')
# ------------VSC REWRITE
# pyright: reportMissingImports=false, reportUnusedVariable=warning, reportUntypedBaseClass=error
#from google.colab import output
#from google.colab import drive
#from google.colab import files
from sys import exit
import zipfile
import hashlib
import os.path
import shutil
import psutil
import random
import glob
import time
import zlib
import sys
import torch 
from pathvalidate import sanitize_filename # for inference cell
import yt_dlp as youtube_dl # for inference cell

start_time = time.time()

isCPU = torch.cuda.is_available()
#@markdown Uncheck if you want to use VocalRemover5 without mounting to drive.
MountDrive = False #@param{type:"boolean"}
#@markdown Use mounting method
method = 'new' #@param ["new","old"]
#@markdown Mounting path; don't touch this if you don't know what you're doing
mounting_path = '/content/drive/MyDrive' #@param ["snippets:","/content/drive/MyDrive","/content/drive/Shareddrives/<your shared drive name>", "/content/drive/Shareddrives/Shared Drive"]{allow-input: true}
#@markdown Force trigger update
ForceUpdate = False #@param{type:"boolean"}

#update channels
ai = 'https://github.com/NaJeongMo/Colaboratory-Notebook-for-Ultimate-Vocal-Remover'
vercheck = 'https://raw.githubusercontent.com/NaJeongMo/Colaboratory-Notebook-for-Ultimate-Vocal-Remover/main/v'
model_ver = 'https://raw.githubusercontent.com/NaJeongMo/Colaboratory-Notebook-for-Ultimate-Vocal-Remover/main/model_list'

class hide_opt: # hide outputs
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_size(bytes, suffix='B'): # read ram
    global svmem
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f'{bytes:.2f}{unit}{suffix}'
        bytes /= factor
    svmem = psutil.virtual_memory()

def validateModelLinks():
    with hide_opt():
        os.system('wget {} -O model_ver'.format(model_ver))
    model_ver_ = open("model_ver", "r")
    model_ver_ = model_ver_.read()
    with hide_opt():
        os.system('wget {} -O model_list'.format(model_ver_))
    model_list = open("model_list", "r")
    model_list = model_list.readlines()
    models = []
    for i in model_list:
        models.append(i)
    os.remove('model_ver')
    os.remove('model_list')
    return models

def installAI():
    print('Installing ai...', end=' ')
    os.system('git clone {} VocalRemover5-COLAB_arch'.format(ai))
    os.chdir('VocalRemover5-COLAB_arch')
    print('done')

    print('Downloading models...', end=' ')
    for i in validateModelLinks():
        with hide_opt():
            zname = sanitize_filename(os.path.basename(i))
            os.system("wget {}".format(i))
            os.system('unzip -o {}'.format(zname))
            os.remove(zname)
    print('done')

def dlvr(): # download vr to colab only
    print('Warning: changes will not be saved.')
    print('Downloading VR5...')

    os.chdir('/content')
    installAI()
    #os.chdir('/content/VocalRemover5-COLAB_arch')
    print ("Success!")

def check_update():
    if os.path.isdir(f'{mounting_path}/VocalRemover5-COLAB_arch'): # check update if ai installed is True
        os.chdir(f'{mounting_path}/VocalRemover5-COLAB_arch')
        print("Checking for updates...", end=" ")
        with hide_opt():
            os.system('wget {} -O check_ver'.format(vercheck))
        f = open("check_ver", "r")
        nver = f.read()
        f = open("v", "r")
        cver = f.read()
        if cver != nver or ForceUpdate:
            print('New update found! {}'.format(nver))
            choice = str(input('Do you want to update? y/n')).lower()
            if choice == 'y':
                os.chdir('../')
                print('Updating ai...',end=' ')
                with hide_opt():
                    os.system('git clone {} temp_VocalRemover5-COLAB_arch'.format(ai))
                os.system('cp -a temp_VocalRemover5-COLAB_arch/* VocalRemover5-COLAB_arch/')
                os.system('rm -rf temp_VocalRemover5-COLAB_arch')
                print('done')

                print('Downloading models...', end=' ')
                os.chdir('VocalRemover5-COLAB_arch')
                for i in validateModelLinks():
                    with hide_opt():
                        zname = sanitize_filename(os.path.basename(i))
                        os.system("wget {}".format(i))
                        os.system('unzip -o {}'.format(zname))
                        os.remove(zname)
                print('done')
                output.clear()
                os.remove(f'{mounting_path}/VocalRemover5-COLAB_arch/v')
                os.rename(f'{mounting_path}/VocalRemover5-COLAB_arch/check_ver',f'{mounting_path}/VocalRemover5-COLAB_arch/v')
                os.chdir(f'{mounting_path}/VocalRemover5-COLAB_arch') # just to make sure

            else:
                print('Skipping update.')
                os.remove(f'{mounting_path}/VocalRemover5-COLAB_arch/v')
                os.rename(f'{mounting_path}/VocalRemover5-COLAB_arch/check_ver',f'{mounting_path}/VocalRemover5-COLAB_arch/v')
                os.chdir(f'{mounting_path}/VocalRemover5-COLAB_arch') # just to make sure
        else:
            os.remove('check_ver')
            print('No update found.')
    else:
            if os.path.isdir('/content/VocalRemover5-COLAB_arch'):
                print("Success!")
            else:
                dlvr()

#-------------Script begin-------------
if os.path.isdir('/content/VocalRemover5-COLAB_arch') == False:
    if os.path.isdir('/content/VocalRemover5-COLAB_arch'):
        print('Success!')
    else:
        dlvr()


###########################################
###################### DEFINE INFERENCE REQ
###########################################
def crc32(fileName):
    with open(fileName, 'rb') as fh:
        hash = 0
        while True:
            s = fh.read(65536)
            if not s:
                break
            hash = zlib.crc32(s, hash)
        return "%08X" % (hash & 0xFFFFFFFF)
def YouTube(link, dl=True):
    inputsha = hashlib.sha1(bytes(link, encoding='utf8')).hexdigest() + '.wav'
    fmt = '251/140/250/139'
    opt = {'format': fmt, 'outtmpl': inputsha, 'updatetime': False, 'nocheckcertificate': True}
    if dl == True:
        print('YouTube link detected')
        print('Downloading...', end=' ')
    with hide_opt():
        with youtube_dl.YoutubeDL(opt) as ydl:
            global desc
            if dl == True:
                desc = ydl.extract_info(link, download=not os.path.isfile(inputsha))
            else:
                desc = ydl.extract_info(link, download=not True)

    titlename = sanitize_filename(desc['title'])
    if dl == True:
        print('done')
        print(titlename)
    if dl == True:
        return titlename, inputsha
    else:
        return titlename
def zipdir(folder, zipname): # LINUX CALL MODIFIED!!!
    if '.zip' in zipname:
        zipname = zipname.strip('.zip')
    os.system('zip -r {}.zip {}'.format(zipname,folder))
# dlFile(input,pretrained_model,isYouTube='http://' in input,export_as_mp3=export_as_mp3)
def dlFile(track,pretrained_model,isYouTube=False,export_as_mp3=False):
    modelname = os.path.splitext(os.path.basename(pretrained_model))[0]
    stems = [f'_{modelname}_Instruments',f'_{modelname}_Vocals']
    filename = os.path.splitext(os.path.basename(track))[0]
    if isYouTube:
        filename = YouTube(track,dl=False)
    if os.path.isdir(filename):
        shutil.rmtree(filename)
        os.mkdir(filename)
    else:
        os.mkdir(filename)
    if os.path.isfile(f'{filename}.zip'):
        os.remove(f'{filename}.zip')
    if export_as_mp3:
        os.chdir('separated/')
        for i in stems:
            wav_path = filename + i + '.wav'
            mp3_path = filename + i + '.mp3'
            os.system(f'ffmpeg -y -i "{wav_path}" -vn -ar 44100 -ac 2 -b:a 320k "{mp3_path}" -loglevel quiet')
        os.chdir('../')
        for move in stems:
            shutil.move('separated/' + filename + move + '.mp3',filename)
    else:
        for move in stems:
            shutil.move('separated/' + filename + move + '.wav',filename)
    with hide_opt():
        os.system(f'zip -r "{filename}.zip" "{filename}"')
    shutil.rmtree(filename)
    files.download(f'{filename}.zip')


##########################################
############################## VSC REWRITE
##########################################
# ------------VSC REWRITE
if os.path.isfile('main.py') == False:
    if MountDrive:
        os.chdir(mounting_path + '/VocalRemover5-COLAB_arch')
    else:
        os.chdir('/content/VocalRemover5-COLAB_arch')

ScanSeparatedFolder = False #@param {type:"boolean"}
#@markdown Convert all files in your tracks folder
convertAll = config.convertAll #@param {type:"boolean"}
if convertAll:
    convertAll = '--convert_all'
else:
    convertAll = ''
#@markdown Model name (Upload your models in models folder)
pretrained_model = config.pretrained_model #@param ["HighPrecison_4band_arch-124m_1.pth","HighPrecison_4band_arch-124m_2.pth","HP2-4BAND-3090_4band_arch-500m_1.pth","HP2-4BAND-3090_4band_arch-500m_2.pth","HP_4BAND_3090_arch-124m.pth","LOFI_2band-1_arch-34m.pth","LOFI_2band-2_arch-34m.pth","NewLayer_4band_arch-130m_1.pth","NewLayer_4band_arch-130m_2.pth","NewLayer_4band_arch-130m_3.pth","Vocal_HP_4BAND_3090_AGG_arch-124m.pth","Vocal_HP_4BAND_3090_arch-124m.pth","HP-KAROKEE-MSB2-3BAND-3090_arch-124m.pth","HP2-MAIN-MSB2-3BAND-3090_arch-500m.pth","HP-4BAND-V2_arch-124m.pth","MGM-v5-2Band-32000-_arch-default-BETA1.pth","MGM-v5-2Band-32000-_arch-default-BETA2.pth","MGM-v5-3Band-44100-_arch-default-BETA.pth","MGM-v5-4Band-44100-_arch-default-BETA1.pth","MGM-v5-4Band-44100-_arch-default-BETA2.pth","MGM-v5-KAROKEE-32000-_arch-default-BETA1.pth","MGM-v5-KAROKEE-32000-_arch-default-BETA2-AGR.pth","MGM-v5-MIDSIDE-44100-_arch-default-BETA1.pth","MGM-v5-MIDSIDE-44100-_arch-default-BETA2.pth","MGM-v5-Vocal_2Band-32000-_arch-default-BETA1.pth","MGM-v5-Vocal_2Band-32000-_arch-default-BETA2.pth","StackedMGM_1band_arch-default.pth"]{allow-input: true}
print('Using pretrained_model:', pretrained_model)

#@markdown ### Arguments
window_size =  config.window_size#@param {type:"integer"}
if window_size < 320:
    print('Warning: window_size lower than 320.')
if window_size < 272:
    window_size = 272
#1band_sr32000_hl512.json  2band_44100_lofi.json  3band_44100_mid.json
#1band_sr44100_hl512.json  2band_48000.json	 4band_44100.json
#2band_32000.json	  3band_44100.json	 ensemble.json
parameter = config.parameter #@param ["Auto detect","1band_sr32000_hl512.json","1band_sr44100_hl512.json", "2band_32000.json" , "2band_44100_lofi.json", "2band_48000.json", "3band_44100.json", "3band_44100_mid.json", "3band_44100_msb2.json", "4band_44100.json", "4band_v2.json"]
parameter = "modelparams/" + parameter
high_end_process = config.high_end_process #@param ["none","mirroring", "mirroring2" , "bypass"]
aggressiveness = config.aggressiveness #@param {type:"string"}
aggressiveness = float(aggressiveness)
#@markdown Mute low volume vocals
postprocess = config.postprocess #@param {type: "boolean"}
if postprocess:
    threshold = config.threshold #@param {type:"number"}
    min_range = 64 #?param {type:"integer"}
    fade_size = 32 #?param {type:"integer"}
    if min_range < fade_size * 2:
        print('min_range must be greater than or equal fade_size * 2')
        print('Using default instead. (except threshold)')
        min_range = 32
        fade_size = 64
    postprocess = f"-p -thres {threshold} -mrange {min_range} -fsize {fade_size}"
else:
    postprocess = ""
#@markdown ### Architecture
nn_architecture = config.nn_architecture #@param ["Auto detect","default", "34 MB", "124 MB", "130 MB", "500 MB"]
if nn_architecture == '34 MB':
    nn_architecture = '33966KB'
elif nn_architecture == '124 MB':
    nn_architecture = '123821KB'
elif nn_architecture == '130 MB':
    nn_architecture = '129605KB'
elif nn_architecture == '500 MB':
    nn_architecture = '537238KB'
elif nn_architecture == 'default':
    pass
#@markdown ---
#@markdown ### Checkboxes
#@markdown Use GPU for faster conversion
gpu = config.gpu #@param {type: "boolean"}
if gpu == True:
    gpu = 0
else:
    gpu = -1
#@markdown Aggressively remove vocals from Instrumental
deepExtraction = config.deepExtraction #@param {type:"boolean"}
if deepExtraction:
    deepExtraction = "-D"
else:
    deepExtraction = ""
#@markdown Flip Instruments and Vocals output (Only for Vocal Models)
isVocal = config.isVocal #@param {type:"boolean"}
if isVocal:
    isVocal = '--isVocal'
else:
    isVocal = ''
#@markdown Hide warnings
suppress = config.suppress #@param {type:"boolean"}
if suppress:
    suppress = '--suppress'
else:
    suppress = ''
#@markdown Export spectogram image
output_image = config.output_image #@param {type: "boolean"}
if output_image:
    output_image = "-I"
else:
    output_image = ""
#@markdown perform Test Time Augmentation to improve the separation quality
tta = config.tta #@param {type: "boolean"}
if tta:
    tta = "-t"
else:
    tta = ""
#@markdown Use custom arguments
useCustomArguments = config.useCustomArguments #@param {type: "boolean"}
CustomArguments = config.CustomArguments#@param {type:"string"}
#@markdown Download files
download = config.download #@param {type:"boolean"}
export_as_mp3 = config.export_as_mp3 #@param {type:"boolean"}
#@markdown Use all model
model_version = "Don't use all model" #@param ["Don't use all model","v5_new", "v5", "all"]
# {none,v5,v5_new,all}
# automation
if nn_architecture == 'Auto detect':
    if 'arch-default' in pretrained_model:
        nn_architecture = 'default'
    elif 'arch-34m' in pretrained_model:
        nn_architecture = '33966KB'
    elif 'arch-124m' in pretrained_model:
        nn_architecture = '123821KB'
    elif 'arch-130m' in pretrained_model:
        nn_architecture = '129605KB'
    elif 'arch-500m' in pretrained_model:
        nn_architecture = '537238KB'
    else:
        print('Error! autoDetect_arch')
        print('Using 124 MB instead.')
        nn_architecture = '123821KB'
if parameter == 'modelparams/Auto detect':
    if '4band' in pretrained_model.lower():
        if 'v2' in pretrained_model.lower():
            parameter = 'modelparams/4band_v2.json'
        else:
            parameter = 'modelparams/4band_44100.json'
    elif '3band' in pretrained_model.lower():
        if 'msb2' in pretrained_model.lower():
            parameter = 'modelparams/3band_44100_msb2.json'
        else:
            parameter = 'modelparams/3band_44100.json'
    elif 'midside' in pretrained_model.lower():
        parameter = 'modelparams/3band_44100_mid.json'
    elif '2band' in pretrained_model.lower():
        if 'lofi' in pretrained_model.lower():
            parameter = 'modelparams/2band_44100_lofi.json'
        else:
            parameter = 'modelparams/2band_48000.json'
    else:
        print('Parameter auto detect failed, using 1band instead.')
        parameter = 'modelparams/1band_sr44100_hl512.json'
if '34m' in pretrained_model or '124m' in pretrained_model or '130m' in pretrained_model or '500m' in pretrained_model:
    pretrained_model = "models/v5_new/" + pretrained_model
else:
    pretrained_model = "models/v5/" + pretrained_model
    if os.path.isfile(pretrained_model) == False:
        print('========================================================')
        print('                Error model not found.')
        print('Custom models should be uploaded in "models/v5/" folder!')
        print('========================================================')
if 'https://' not in input:
    if ScanSeparatedFolder:
        if input in ''.join(glob.glob('separated/*')):
            input = 'separated/' + input
        else:
            print('File not found in separated folder.')
            input = 'tracks/' + input
    else:
        input = 'tracks/' + input
elif 'https://' in input and ScanSeparatedFolder:
    print('Skipping "Separated" folder scan since a link is given.')

if model_version == "Don't use all model":
    model_version = 'none'

# --------------AI----------------
start_time = time.time()
#window_size,os.path.splitext(os.path.basename(pretrained_model)[0]),os.path.splitext(os.path.basename(parameter)[0],settings_tta = 'True' if tta else 'False'
#settings_tta = 'True' if tta else 'False',os.path.splitext(os.path.basename(parameter)[0],os.path.splitext(os.path.basename(pretrained_model)[0]),window_size
settings_tta = 'True' if tta else 'False'
settings_deepExtraction = 'True' if deepExtraction else 'False'

print('Window size: {}'.format(window_size))
print('Model: {}'.format(os.path.splitext(os.path.basename(pretrained_model))[0]))
print('Parameter: {}'.format(os.path.splitext(os.path.basename(parameter))[0]))
print('Aggressiveness: {}'.format(aggressiveness))
print('High end process: {}'.format(high_end_process))

print('TTA: {}'.format(settings_tta))
print('Deep Extraction: {}'.format(settings_deepExtraction))
print()

if useCustomArguments == False:
    os.system(f'python3.8 main.py -i "{input}" {convertAll} --useAllModel "{model_version}" --model_params "{parameter}" -P "{pretrained_model}" -w {window_size} -H "{high_end_process}" --aggressiveness {aggressiveness} -n "{nn_architecture}" -g {gpu} {deepExtraction} {isVocal} {suppress} {output_image} {postprocess} {tta}')
    if download and convertAll:
        sys.exit("No no, this is not an error but downloading with convertAll is not yet possible. Please DON'T report this to me (Hv) or the server")
    if download:
        dlFile(input,pretrained_model,isYouTube='https://' in input,export_as_mp3=export_as_mp3)
else:
    os.system(f'python3.8 main.py {CustomArguments}')
print('Notebook took: {0:.{1}f}s'.format(time.time() - start_time, 1))

############################################
###################### Make Zipfile and Move
############################################
import os
import shutil

save_path = '/content/dataset_Infer/'
if not os.path.isdir(save_path):
    os.makedirs(save_path)

process_path = '/content/VocalRemover5-COLAB_arch/separated'

for i, file in enumerate(os.listdir(process_path)):
    if 'Vocals.wav' in file:
        Instruments = file.split('_')[:-1]
        Instruments = ('_').join(Instruments+['Instruments.wav'])
        file_nospace = file.replace(' ', '')
        Instruments_nospace = Instruments.replace(' ', '')
        shutil.move(os.path.join(process_path, file), os.path.join(save_path, file_nospace))
        shutil.move(os.path.join(process_path, Instruments), os.path.join(save_path, Instruments_nospace))

## rm files in separated folder
if os.path.isdir(process_path):
    for file in os.listdir(process_path):
        os.remove(os.path.join(process_path, file))

print('total process time:', round(time.time()-start_time, 2))