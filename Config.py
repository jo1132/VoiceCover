import torch

class Isolate_Vocals_config:
    def __init__(self):
        self.input_path = 'https://www.youtube.com/watch?v=EVJjmMW7eII'
        self.pretrained_model = "HP2-4BAND-3090_4band_arch-500m_1.pth"
        self.convertAll = False #if input_path is start http = False / else True
        self.window_size =  512
        self.parameter = "Auto detect"
        self.high_end_process = 'mirroring'
        self.aggressiveness = '0.3'
        self.postprocess = False
        self.threshold = 0.2
        self.nn_architecture = 'Auto detect'
        self.gpu = torch.cuda.is_available()
        self.deepExtraction = False
        self.isVocal = False
        self.suppress = True
        self.output_image = False
        self.tta = True
        self.useCustomArguments = False
        self.CustomArguments = "-h"
        self.download = False
        self.export_as_mp3 = False

        self.data_augmentation_speedup = True
        self.data_augmentation_slowdown = True


class Voice_Model_Training_config:
    def __init__(self):
        self.experiment_name = "experiment_name" #@param {type:"string"}
        self.dataset = "zipfile.zip"  #@param {type:"string"}
        self.pretrain_type = "OV2" #@param ["original", "OV2", "RIN_E3"] {allow-input: false}
        self.path_to_training_folder = "/content/datasets/"
        self.model_architecture = "v2" #@param ["v1","v2"] {allow-input: false}
        self.target_sample_rate = "40k" #@param ["32k", "40k", "48k"] {allow-input: false}
        self.cpu_threads = 2 #@param {type:"integer"}
        self.speaker_id = 0 #@param {type:"integer"}
        self.pitch_extraction_algorithm = "rmvpe" #@param ["harvest", "crepe", "mangio-crepe", "rmvpe"] {allow-input: false}
        self.crepe_hop_length = 64 #@param {type:"integer"}
        self.pitch_guidance = True #@param {type:"boolean"}
        self.save_frequency = 10 #@param {type:"integer"}
        self.total_epochs = 100 #@param {type:"integer"}
        self.batch_size = 8 #@param {type:"integer"}
        self.save_only_latest_ckpt = True #@param {type:"boolean"}
        self.cache_all_training_sets = False #@param {type:"boolean"}
        self.save_small_final_model = True #@param {type:"boolean"}
        self.use_manual_stepToEpoch = False #@param {type:"boolean"}
        self.manual_stepToEpoch = 000 #@param {type:"integer"}