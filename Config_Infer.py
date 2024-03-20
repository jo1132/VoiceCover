import torch
from multiprocessing import cpu_count

class Isolate_Vocals_config:
    def __init__(self):
        self.input_path = 'https://www.youtube.com/watch?v=VcEDy-djQXs'
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


class Inference_config:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else 'cpu'
        self.is_half = True
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                print("16系/10系显卡和P40强制单精度")
                self.is_half = False
                for config_file in ["32k.json", "40k.json", "48k.json"]:
                    with open(f"configs/{config_file}", "r") as f:
                        strr = f.read().replace("true", "false")
                    with open(f"configs/{config_file}", "w") as f:
                        f.write(strr)
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
        elif torch.backends.mps.is_available():
            print("没有发现支持的N卡, 使用MPS进行推理")
            self.device = "mps"
        else:
            print("没有发现支持的N卡, 使用CPU进行推理")
            self.device = "cpu"
            self.is_half = True

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G显存配置
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G显存配置
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max