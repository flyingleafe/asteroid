
import librosa 
import soundfile as sf
from asteroid import DPTNet, SMoLnet, RegressionFCNN, DCUNet, WaveUNet
import numpy as np
import torch

def denoise_audio(audio_path, model, denoised_file_path):
    noisy, sr = librosa.load(audio_path, sr=8000)
    noisy = torch.tensor(noisy)
    noisy = noisy.cuda()
    model = model.cuda()
    denoised = model(noisy).detach().flatten().cpu().numpy()
    sf.write(denoised_file_path, denoised, samplerate=8000)

baseline_model = RegressionFCNN.from_pretrained('Drone_Models_selected/baseline_model_v1.pt')
smolnet_model = SMoLnet.from_pretrained('Drone_Models_selected/SMoLnet.pt')
dcunet_model = DCUNet.from_pretrained('Drone_Models_selected/dcunet_20_random_v2.pt')
dptnet_model = DPTNet.from_pretrained('Drone_Models_selected/dptnet_model.pt')
waveunet_model = WaveUNet.from_pretrained('Drone_Models_selected/waveunet_model_adapt.pt')

# baseline_model = RegressionFCNN.from_pretrained('/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/Drone_Models_selected/baseline_model_v1.pt')
# smolnet_model = SMoLnet.from_pretrained('/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/Drone_Models_selected/SMoLnet.pt')
# dcunet_model = DCUNet.from_pretrained('/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/Drone_Models_selected/dcunet_20_random_v2.pt')
# dptnet_model = DPTNet.from_pretrained('/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/Drone_Models_selected/dptnet_model.pt')
# waveunet_model = WaveUNet.from_pretrained('/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/Drone_Models_selected/waveunet_model_adapt.pt')


models_dict = {
        'RegressionFCNN': baseline_model,
        'SMoLnet': smolnet_model,
        'DCUNet': dcunet_model,
        'DPTNet': dptnet_model,
        'WaveUNet': waveunet_model
        }

noisy_path = 'test1.wav'
denoised_file_path =  'denoised.wav'
for name, model in models_dict.items():
    print('processing: ', name)
    denoised_file_path = str(name) + '.wav'
    denoise_audio(noisy_path, model, denoised_file_path)

