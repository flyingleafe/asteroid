
import librosa 
import soundfile as sf
from asteroid import DPTNet, SMoLnet, RegressionFCNN, DCUNet, WaveUNet
import numpy
import torch

def denoise_audio(audio_path, model, denoised_file_path):
    noisy, sr = librosa.load(audio_path)
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

noisy_path = 'test1.wav'
model = baseline_model
# denoised_file_path = audio_path + 'denoised.wav'
denoised_file_path =  'denoised.wav'
denoise_audio(noisy_path, baseline_model, denoised_file_path)

