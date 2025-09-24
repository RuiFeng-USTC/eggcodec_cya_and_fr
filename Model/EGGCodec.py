import torch
import torch.nn as nn
from Model.encodec.model import EncodecModel
import numpy as np
   
class EGGCodec(nn.Module):
    def __init__(cls, model_path = None, device='cpu'):
        super(EGGCodec, cls).__init__()
        cls.device = device
        target_bandwidths = [1.5, 3., 6, 12., 24.]
        sample_rate = 16000
        channels = 1
        modelEncodec = EncodecModel._get_model(
                    target_bandwidths, sample_rate, channels,
                    causal=False, model_norm='time_group_norm', audio_normalize=True,
                    segment=None, name='my_encodec_16khz')
        
        
        modelEncodec.load_state_dict(torch.load(model_path, map_location=cls.device))
        cls.pre_model = modelEncodec


    
    def forward(cls, audio):
        
        device = cls.device
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio).to(device)
        audio = audio.to(device)
        
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
            
        with torch.no_grad():
            
            # train is necessary to make sure the model to keep the right logic of the inference
            cls.pre_model.train()
            input = audio.unsqueeze(1)
            regg, _, _ = cls.pre_model(input)
            regg = regg.squeeze([1])
        return regg
