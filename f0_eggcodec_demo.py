# -*-coding:utf-8-*-

import os
import argparse

import numpy as np

from matplotlib import pyplot as plt

import torch

import Model.EGGCodec as EGGCodec 

import librosa

import soundfile as sf
if __name__ == "__main__":
    
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use (default: 0)')

    parser.add_argument('--audio_path', type=str, default=os.path.join('data_and_result', 'audio.wav'), help='Path to the audio file')

    parser.add_argument('--output_path', type=str, default=os.path.join('data_and_result', 'reco_lar.wav'), help='Path to save the output files')
    
    args = parser.parse_args()

    gpu = args.gpu
    
    audio_path = args.audio_path
    
    output_path = args.output_path

    center = True
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    
    
    sr = 16000
    step_size = 20
    
    model_path = 'model_weights.pth'
    

    model = EGGCodec.EGGCodec(model_path=model_path, device=device)
    
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # 转为tensor
    audio = torch.tensor(audio).to(device)
    
    model.to(device)
    # 读取音频文件
    
    reco_egg = model.get_egg(audio)
    reco_egg = reco_egg.squeeze().cpu().numpy()
    
    sf.write(output_path, reco_egg, sr)
    # 存储结果 为音频文件
    
    exit()
