# -*-coding:utf-8-*-

import os
import argparse

import numpy as np

from matplotlib import pyplot as plt


import Model.EGGCodec as EGGCodec 

import librosa

import soundfile as sf
if __name__ == "__main__":
    
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description="Process some parameters.")


    parser.add_argument('--audio_path', type=str, default=os.path.join('data_and_result', 'audio.wav'), help='Path to the audio file')
                        
    parser.add_argument('--lar_path', type=str, default=os.path.join('data_and_result', 'lar.wav'), help='Original lar files')
    
    parser.add_argument('--reco_lar_path', type=str, default=os.path.join('data_and_result', 'reco_lar.wav'), help='Path to save the output files')
    
    parser.add_argument('--pitch_native_path', type=str, default=os.path.join('data_and_result', 'pitch_native.txt'), help='Path to the original pitch file')
                        
    parser.add_argument('--pitch_from_reco_egg_path', type=str, default=os.path.join('data_and_result', 'pitch_from_reco_egg.txt'), help='Path to the pitch from peakdet files')
    
    
    args = parser.parse_args()
    
    audio_path = args.audio_path
    
    lar_path = args.lar_path
    
    reco_lar_path = args.reco_lar_path
    
    pitch_native_path = args.pitch_native_path
    
    pitch_from_reco_egg_path = args.pitch_from_reco_egg_path

    audio_wav = librosa.load(audio_path, sr=16000)[0]
    
    lar_wav = librosa.load(lar_path, sr=16000)[0]
    
    reco_lar_wav = librosa.load(reco_lar_path, sr=16000)[0]
    
    pitch_native = np.loadtxt(pitch_native_path)
    
    pitch_from_reco_egg = np.loadtxt(pitch_from_reco_egg_path)
    
    num_frames = min(len(pitch_native), len(pitch_from_reco_egg))
    
    # 4行图
    fig, axs = plt.subplots(4, 1, figsize=(10, 12))
    
    time_sp = np.arange(len(audio_wav)) / 16000
    
    time_frame = np.arange(num_frames) * 0.01
    
    # 第一行音频
    axs[0].plot(time_sp, audio_wav)
    
    axs[0].set_title('Audio Waveform')
    
    
    # 第二行原始lar
    axs[1].plot(time_sp, lar_wav)
    axs[1].set_title('Original Lar Waveform')
    
    # 第三行重建lar
    axs[2].plot(time_sp, reco_lar_wav)
    axs[2].set_title('Reconstructed Lar Waveform')
    
    # 第四行f0对比
    axs[3].plot(time_frame, pitch_native[:num_frames, 1], label='Native Pitch', alpha=0.7)
    axs[3].plot(time_frame, pitch_from_reco_egg[:num_frames, 1], label='Pitch from Reco Egg', alpha=0.7)
    axs[3].set_title('Pitch Comparison')
    axs[3].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join('data_and_result', 'comparison_plot.pdf'))