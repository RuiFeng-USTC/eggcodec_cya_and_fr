import torch.nn as nn
import torch.nn.functional as F
import torch
from librosa.filters import mel as librosa_mel_fn
from torch.nn.utils.parametrizations import weight_norm
import numpy as np

class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=22050,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
        device='cuda'
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(win_length, device=device).float()
        mel_basis = librosa_mel_fn(
            sr=sampling_rate, 
            n_fft=n_fft, 
            n_mels=n_mel_channels, 
            fmin=mel_fmin, 
            fmax=mel_fmax
        )

        mel_basis = torch.from_numpy(mel_basis).to(device).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        
    def mel_spec_item(self, audio_item):
        fft = torch.stft(
            audio_item,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        power_spectrum = torch.abs(fft) ** 2
        mel_output = torch.matmul(self.mel_basis, power_spectrum)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec
    
    def mel_spec_batch2(self, audio_batch):
        fft = torch.stft(
            audio_batch,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        power_spectrum = torch.abs(fft) ** 2
        mel_output = torch.matmul(self.mel_basis, power_spectrum)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec
    
    def mel_spec_batch(self, audio_batch):
        batch_size = audio_batch.size(0)
        mel_batch = []
        for i in range(batch_size):
            mel = self.mel_spec_item(audio_batch[i])
            mel_batch.append(mel)
        return torch.stack(mel_batch, dim=0)
    def forward(self, audioin):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audioin, (p, p), "reflect").squeeze(1)
        
        # fft = torch.stft(
        #     audio,
        #     n_fft=self.n_fft,
        #     hop_length=self.hop_length,
        #     win_length=self.win_length,
        #     window=self.window,
        #     center=False,
        #     return_complex=True,
        # )
        # # 计算能量谱
        # power_spectrum = torch.abs(fft) ** 2  # 取复数的绝对值的平方
        # import time
        # start = time.time()
        # log_mel_spec = self.mel_spec_batch(audio)
        # print('time:', time.time()-start)
        # start = time.time()
        log_mel_spec = self.mel_spec_batch2(audio)
        # print('time:', time.time()-start)
        
        # # 验证两种方法是否一致
        # print(torch.sum(torch.abs(log_mel_spec-log_mel_spec2)))
        # 对每个时间点的频谱能量进行求和
        # energy_spectrum = torch.sum(power_spectrum, dim=-1)  # dim=-1 是频率维度
        # mel_basis_expanded = self.mel_basis.unsqueeze(0)  # [1, 64, 153]

        # 在批量维度上进行广播
        # mel_basis_expanded = mel_basis_expanded.expand(power_spectrum.size(0), -1, -1)  # [batch_size, 64, 153]
        # 应用梅尔滤波器组
        # mel_output = torch.matmul(self.mel_basis, power_spectrum)
        # mel_output = torch.matmul(mel_basis_expanded, power_spectrum)
        # mel_output = torch.einsum('bij,bjk->bik', mel_basis_expanded, power_spectrum)

        # mel_output = torch.matmul(self.mel_basis, torch.sum(torch.pow(fft, 2), dim=[-1]))
        # log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec