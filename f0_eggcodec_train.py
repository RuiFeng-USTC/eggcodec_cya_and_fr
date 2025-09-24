# -*-coding:utf-8-*-
# 优化版本
# 重构部分代码, 优化部分代码
# 标准库
import os
import random
import json
import datetime
import warnings
import shutil
import argparse
from pathlib import Path

# 数值计算和数据处理库
import numpy as np
import pandas as pd

# 音频处理库
import soundfile
import wave
import librosa



from matplotlib import pyplot as plt
from tqdm import tqdm

from preprocess.process import *
import preprocess.Egg as Egg
import Custom.CustomModel as CM
from Custom.CustomDataset import *
from Custom.CustomDataset import CustomDataset_flow
import ArgsManager.ArgsDefine as ArgsDef
from VisualizeKit.visualize_utils import *
import PlatformManager.LocationLocator as PM
import signal_tool.audio_signal as asp


import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn as nn
import torch.optim as optim

from datasets import Dataset, DatasetDict, Audio, ClassLabel, Sequence, Value, Array2D
from transformers import AutoFeatureExtractor, AutoModelForAudioFrameClassification, TrainingArguments, Trainer, Wav2Vec2Processor
import torch.nn.functional as F


import evaluate
from encodec.model import EncodecModel
from encodec.msstftd import MultiScaleSTFTDiscriminator
from encodec.audio_to_mel import Audio2Mel
from multiprocessing import Process



def basic_loss(wav1, wav2, time_domain_loss, freq_domain_loss, k_time_domain_loss, k_freq_domain_loss, sample_rate=24000, device='cuda'):
    
    l1Loss = torch.nn.L1Loss(reduction='mean')
    l2Loss = torch.nn.MSELoss(reduction='mean')
    loss = torch.tensor([0.0], device=device, requires_grad=True)
    
    freq_loss = torch.tensor([0.0], device=device, requires_grad=True)
    if freq_domain_loss != 'none':
        for i in range(5, 11):
            fft = Audio2Mel(win_length=2 ** i, hop_length=2 ** i // 4, n_mel_channels=64, sampling_rate=sample_rate, device=device)
            if freq_domain_loss == 'l1':
                freq_loss = freq_loss + l1Loss(fft(wav1), fft(wav2))
            elif freq_domain_loss == 'l2':
                freq_loss = freq_loss + l2Loss(fft(wav1), fft(wav2))
            elif freq_domain_loss == 'l1_l2': 
                freq_loss = freq_loss + l1Loss(fft(wav1), fft(wav2)) + l2Loss(fft(wav1), fft(wav2))
            else:
                raise ValueError(f'freq_domain_loss {freq_domain_loss} is not supported.')
        freq_loss = freq_loss / 6
    
    time_loss = torch.tensor([0.0], device=device, requires_grad=True)
    if time_domain_loss != 'none':
        if time_domain_loss == 'l1':
            time_loss = time_loss + l1Loss(wav1, wav2)
        elif time_domain_loss == 'l2':
            time_loss = time_loss + l2Loss(wav1, wav2)
        elif time_domain_loss == 'l1_l2':
            time_loss = time_loss + l1Loss(wav1, wav2) + l2Loss(wav1, wav2)
        elif time_domain_loss == 'ppmc':
            time_loss = asp.ppmc_fcn(wav1.squeeze(1), wav2.squeeze(1))
        elif time_domain_loss == 'cosine':
            time_loss = 1 - F.cosine_similarity(wav1.squeeze(1), wav2.squeeze(1), dim=1).mean()
        elif time_domain_loss == 'cosine_l1_l2':
            time_loss = (l1Loss(wav1, wav2) + l2Loss(wav1, wav2)) / 100 + (1 - F.cosine_similarity(wav1.squeeze(1), wav2.squeeze(1), dim=1).mean())
    
    loss = k_time_domain_loss * time_loss + k_freq_domain_loss * freq_loss
    return loss


def disc_loss(logits_real, logits_fake, device):
    cx = torch.nn.ReLU()
    lossd = torch.tensor([0.0], device=device, requires_grad=True)
    for tt1 in range(len(logits_real)):
        lossd = lossd + torch.mean(cx(1-logits_real[tt1])) + torch.mean(cx(1+logits_fake[tt1]))
    lossd = lossd / len(logits_real)
    return lossd


def adversarial_loss(logits_fake, device):
    relu = torch.nn.ReLU()
    loss = torch.tensor([0.0], device=device, requires_grad=True)
    for tt1 in range(len(logits_fake)):
        loss = loss + torch.mean(relu(1 - logits_fake[tt1])) / len(logits_fake)
    return loss


def feature_matching_loss(fmap_real, fmap_fake, device):
    l1Loss = torch.nn.L1Loss(reduction='mean')
    loss = torch.tensor([0.0], device=device, requires_grad=True)
    factor = 100 / (len(fmap_real) * len(fmap_real[0]))
    for tt1 in range(len(fmap_real)):
        for tt2 in range(len(fmap_real[tt1])):
            loss = loss + l1Loss(fmap_real[tt1][tt2].detach(), fmap_fake[tt1][tt2]) * factor
    return loss * (2 / 3)

    


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description="Process some parameters.")
    
    parser.add_argument('--method', type=str, default='encodec_conv_f0loss', help='Method to be used (default: %(default)s)')      # [mel_mlp, mfcc_mlp, wav2vec2_mlp, encodec_mlp, encodec_w2v2_mlp, encodec_conv, direct_dsp, encodec_dsp, encodec_conv_f0loss]
    
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use (default: 0)')

    parser.add_argument('--target_format', type=str, default='f0', help='Target format (default: %(default)s)') # ['f0', 'cent']
    
    parser.add_argument('--train_corpus', type=str, default='PTDB', help='Train corpus (default: %(default)s)') # ['IRONIC', 'CLD', 'PTDB']
    
    parser.add_argument('--fc_highpass', type=int, default=0, help='fc_highpass (default: %(default)s)')
    
    parser.add_argument('--extra', type=str, default='', help='Extra information (default: %(default)s)')
    
    parser.add_argument('--use_noise', type=str, default='true', help='use noise (default: %(default)s)') # ['true', 'false']
    
    parser.add_argument('--snr_db', type=float, default=40, help='SNR (default: %(default)s)')
    
    parser.add_argument('--k_time_domain_loss', type=float, default=100, help='k_time_domain_loss (default: %(default)s)')
    
    parser.add_argument('--time_domain_loss', type=str, default='cosine', help='time domain loss (default: %(default)s)') # ['none', 'l1', 'l2', 'cosine', 'ppmc']
    
    parser.add_argument('--k_freq_domain_loss', type=float, default=1, help='k_freq_domain_loss (default: %(default)s)')
    
    parser.add_argument('--freq_domain_loss', type=str, default='l1_l2', help='freq domain loss (default: %(default)s)') # ['none', 'l1', 'l2', 'l1_l2']
    
    parser.add_argument('--use_gan', type=str, default='false', help='use GAN (default: %(default)s)') # ['true', 'false']
    
    parser.add_argument('--all_flag', type=str, default='false', help='all_flag (default: %(default)s)') # 用于一遍准备所有数据集的
    
    
    
    args = parser.parse_args()
    
    method = args.method
    
    gpu = args.gpu
        
    target_format = args.target_format
    
    train_corpus = args.train_corpus
    
    extra = args.extra
    
    fc_highpass = args.fc_highpass
    use_noise = args.use_noise
    snr_db = args.snr_db
    k_time_domain_loss = args.k_time_domain_loss
    time_domain_loss = args.time_domain_loss
    k_freq_domain_loss = args.k_freq_domain_loss
    freq_domain_loss = args.freq_domain_loss
    use_gan = args.use_gan
    
    
    all_flag = args.all_flag
    center = True
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    
    pading_single_length_dict = {'mel_mlp': 0, 'mfcc_mlp': 0, 'wav2vec2_mlp': 320, 'conv_egg_mlp':320, 'conv_audio_mlp':320}
    try:
        padding_single_length = pading_single_length_dict[method]
    except:
        padding_single_length = 320
    
    sr = 16000
    step_size = 20
    
    process = False
    
    locator = PM.DataLocationLocator()
    
    
    # 自适应周期滤波
    if fc_highpass == 0:
        print('No filter')
        add_filter_l_args = None
    else:
        add_filter_l_args = ArgsDef.filter_args_cls(filter_method='highpass', filter_freq=(fc_highpass,), order=1, zerophase=True)
    


    debug_args = None
    eval_args = None
    
    if train_corpus == 'IRONIC' or all_flag == 'true':
        # debug_args = ArgsDef.debug_args_cls(debug_file='M03', debug_time=(2, 3))
        dp_args_train = ArgsDef.F0DataProcessArgs(locator=locator,
                                            corpus=locator.get_absolute_path('corpus/IRONIC'),
                                            dp_func_name = 'IRONIC',
                                            center=center,
                                            vad_flag=True,
                                            sr=sr,
                                            band=30,
                                            step_size=step_size,
                                            process=process, 
                                            add_filter_l_args=add_filter_l_args,
                                            num_workers=20,
                                            batch_length=48000,
                                            method=method,
                                            task='f0',
                                            padding_single_length=padding_single_length,
                                            target_format=target_format,
                                            debug=debug_args,
                                            eval=eval_args,
                                            f0_window_size=512,
                                            egg_window_size=512,)
    if train_corpus == 'CLD' or all_flag == 'true':
        dp_args_train = ArgsDef.F0DataProcessArgs(locator=locator,
                                            corpus=locator.get_absolute_path('corpus/CLD'),
                                            dp_func_name = 'CLD',
                                            center=center,
                                            sr=sr,
                                            band=30,
                                            step_size=step_size,
                                            process=process, 
                                            add_filter_l_args=add_filter_l_args,
                                            num_workers=20,
                                            batch_length=48000,
                                            method=method,
                                            task='f0',
                                            padding_single_length=padding_single_length,
                                            target_format=target_format,
                                            debug=debug_args,
                                            eval=eval_args,
                                            f0_window_size=512,
                                            egg_window_size=512,)
    
    if train_corpus == 'PTDB' or all_flag == 'true':
        # debug_args = ArgsDef.debug_args_cls(debug_file='M03', debug_time=(4.75, 5))
        dp_args_train = ArgsDef.F0DataProcessArgs(locator=locator,
                                                corpus=locator.get_absolute_path('corpus/PTDB_TUG'),
                                                dp_func_name = 'PTDB_TUG',
                                                center=center,
                                                sr=sr,
                                                band=30,
                                                step_size=step_size,
                                                process=process, 
                                                add_filter_l_args=add_filter_l_args,
                                                num_workers=24,
                                                batch_length=48000,
                                                method=method,
                                                task='f0',
                                                padding_single_length=padding_single_length,
                                                target_format=target_format,
                                                debug=debug_args,
                                                eval=eval_args,
                                                f0_window_size=512,
                                                egg_window_size=512,)
    
    
    if train_corpus == 'CMU' or all_flag == 'true':
        dp_args_train = ArgsDef.F0DataProcessArgs(locator=locator,
                                                corpus=locator.get_absolute_path('corpus/CMU_ARCTIC'),
                                                dp_func_name = 'CMU_ARCTIC',
                                                center=center,
                                                sr=sr,
                                                band=30,
                                                step_size=step_size,
                                                process=process, 
                                                add_filter_l_args=add_filter_l_args,
                                                num_workers=24,
                                                batch_length=48000,
                                                method=method,
                                                task='f0',
                                                padding_single_length=padding_single_length,
                                                target_format=target_format,
                                                debug=debug_args,
                                                eval=eval_args,
                                                f0_window_size=512,
                                                egg_window_size=512,)
        
    if train_corpus == 'MOCHA' or all_flag == 'true':
        dp_args_train = ArgsDef.F0DataProcessArgs(locator=locator,
                                                corpus=locator.get_absolute_path('corpus/MOCHA_TIMIT'),
                                                dp_func_name = 'MOCHA_TIMIT',
                                                center=center,
                                                sr=sr,
                                                band=30,
                                                step_size=step_size,
                                                process=process, 
                                                add_filter_l_args=add_filter_l_args,
                                                num_workers=24,
                                                batch_length=48000,
                                                method=method,
                                                task='f0',
                                                padding_single_length=padding_single_length,
                                                target_format=target_format,
                                                debug=debug_args,
                                                eval=eval_args,
                                                f0_window_size=512,
                                                egg_window_size=512,)
        
        
    Data_list_train, dp_args_train = get_data(dp_args_train)
    # Data_list_train = Data_list_train[:2]
    train_set = CustomDataset_f0(Data_list_train, dp_args_train, device=device)
    
    
    dp_args_valid = ArgsDef.F0DataProcessArgs(locator=locator,
                                            corpus=locator.get_absolute_path('corpus/CSTR_FDA'),
                                            dp_func_name = 'CSTR_FDA',
                                            center=center,
                                            sr=sr,
                                            band=30,
                                            step_size=step_size,
                                            process=process, 
                                            add_filter_l_args=add_filter_l_args,
                                            num_workers=12,
                                            batch_length='original',
                                            method=method,
                                            task='f0',
                                            padding_single_length=padding_single_length,
                                            target_format=target_format,
                                            debug=debug_args,
                                            eval=eval_args,
                                            f0_window_size=512,
                                            egg_window_size=512,)
    
    Data_list_valid, dp_args_valid = get_data(dp_args_valid)
    valid_set = CustomDataset_f0(Data_list_valid, dp_args_valid, device=device)
    
    save_epochs = 5
    eval_epochs = 1
    lr = 1e-3
    opt_args = ArgsDef.optimizer_args_cls(optimizer_name='AdamW',
                                          k=5, # k折交叉验证
                                            batch_size=32,# 批处理大小
                                            eval_epochs=1,# 训练时评估频率
                                            save_epochs=save_epochs,# 保存频率
                                            num_train_epochs=20,# 训练轮数
                                            record_epochs=1,# 记录频率
                                            lr=lr,    # 初始学习率
                                            shuffle=True) # 是否打乱数据集
    
    if  use_gan == 'true':
        extra_gan = 'use_gan'
    else:
        extra_gan = 'no_gan'
    
    output_path, output_dir_of_one_fold, output_dir_of_one_fold_pic, output_dir_of_one_fold_pkl, output_dir_of_one_fold_pth = locator.get_output_folder(method=method, extra=f'{extra}_{train_corpus}_{extra_gan}', k=0, k_total=5, output_folder='EggCodec_output')
    
    
    plot_process = Process(target=asp.process_pkl_files_egg, args=(output_dir_of_one_fold_pkl, output_dir_of_one_fold_pic, 3))
    plot_process.start()

    ArgsDef.save_config(output_path,
                        extra_info='',
                        dp_args_train_0=dp_args_train,
                        dp_args_valid=dp_args_valid,
                        opt_args=opt_args,
                        args=args)
    train_dl = DataLoader(train_set, batch_size=14, shuffle=opt_args.shuffle)
    valid_dl = DataLoader(valid_set, batch_size=1, shuffle=False)

    
    train_dl_list = [train_dl]
    valid_dl_list = [valid_dl]
    

    
    target_bandwidths = [12., 24.]
    sample_rate = 16000
    channels = 1
    model = EncodecModel._get_model(
                target_bandwidths, sample_rate, channels,
                causal=False, model_norm='time_group_norm', audio_normalize=True,
                segment=None, name='my_encodec_16khz')
    model.train()
    model.train_quantization = True
    model.to(device)
    
    if use_gan == 'true':
        disc = MultiScaleSTFTDiscriminator(filters=32)
        disc.train()
        disc.to(device)
        

    
    if opt_args.optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=opt_args.lr)
    elif opt_args.optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=opt_args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    elif opt_args.optimizer_name == 'AdamW':
        optimizer = optim.Adam(model.parameters(), lr=opt_args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        if use_gan == 'true':
            optimizer_disc = optim.Adam(disc.parameters(), lr=opt_args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    else:
        raise ValueError(f'Optimizer {opt_args.optimizer_name} is not supported.')
    
    if opt_args.stepLR_step_size is not None and opt_args.srepLR_gamma is not None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt_args.stepLR_step_size, gamma=opt_args.srepLR_gamma)

    best_metric_train = 1e100
    best_metric_eval = 1e100
    start_epoch = 2

    record_loss_acc = RecordLossAcc(locator.join(output_dir_of_one_fold, 'epoch.png'))
    record_loss_acc.set_metric_options('loss', {'yscale': 'log'})
    train_d = False
    num_step = 0
    last_loss = 0
    start_time = time.time()
    for epoch in tqdm(range(1, opt_args.num_train_epochs + 1), desc="Epoch"):
        model.train()
        model.to(device)
        
        train_loss = 0.0
        train_accuracy_rpa = 0.0
        train_err_gpe = 0.0
        train_num_frames_loss = 0.0
        train_num_frames_metrix = 0.0
        
        

        for train_dl in train_dl_list:
            
            for i_batch, batch in enumerate(train_dl):
                train_d = not train_d
                
                wav_audio_ori = batch['wav_audio_ori']
                if use_noise == 'true':
                    wav_audio_ori = asp.add_noise(wav_audio_ori, snr_db)
                wav_egg_ori = batch['wav_egg_filter_ori']
                
                wav_audio_ori = wav_audio_ori.unsqueeze(1)
                wav_egg_ori = wav_egg_ori.unsqueeze(1)
                wav_audio_ori = wav_audio_ori.to(device)
                wav_egg_ori = wav_egg_ori.to(device)
                
                

                optimizer.zero_grad()
                
                
                output, loss_enc, _ = model(wav_audio_ori)
                
                
                loss_of_basic = basic_loss(wav1=wav_egg_ori, 
                                    wav2=output, 
                                    time_domain_loss=time_domain_loss, 
                                    freq_domain_loss=freq_domain_loss, 
                                    k_time_domain_loss=k_time_domain_loss, 
                                    k_freq_domain_loss=k_freq_domain_loss, 
                                    sample_rate=sr, device=device)
                
                if use_gan == 'true':
                    logits_real, fmap_real = disc(wav_egg_ori)
                    
                    if train_d:
                        with torch.no_grad():
                            fake_output = output.detach()  # 分离生成器的计算图
                        logits_fake_d, _ = disc(fake_output)
                        loss_of_disc = disc_loss(logits_real, logits_fake_d, device=device)
                        
                        if loss_of_disc > last_loss/2:
                            optimizer_disc.zero_grad()
                            loss_of_disc.backward()
                            optimizer_disc.step()
                        last_loss = 0
                    
                    logits_fake, fmap_fake = disc(output)
                    
                    loss_of_feature_match = feature_matching_loss(fmap_real, fmap_fake, device=device)
                    
                    loss_of_adv = adversarial_loss(logits_fake, device=device)
                    
                    loss = loss_of_basic + loss_of_feature_match + loss_of_adv
                
                else:
                    
                    loss = loss_of_basic
                    
                last_loss += loss.item()
                loss_enc.backward(retain_graph=True)
                loss.backward()
                optimizer.step()
                
                num_frames = int(wav_audio_ori.size(0) * wav_audio_ori.size(2) /(sr / step_size))
                train_loss_item = loss.item()
                train_num_frames_loss += num_frames
                train_loss += train_loss_item * num_frames

                
                if epoch % eval_epochs == 0 and epoch != 0:
                    train_accuracy_rpa, train_err_gpe, train_num_frames_metrix = np.nan, np.nan, np.nan
                    pass
                else:
                    train_accuracy_rpa, train_err_gpe, train_num_frames_metrix = np.nan, np.nan, np.nan
                    



                num_step += 1
            # end-for i_batch, batch in enumerate(train_dl):
        
        if opt_args.stepLR_step_size is not None and opt_args.srepLR_gamma is not None:
            scheduler.step()
            
        eval_loss = 0.0
        eval_mae = 0.0
        eval_accuracy_rpa = 0.0
        eval_err_gpe = 0.0
        eval_num_frames_loss = 0.0
        eval_num_frames_metrix = 0.0
        
        max_loss = 0
        min_loss = 1e10
        
        max_audio_ori = None
        max_egg_ori = None
        max_egg_rec = None
        
        min_audio_ori = None
        min_egg_ori = None
        min_egg_rec = None
        
        
        flag_eval_valid = True
        time_acc = 0.0
        for valid_dl in valid_dl_list:
            for i_batch, batch in enumerate(valid_dl):

                with torch.no_grad():
                    wav_audio_ori = batch['wav_audio_ori']
                    wav_egg_ori = batch['wav_egg_filter_ori']

                    wav_audio_ori = wav_audio_ori.unsqueeze(1)
                    wav_egg_ori = wav_egg_ori.unsqueeze(1)
                    
                    wav_audio_ori = wav_audio_ori.to(device)
                    wav_egg_ori = wav_egg_ori.to(device)

                    output, loss_enc, _ = model(wav_audio_ori)
                    loss = basic_loss(wav1=wav_egg_ori, 
                                      wav2=output, 
                                      time_domain_loss=time_domain_loss, 
                                      freq_domain_loss=freq_domain_loss, 
                                      k_time_domain_loss=k_time_domain_loss, 
                                      k_freq_domain_loss=k_freq_domain_loss, 
                                      sample_rate=sr, 
                                      device=device)
                    
                    if loss.item() > max_loss:
                        max_loss = loss.item()
                        max_audio_ori = wav_audio_ori
                        max_egg_ori = wav_egg_ori
                        max_egg_rec = output
                    
                    if loss.item() < min_loss:
                        min_loss = loss.item()
                        min_audio_ori = wav_audio_ori
                        min_egg_ori = wav_egg_ori
                        min_egg_rec = output
                        

                    num_frames = int(wav_audio_ori.size(0) * wav_audio_ori.size(2) /(sr / step_size))
                    
                    eval_loss_item = loss.item()
                    eval_loss += eval_loss_item * num_frames
                    eval_num_frames_loss += num_frames
                    eval_mae += np.abs(output.cpu().numpy().squeeze() - wav_egg_ori.cpu().numpy().squeeze()).sum()
                    
                    

                    if epoch % eval_epochs == 0 and epoch != 0:
                        start_time_ = time.time()
                        eval_accuracy_rpa_item, eval_err_gpe_item, eval_num, flag = asp.egg_rpa_and_gpe_timeout(output, batch, timeout=1.5, default_return=(0, num_frames, num_frames))
                        eval_accuracy_rpa += eval_accuracy_rpa_item
                        eval_err_gpe += eval_err_gpe_item
                        eval_num_frames_metrix += eval_num
                        time_cost_ = time.time() - start_time_
                        time_acc += time_cost_
                        if flag == False:
                            flag_eval_valid = False
                        
                    else:
                        eval_accuracy_rpa_item, eval_err_gpe_item, eval_num_frames_metrix = np.nan, np.nan, np.nan
        if flag_eval_valid == False:
            print('Eval invalid!')
        

        
        asp.encodec_record(max_audio_ori, max_egg_ori, max_egg_rec, max_audio_ori, max_audio_ori, sr=sr, output_dir=f'{output_dir_of_one_fold_pkl}/epoch{epoch}_step{num_step}_max.pkl')
        asp.encodec_record(min_audio_ori, min_egg_ori, min_egg_rec, min_audio_ori, min_audio_ori, sr=sr, output_dir=f'{output_dir_of_one_fold_pkl}/epoch{epoch}_step{num_step}_min.pkl')


        eval_loss /= eval_num_frames_loss + 1e-10
        eval_mae /= eval_num_frames_loss + 1e-10
        eval_accuracy_rpa /= eval_num_frames_metrix + 1e-10
        eval_err_gpe /= eval_num_frames_metrix + 1e-10
        


        train_loss /= train_num_frames_loss + 1e-10
        train_accuracy_rpa /= train_num_frames_metrix + 1e-10
        train_err_gpe /= train_num_frames_metrix + 1e-10


        
        record_loss_acc.append(epoch=epoch,
                               time_cost=time.time() - start_time,
                               metrics={
                                ('train', 'loss'): train_loss,
                                ('train', 'acc'): train_accuracy_rpa,
                                ('train', 'err'): train_err_gpe,
                                ('valid', 'loss'): eval_loss,
                                ('valid', 'acc'): eval_accuracy_rpa,
                                ('valid', 'err'): eval_err_gpe,
                                ('valid', 'mae'): eval_mae,
                               })
        


        if epoch % opt_args.record_epochs == 0:
            record_loss_acc.plot()
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Time: {time.time() - start_time:.2f}s - Epoch {epoch + 1}/{opt_args.num_train_epochs} - step {num_step} - Train loss: {train_loss:.2f} - Eval loss: {eval_loss:.2f} - Eval MAE: {eval_mae:.2f} - Eval acc: {eval_accuracy_rpa:.2f} - Eval err: {eval_err_gpe:.2f} - lr: {lr_now:.6e}", flush=True)


        if epoch % opt_args.save_epochs == 0 and epoch != 0:
            torch.save(model.state_dict(), f'{output_dir_of_one_fold_pth}/model_check_{epoch}.pth')
            print('Saving model...')
            
        if not np.isnan(eval_err_gpe) and eval_err_gpe < best_metric_eval and epoch > start_epoch:
            best_metric_eval = eval_err_gpe
            torch.save(model.state_dict(), f'{output_dir_of_one_fold_pth}/model_best_eval.pth')
            with open(os.path.join(output_dir_of_one_fold_pth, f'best_metric_eval.txt'), 'w') as f:
                f.write(f'epoch: {epoch}, best_metric_eval: {best_metric_eval:.2f}')
            print('Saving best model_eval...')



    torch.save(model.state_dict(), f'{output_dir_of_one_fold_pth}/model_finish.pth')
    time_cost = time.time() - start_time
    
    with open(os.path.join(output_dir_of_one_fold, f'time_cost.txt'), 'w') as f:
        f.write(f'Time cost: {time_cost:.2f}s')
    print('Saving model...')
    
    plot_process.terminate()
    plot_process.join()

