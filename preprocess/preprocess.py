import os
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
from scipy.signal import butter, filtfilt  # 导入带通滤波所需库
import random

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    设计带通滤波器。

    :param lowcut: 低频截止频率
    :param highcut: 高频截止频率
    :param fs: 采样率
    :param order: 滤波器阶数
    :return: 滤波器系数
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    应用带通滤波器到数据上。

    :param data: 输入数据
    :param lowcut: 低频截止频率
    :param highcut: 高频截止频率
    :param fs: 采样率
    :param order: 滤波器阶数
    :return: 滤波后的数据
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def process_audio_files(input_folder, output_folder, sample_rates, pad_length, num_segments, segment_length, lowcut=2000, highcut=3000):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for root, dirs, files in os.walk(input_folder):
        for file in tqdm(files):
            if file.lower().endswith(('.wav', '.mp3', '.flac')):
                file_path = os.path.join(root, file)
                file_name = os.path.splitext(file)[0]

                for sr in sample_rates:
                    # 创建对应采样率的输出子文件夹
                    sr_folder = os.path.join(output_folder, f'{sr}')
                    if not os.path.exists(sr_folder):
                        os.makedirs(sr_folder)

                    # 以指定采样率读取音频文件
                    audio, _ = librosa.load(file_path, sr=sr)

                    # 在音频的开头和结尾补零
                    padded_audio = np.pad(audio, (pad_length*sr, pad_length*sr), 'constant', constant_values=0)

                    # 计算可截取的最大起始位置
                    max_start = len(padded_audio) - segment_length

                    for i in range(num_segments):
                        # 随机选择起始位置
                        if max_start > 0:
                            start = np.random.randint(0, max_start)
                        else:
                            start = 0
                        end = start + segment_length

                        # 截取片段
                        segment = padded_audio[start:end]

                        if np.max(np.abs(segment)) == 0:
                            break

                        # 生成带通噪声
                        noise = np.random.normal(0, 1, segment.shape)
                        bandpass_noise = butter_bandpass_filter(noise, lowcut+random.randint(-500, 500), highcut+random.randint(-500, 500), sr)
                        noise_amplitude = 0.02 * np.max(np.abs(segment))
                        bandpass_noise = noise_amplitude * bandpass_noise
                        band_noisy_segment = segment + bandpass_noise

                        # 计算噪声幅值
                        noise_amplitude = 0.02 * np.max(np.abs(segment))
                        # 生成高斯噪声
                        noise = np.random.normal(0, noise_amplitude, segment.shape)
                        # 给片段添加高斯噪声
                        noisy_segment = segment + noise

                        # 保存片段
                        segment_file_name = f'{file_name}_{i}.wav'
                        segment_file_path = os.path.join(sr_folder, segment_file_name)
                        sf.write(segment_file_path, segment, sr)

                        # 保存片段
                        noisy_segment_file_name = f'{file_name}_{i}_noise_02.wav'
                        noisy_segment_file_path = os.path.join(sr_folder, noisy_segment_file_name)
                        sf.write(noisy_segment_file_path, noisy_segment, sr)

                        # 保存片段
                        noisy_segment_file_name = f'{file_name}_{i}_bnoise_02.wav'
                        noisy_segment_file_path = os.path.join(sr_folder, noisy_segment_file_name)
                        sf.write(noisy_segment_file_path, band_noisy_segment, sr)

                        # 计算噪声幅值
                        noise_amplitude = 0.2 * np.max(np.abs(segment))
                        # 生成高斯噪声
                        noise = np.random.normal(0, noise_amplitude, segment.shape)
                        # 给片段添加高斯噪声
                        noisy_segment = segment + noise

                        # 生成带通噪声
                        noise = np.random.normal(0, 1, segment.shape)
                        bandpass_noise = butter_bandpass_filter(noise, lowcut+random.randint(-500, 500), highcut+random.randint(-500, 500), sr)
                        noise_amplitude = 0.2 * np.max(np.abs(segment))
                        bandpass_noise = noise_amplitude * bandpass_noise
                        band_noisy_segment = segment + bandpass_noise

                        # 保存片段
                        noisy_segment_file_name = f'{file_name}_{i}_noise_2.wav'
                        noisy_segment_file_path = os.path.join(sr_folder, noisy_segment_file_name)
                        sf.write(noisy_segment_file_path, noisy_segment, sr)

                        # 保存片段
                        noisy_segment_file_name = f'{file_name}_{i}_bnoise_2.wav'
                        noisy_segment_file_path = os.path.join(sr_folder, noisy_segment_file_name)
                        sf.write(noisy_segment_file_path, band_noisy_segment, sr)

                        # 计算噪声幅值
                        noise_amplitude = random.uniform(0, 0.3) * np.max(np.abs(segment))
                        # 生成高斯噪声
                        noise = np.random.normal(0, noise_amplitude, segment.shape)
                        # 给片段添加高斯噪声
                        noisy_segment = segment + noise

                        # 生成带通噪声
                        noise = np.random.normal(0, 1, segment.shape)
                        bandpass_noise = butter_bandpass_filter(noise, lowcut+random.randint(-500, 500), highcut+random.randint(-500, 500), sr)
                        noise_amplitude = random.uniform(0, 0.3) * np.max(np.abs(segment))
                        bandpass_noise = noise_amplitude * bandpass_noise
                        band_noisy_segment = segment + bandpass_noise

                        # 保存片段
                        noisy_segment_file_name = f'{file_name}_{i}_rnoise_2.wav'
                        noisy_segment_file_path = os.path.join(sr_folder, noisy_segment_file_name)
                        sf.write(noisy_segment_file_path, noisy_segment, sr)

                        # 保存片段
                        noisy_segment_file_name = f'{file_name}_{i}_rbnoise_2.wav'
                        noisy_segment_file_path = os.path.join(sr_folder, noisy_segment_file_name)
                        sf.write(noisy_segment_file_path, band_noisy_segment, sr)


if __name__ == "__main__":
    input_folder = 'unpreprocessed'  # 替换为实际的输入文件夹路径
    output_folder = 'preprocessed_2'  # 替换为实际的输出文件夹路径
    sample_rates = [8094]  # 三种采样率
    pad_length = 2    # 补零时长
    num_segments = 2  # 片段数
    segment_length = 32400  # 每个片段的采样点数

    
    for fold in ["awake","diaper","hug","hungry","sleepy","uncomfortable"]:
        class_root = os.path.join(input_folder, fold)
        pre_class_root = os.path.join(output_folder, fold)
        process_audio_files(class_root, pre_class_root, sample_rates, pad_length, num_segments, segment_length)