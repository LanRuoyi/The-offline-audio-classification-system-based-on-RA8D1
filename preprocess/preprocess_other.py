import os
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

def process_audio_files(input_folder, output_folder, sample_rates, pad_length, num_segments, segment_length):
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

                        # # 计算噪声幅值
                        # noise_amplitude = 0.07 * np.max(np.abs(segment))
                        # # 生成高斯噪声
                        # noise = np.random.normal(0, noise_amplitude, segment.shape)
                        # # 给片段添加高斯噪声
                        # noisy_segment = segment + noise

                        # 保存片段
                        segment_file_name = f'{file_name}_{i}.wav'
                        segment_file_path = os.path.join(sr_folder, segment_file_name)
                        sf.write(segment_file_path, segment, sr)

                        # # 保存片段
                        # noisy_segment_file_name = f'{file_name}_{i}_noise_07.wav'
                        # noisy_segment_file_path = os.path.join(sr_folder, noisy_segment_file_name)
                        # sf.write(noisy_segment_file_path, noisy_segment, sr)


if __name__ == "__main__":
    input_folder = 'unpreprocessed/other_2'  # 替换为实际的输入文件夹路径
    output_folder = 'preprocessed_2/other'  # 替换为实际的输出文件夹路径
    sample_rates = [8094]  # 三种采样率
    pad_length = 2    # 补零时长
    num_segments = 1  # 片段数
    segment_length = 32400  # 每个片段的采样点数

    process_audio_files(input_folder, output_folder, sample_rates, pad_length, num_segments, segment_length)