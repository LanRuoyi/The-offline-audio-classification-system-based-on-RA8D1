import os
import shutil
import pandas as pd
from tqdm import tqdm
# 读取CSV文件
df = pd.read_csv('./audio/esc50.csv')  # 确保esc50.csv在当前目录或提供完整路径

# 筛选出分类不是'crying_baby'的行
non_crying_baby_df = df[df['category'] != 'crying_baby']

# 创建目标文件夹（如果不存在）
output_dir = 'non_crying_baby_audios'
os.makedirs(output_dir, exist_ok=True)

# 原始音频文件所在的文件夹路径（根据实际情况修改）
audio_dir = 'audio'  # 假设音频文件在audio文件夹中

# 复制每个非crying_baby的音频文件到新文件夹
for filename in tqdm(non_crying_baby_df['filename']):
    src_path = os.path.join(audio_dir, filename)
    dst_path = os.path.join(output_dir, filename)
    
    # 检查源文件是否存在
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)  # 保留元数据的复制
        print(f"已复制: {filename}")
    else:
        print(f"文件不存在，跳过: {filename}")

print(f"操作完成！共复制了 {len(non_crying_baby_df)} 个音频文件到 '{output_dir}' 文件夹。")