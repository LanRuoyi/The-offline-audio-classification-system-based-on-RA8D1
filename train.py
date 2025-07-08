import os
import librosa
import numpy as np
import pickle as pkl
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa
from tqdm import tqdm  # 训练进度条
from keras import layers
import matplotlib.pyplot as plt
import datetime

model_6 = tf.keras.Sequential([
    # Block 1
    layers.Conv2D(8, (3, 3), padding='same', activation='relu', input_shape=(64, 64, 1)),
    layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Block 2
    layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Block 3
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),

    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),

    layers.GlobalAvgPool2D(),

    # Classifier
    layers.Dense(6)
])

model_2 = tf.keras.Sequential([
    # Block 1
    layers.Conv2D(8, (3, 3), padding='same', activation='relu', input_shape=(64, 64, 1)),
    # layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Block 2
    layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
    # layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Block 3
    layers.SeparableConv2D(32, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.SeparableConv2D(64, (3, 3), padding='same', activation='relu'),

    layers.GlobalAvgPool2D(),

    # Classifier
    layers.Dense(2)
])


# 定义函数将模型 summary 保存到文件
def save_model_summary(model, summary_path):
    with open(summary_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
# 自定义回调函数，用于在每个 epoch 结束时显示学习率
class ShowLearningRate(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # 获取当前优化器的学习率
        lr = self.model.optimizer.lr
        # 如果学习率是一个调度器，需要计算当前步数对应的学习率
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            step = self.model.optimizer.iterations
            lr = lr(step)
        print(f'\nEpoch {epoch + 1} 结束时的学习率: {lr.numpy()}')

def audio_to_mel_spec(audio_path, target_length=6.4, sr=17600, n_fft=2048, hop_length=512, n_mels=56, resize=True):
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=sr)
    audio = audio.astype(np.float32)
    normalization_factor = 1 / np.max(np.abs(audio))
    audio = audio * normalization_factor


    audio = (audio * np.random.randint(500, 2047)).astype(np.int16)
    audio = audio.astype(np.float32)


    normalization_factor = 1 / np.max(np.abs(audio))
    audio = audio * normalization_factor

    # print(audio)

    # 生成梅尔频谱图
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, power=1, n_mels=n_mels)
    mel_spec = mel_spec.T
    # print(mel_spec.shape)

    # 将像素值归一化到[0, 1]
    normalized_img = (mel_spec - np.min(mel_spec)) / (np.max(mel_spec) - np.min(mel_spec))

    return normalized_img[:, :, np.newaxis]  # 增加通道维度


if __name__ == "__main__":
    model = model_2 # model_6
    model.build(input_shape=(None, 64, 64, 1))
    model.summary()

    NUM_EPOCHS = 300

    all_file = []

    root = 'preprocessed_2' # 'preprocessed_6'

    train_img = []
    train_labels = []
    test_img = []
    test_labels = []

    folds = os.listdir(root)
    for idx, fold in enumerate(["awake","diaper","hug","hungry","sleepy","uncomfortable"]):  # ["baby","other"]
        class_root = os.path.join(root, fold)
        for sr in [8094]:  #[8000, 8100, 8200]
            class_sr_root = os.path.join(class_root, f'{sr}')
            files = os.listdir(class_sr_root)
            for file in files:
                file_path = os.path.join(class_sr_root, file)
                # print((file_path, int(file_path.split('-')[1])))
                all_file.append(((file_path, sr), idx))
    train_files, test_files = train_test_split(all_file, test_size=0.2, random_state=30)
    for file, label in tqdm(train_files):
        feature = audio_to_mel_spec(file[0], target_length=4, sr=file[1], n_fft=1024, hop_length=512, n_mels=64)
        train_img.append(feature)
        train_labels.append(label)
        feature = audio_to_mel_spec(file[0], target_length=4, sr=file[1], n_fft=1024, hop_length=512, n_mels=64)
        train_img.append(feature)
        train_labels.append(label)
        feature = audio_to_mel_spec(file[0], target_length=4, sr=file[1], n_fft=1024, hop_length=512, n_mels=64)
        train_img.append(feature)
        train_labels.append(label)
    for file, label in tqdm(test_files):
        feature = audio_to_mel_spec(file[0], target_length=4, sr=file[1], n_fft=1024, hop_length=512, n_mels=64)
        test_img.append(feature)
        test_labels.append(label)
    
    with open('./train_img.pkl', 'wb') as f:
        pkl.dump(train_img, f)
    with open('./train_labels.pkl', 'wb') as f:
        pkl.dump(train_labels, f)
    with open('./test_img.pkl', 'wb') as f:
        pkl.dump(test_img, f)
    with open('./test_labels.pkl', 'wb') as f:
        pkl.dump(test_labels, f)
    del train_img, train_labels, test_img, test_labels
    """"""
    with open('./train_img.pkl', 'rb') as f:
        train_img = pkl.load(f)
    with open('./train_labels.pkl', 'rb') as f:
        train_labels = pkl.load(f)
    with open('./test_img.pkl', 'rb') as f:
        test_img = pkl.load(f)
    with open('./test_labels.pkl', 'rb') as f:
        test_labels = pkl.load(f)
    print(f"train num {len(train_labels)}")
    print(f"test num {len(test_labels)}")
    train_img, train_labels = np.array(train_img), np.array(train_labels)
    test_img, test_labels = np.array(test_img), np.array(test_labels)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_img, train_labels)).shuffle(256000).batch(256) # .shuffle(128000).
    test_dataset = tf.data.Dataset.from_tensor_slices((test_img, test_labels)).batch(256)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimizer = tfa.optimizers.AdamW(weight_decay=1e-4)

    model.compile(
        optimizer=optimizer,
        loss=loss_object,
        metrics=['accuracy']
    )
    show_lr = ShowLearningRate()

    # 创建新的训练文件夹
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_folder = os.path.join('train_runs', timestamp)
    os.makedirs(train_folder, exist_ok=True)

    # 定义模型 summary 保存路径
    summary_path = os.path.join(train_folder, 'model_summary.txt')
    # 保存模型 summary
    save_model_summary(model, summary_path)

    # 更新模型保存路径和日志文件路径
    model_path = os.path.join(train_folder, 'test_model.h5')
    log_path = os.path.join(train_folder, 'training_log.csv')
    plot_path = os.path.join(train_folder, 'training_history.png')

    # 模型保存回调
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=50)
    # 添加 CSVLogger 回调
    csv_logger = CSVLogger(log_path, separator=',', append=False)

    del train_img, train_labels, test_img, test_labels
    print('preprocess finished')
    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=NUM_EPOCHS,
        callbacks=[checkpoint, csv_logger, ], # show_lr
    )
    print('train finished')
    # 可选：加载最佳模型进行验证
    best_model = tf.keras.models.load_model(model_path)
    val_loss, val_acc = best_model.evaluate(test_dataset)

     # 绘制折线图
    history = np.genfromtxt(log_path, delimiter=',', names=True)
    epochs = range(1, len(history) + 1)

    # 绘制损失曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    print(train_folder)
    # plt.show()
