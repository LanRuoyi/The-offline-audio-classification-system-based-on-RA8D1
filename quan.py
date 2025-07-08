import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pickle as pkl
from tqdm import tqdm
import os
root = "./train_runs/.../"

# 1. 加载H5模型
model = tf.keras.models.load_model(os.path.join(root, 'test_model.h5'), compile=False)

# 2. 创建TFLite转换器
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. 配置量化参数
converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("模型输入类型:", model.input.shape)
# 定义代表数据集（需替换为真实数据）
def representative_dataset():
    with open('./train_img.pkl', 'rb') as f:
        train_img = pkl.load(f)
    train_img = np.array(train_img)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_img).batch(1).take(1000)
    for data in tqdm(train_dataset):
        # print(data.shape)
        yield [tf.dtypes.cast(data, tf.float32)]

converter.representative_dataset = representative_dataset

# 设置全整数量化参数
converter.inference_input_type = tf.int8   # 或 tf.uint8
converter.inference_output_type = tf.int8  # 或 tf.uint8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# 启用新版量化器（可选，根据TF版本调整）
converter.experimental_new_quantizer = True

# 4. 转换并保存模型
tflite_quant_model = converter.convert()
with open(os.path.join(root, './quantized_model.tflite'), 'wb') as f:
    f.write(tflite_quant_model)

print("模型已成功转换并量化！")


# ---------------------- 新增：验证TFLite模型准确率 ----------------------
# 加载测试数据（根据你的实际数据路径调整）
with open('./test_img.pkl', 'rb') as f:
    test_img = np.array(pkl.load(f))  # 转为numpy数组
with open('./test_labels.pkl', 'rb') as f:
    test_labels = np.array(pkl.load(f))  # 转为numpy数组

# 初始化TFLite解释器
interpreter = tf.lite.Interpreter(model_path=os.path.join(root, './quantized_model.tflite'))
interpreter.allocate_tensors()

# 获取输入输出张量信息
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 统计正确预测数
correct_predictions = 0
total_samples = len(test_labels)

# 使用tqdm创建进度条（保存为变量以便动态更新）
pbar = tqdm(zip(test_img, test_labels), total=total_samples, desc="验证中")

# 遍历测试数据（通过enumerate获取当前迭代次数）
for i, (img, true_label) in enumerate(pbar):
    # 预处理输入（匹配模型输入要求）
    img = img.astype(np.float32)  # 转为float32（若原始数据是其他类型）
    img = np.expand_dims(img, axis=0)  # 添加批次维度（假设模型输入为[1, H, W, C]）

    # 量化输入（若模型要求int8输入）
    if input_details[0]['dtype'] == np.int8:
        input_scale, input_zero_point = input_details[0]['quantization']
        img_quant = img / input_scale + input_zero_point  # 应用量化参数
        img_quant = img_quant.astype(np.int8)  # 转为int8
        interpreter.set_tensor(input_details[0]['index'], img_quant)
    else:
        interpreter.set_tensor(input_details[0]['index'], img)

    # 执行推理
    interpreter.invoke()

    # 获取输出并反量化（若输出是int8）
    output = interpreter.get_tensor(output_details[0]['index'])
    if output_details[0]['dtype'] == np.int8:
        output_scale, output_zero_point = output_details[0]['quantization']
        output = (output - output_zero_point) * output_scale  # 反量化为float

    # 计算预测标签（假设是分类任务，取最大值索引）
    pred_label = np.argmax(output)
    if pred_label == true_label:
        correct_predictions += 1

    # 实时计算当前准确率（已处理i+1个样本）
    current_accuracy = correct_predictions / (i + 1)
    # 更新进度条后缀显示实时准确率
    pbar.set_postfix(当前准确率=f"{current_accuracy * 100:.2f}%")

# 最终准确率（与实时显示的最终值一致）
accuracy = correct_predictions / total_samples
print(f"\n最终量化模型准确率: {accuracy * 100:.2f}%")


