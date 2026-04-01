import torch  # 导入 PyTorch 深度学习框架
from torch.utils.data import TensorDataset  # 从 PyTorch 工具包中导入张量数据集类
from torch.utils.data import DataLoader  # 从 PyTorch 工具包中导入数据加载器类
import torch.nn as nn  # 导入 PyTorch 神经网络模块
import torch.optim as optim  # 导入 PyTorch 优化算法模块
from sklearn.model_selection import train_test_split  # 从 scikit-learn 导入训练集和测试集划分函数
from sklearn.preprocessing import LabelEncoder  # 导入标签编码器
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score  # 导入评估指标
import matplotlib.pyplot as plt  # 导入 matplotlib 绘图库
import numpy as np  # 导入 NumPy 科学计算库
import pandas as pd  # 导入 pandas 数据处理库
import time  # 导入时间模块
import os  # 导入操作系统接口模块
import seaborn as sns  # 导入 seaborn 用于绘制混淆矩阵


# todo 1. 定义函数，构建数据集
def create_dataset():
    data = pd.read_csv('./data/体检报告.csv')
    
    # 处理缺失值：删除含有缺失值的行或使用填充
    data = data.dropna()
    
    # 获取 x 特征列和 y 标签列
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    # 把特征列转换为浮点数
    x = x.astype(np.float32)

    # 对标签进行编码，将字符串转换为整数
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # 保存标签编码器以便后续使用
    print(f"标签类别：{label_encoder.classes_}")

    # 切分训练集和测试集
    # 参数 1：特征列 参数 2：标签列 参数 3：测试集占比 参数 4：随机种子 参数 5：样本分布（参考 y 的类别抽取数据）
    x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=23, stratify=y_encoded)

    # 创建张量数据集，使用 np.array() 确保转换为数组
    train_dataset = TensorDataset(torch.tensor(np.array(x_train)), torch.tensor(np.array(y_train)))
    test_dataset = TensorDataset(torch.tensor(np.array(x_test)), torch.tensor(np.array(y_test)))

    # 返回数据集                           9(充当输入特征数）5（输出标签数）(0,1,2,3)
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y_encoded)), label_encoder


# todo 2. 搭建神经网络
class PhonePriceModel(nn.Module):
    # 1.在init方法中初始化父类成员，搭建神经网络
    def __init__(self, input_size, output_size):
        # 初始父类成员
        super().__init__()
        # 隐藏层1
        self.linear1 = nn.Linear(input_size, 128)
        # 隐藏层2
        self.linear2 = nn.Linear(128, 256)
        # 输出层
        self.output = nn.Linear(256, output_size)

    # 前向传播
    def forward(self, x):
        # 隐藏层1：加权求和+激活函数
        x = torch.relu(self.linear1(x))
        # 隐藏层2：加权求和+激活函数
        x = torch.relu(self.linear2(x))
        # 输出层：加权求和
        x = self.output(x)
        return x


# todo 3. 模型训练
def train(train_dataset, input_size, output_size):
    # 创建数据加载器，流程：数据集 -> 数据加载器 -> 模型 -> 损失函数 -> 优化器 -> 训练
    # 参数 1：数据集 16000 条 2：每批 16 条数据 3：是否打乱数据
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    model = PhonePriceModel(input_size, output_size)
    # 定义损失函数，多分类交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器，SGD 优化算法
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # 模型训练
    epochs = 50
    # 开始训练
    for epoch in range(epochs):
        # 定义变量，记录损失值，训练批次数
        total_loss, batch_num = 0.0, 0
        # 定义变量表示开始时间
        start_time = time.time()
        # 开始训练
        for x, y in train_loader:
            # 切换模型（状态）
            model.train()
            # 模型预测
            y_pred = model(x)
            # 计算损失值
            loss = criterion(y_pred, y)
            # 反向传播，梯度清零，优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 累计损失值
            total_loss += loss.item()  # 把本轮平均损失值加到累计损失值中
            batch_num += 1
            # 打印损失值
            print(
                f'第{epoch + 1}轮，第{batch_num}批，损失值：{total_loss / batch_num:.4f},用时：{time.time() - start_time:.2f}秒')

        # 保存模型
        # 确保模型目录存在
        os.makedirs('./model', exist_ok=True)
        torch.save(model.state_dict(), './model/predict.pth')


# todo 4. 模型测试
def test(test_dataset, input_size, output_size, label_encoder):
    # 创建神经网络分类对象
    model = PhonePriceModel(input_size, output_size)
    # 加载模型参数
    model.load_state_dict(torch.load('./model/predict.pth', weights_only=True))
    # 创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    # 定义变量 记录预测正确的样本
    correct = 0
    
    # 收集所有预测结果和真实标签
    all_preds = []
    all_labels = []
    
    # 从数据加载器中获取数据
    for x, y in test_loader:
        # 模型切换为测试模式
        model.eval()
        # 模型预测
        with torch.no_grad():
            y_pred = model(x)
        # 根据加权求和，得到类别，argmax 找最大值对应的索引
        y_pred = torch.argmax(y_pred, dim=1)
        
        # 收集预测结果和真实标签
        all_preds.extend(y_pred.numpy())
        all_labels.extend(y.numpy())
        
        # 统计预测正确样本个数
        correct += (y_pred == y).sum().item()

    # 打印准确率
    print(f'测试集准确率：{correct / len(test_dataset):.4f}')
    
    # 计算并打印详细的分类报告
    print("\n" + "="*60)
    print("详细分类报告:")
    print("="*60)
    target_names = label_encoder.classes_
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))
    
    # 计算宏观和加权平均指标
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    precision_weighted = precision_score(all_labels, all_preds, average='weighted')
    recall_weighted = recall_score(all_labels, all_preds, average='weighted')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    print("="*60)
    print("整体评估指标:")
    print("="*60)
    print(f"准确率 (Accuracy):  {accuracy:.4f}")
    print(f"精确率 (Precision - Macro):    {precision_macro:.4f}")
    print(f"召回率 (Recall - Macro):       {recall_macro:.4f}")
    print(f"F1 分数 (F1-Score - Macro):      {f1_macro:.4f}")
    print(f"\n精确率 (Precision - Weighted): {precision_weighted:.4f}")
    print(f"召回率 (Recall - Weighted):    {recall_weighted:.4f}")
    print(f"F1 分数 (F1-Score - Weighted):   {f1_weighted:.4f}")

# todo 5. 实际预测函数（供 Java 系统调用）
def predict_single(health_data, model, label_encoder, input_size, output_size):
    """
    对单个体检报告数据进行疾病预测
    
    参数:
        health_data: 字典或列表，包含体检指标数据
                    如果是字典：{'血糖': 107.38, '长期血糖水平': 4.93, ...}
                    如果是列表：[107.38, 4.93, 109.25, 74.1, 129.2, 52.11, 68.84, 10.17, 61.54]
        model: 训练好的模型
        label_encoder: 标签编码器
        input_size: 输入特征数量
        output_size: 输出类别数量
    
    返回:
        dict: 包含预测结果的字典
              {
                  'predicted_class': 'Fit',  # 预测类别
                  'predicted_class_id': 2,   # 预测类别编号
                  'probabilities': {...},    # 各类别概率
                  'confidence': 0.85         # 置信度（最高概率）
              }
    """
    # 确保模型处于评估模式
    model.eval()
    
    # 定义特征列的顺序（必须与训练时一致）
    feature_columns = ['血糖', '长期血糖水平', '测量血压的最高值', '测量血压的最低值', 
                       '低密度脂蛋白', '高密度脂蛋白', '甘油三酯 ', '血红蛋白', '平均红细胞体积']
    
    try:
        # 如果输入是字典，转换为列表
        if isinstance(health_data, dict):
            data_list = [float(health_data[col]) for col in feature_columns]
        else:
            data_list = [float(x) for x in health_data]
        
        # 转换为 tensor
        input_tensor = torch.tensor([data_list], dtype=torch.float32)
        
        # 模型预测
        with torch.no_grad():
            output = model(input_tensor)
            # 使用 softmax 获取概率
            probabilities = torch.nn.functional.softmax(output, dim=1)
            # 获取预测类别
            confidence, predicted = torch.max(probabilities, 1)
        
        # 获取所有类别的概率
        all_probs = probabilities[0].numpy()
        prob_dict = {label_encoder.classes_[i]: float(all_probs[i]) 
                     for i in range(len(label_encoder.classes_))}
        
        # 构建返回结果
        result = {
            'predicted_class': label_encoder.classes_[predicted.item()],
            'predicted_class_id': predicted.item(),
            'probabilities': prob_dict,
            'confidence': float(confidence.item())
        }
        
        return result
        
    except Exception as e:
        print(f"预测出错：{str(e)}")
        return None

# todo 6. 批量预测函数
def predict_batch(health_data_list, model, label_encoder, input_size, output_size):
    """
    批量预测多个体检报告数据
    
    参数:
        health_data_list: 列表，包含多个体检报告数据
                         每个元素可以是字典或列表
        model: 训练好的模型
        label_encoder: 标签编码器
        input_size: 输入特征数量
        output_size: 输出类别数量
    
    返回:
        list: 包含所有预测结果的列表
    """
    model.eval()
    results = []
    
    for health_data in health_data_list:
        result = predict_single(health_data, model, label_encoder, input_size, output_size)
        results.append(result)
    
    return results

# todo 7. 加载模型工具函数
def load_model(input_size, output_size, model_path='./model/predict.pth'):
    """
    加载训练好的模型
    
    参数:
        input_size: 输入特征数量
        output_size: 输出类别数量
        model_path: 模型文件路径
    
    返回:
        model: 加载好参数的模型
    """
    model = PhonePriceModel(input_size, output_size)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

if __name__ == '__main__':  # 程序主入口
    train_dataset, test_dataset, input_size, output_size, label_encoder = create_dataset()

    train(train_dataset, input_size, output_size)

    test(test_dataset, input_size, output_size, label_encoder)
    
    # ========== 示例：如何使用预测函数 ==========
    print("\n" + "="*60)
    print("实际预测示例:")
    print("="*60)
    
    # 加载模型
    model = load_model(input_size, output_size)
    
    # 示例 1: 使用字典格式的数据进行预测
    sample_data_dict = {
        '血糖': 107.38,
        '长期血糖水平': 4.93,
        '测量血压的最高值': 109.25,
        '测量血压的最低值': 74.1,
        '低密度脂蛋白': 129.2,
        '高密度脂蛋白': 52.11,
        '甘油三酯 ': 68.84,
        '血红蛋白': 10.17,
        '平均红细胞体积': 61.54
    }
    
    print("\n示例 1 - 字典格式数据预测:")
    result = predict_single(sample_data_dict, model, label_encoder, input_size, output_size)
    if result:
        print(f"预测结果：{result['predicted_class']}")
        print(f"置信度：{result['confidence']:.4f}")
        print(f"各类别概率:")
        for cls, prob in result['probabilities'].items():
            print(f"  {cls}: {prob:.4f}")
    
    # 示例 2: 使用列表格式的数据进行预测
    sample_data_list = [108.65, 5.43, 92.61, 62.02, 51.18, 44.06, 78.82, 12.29, 91.04]
    
    print("\n示例 2 - 列表格式数据预测:")
    result = predict_single(sample_data_list, model, label_encoder, input_size, output_size)
    if result:
        print(f"预测结果：{result['predicted_class']}")
        print(f"置信度：{result['confidence']:.4f}")
    
    # 示例 3: 批量预测
    print("\n示例 3 - 批量预测:")
    batch_data = [
        [107.38, 4.93, 109.25, 74.1, 129.2, 52.11, 68.84, 10.17, 61.54],
        [108.65, 5.43, 92.61, 62.02, 51.18, 44.06, 78.82, 12.29, 91.04],
        [76.57, 6.26, 164.53, 93.88, 111.2, 45.5, 87.27, 13.8, 87.86]
    ]
    batch_results = predict_batch(batch_data, model, label_encoder, input_size, output_size)
    for i, result in enumerate(batch_results):
        if result:
            print(f"样本{i+1}: {result['predicted_class']} (置信度：{result['confidence']:.4f})")

