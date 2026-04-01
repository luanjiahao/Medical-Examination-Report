from flask import Flask, request, jsonify
import sys
import os

# 确保可以导入 main 模块
sys.path.append(os.path.dirname(__file__))

from main import load_model, predict_single, create_dataset

app = Flask(__name__)

# 全局变量加载模型
model = None
label_encoder = None
input_size = 9
output_size = 5


def initialize_model():
    """初始化模型"""
    global model, label_encoder, input_size, output_size
    print("正在加载模型...")
    # 加载数据集获取标签编码器
    train_dataset, _, input_size, output_size, label_encoder = create_dataset()
    # 加载训练好的模型
    model = load_model(input_size, output_size)
    print("模型加载完成！")


# 在应用启动时立即加载模型
initialize_model()


@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '请提供 JSON 格式的数据'}), 400

        result = predict_single(data, model, label_encoder, input_size, output_size)

        if result:
            return jsonify(result)
        else:
            return jsonify({'error': '预测失败'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({'status': 'ok', 'message': '服务运行正常'})


if __name__ == '__main__':
    print("=" * 60)
    print("体检报告预测 API 服务")
    print("=" * 60)
    print(f"服务地址：http://localhost:5000")
    print(f"预测接口：POST http://localhost:5000/predict")
    print(f"健康检查：GET http://localhost:5000/health")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)
