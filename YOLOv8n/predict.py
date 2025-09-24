from ultralytics import YOLO

# 配置参数
model_path = 'runs/detect/train12/weights/best.pt'  # 训练好的模型路径
image_path = 'ultralytics/assets/guowang1.jpg'  # 单张图片路径
output_dir = 'runs/predict/'  # 保存预测结果的目录

# 加载训练好的 YOLO 模型
model = YOLO(model_path)

# 进行预测
results = model.predict(source=image_path, save=True, save_dir=output_dir)

# 打印预测结果
# print("预测完成！以下是详细结果：")
# print("边界框 (中心点 x, y, 宽度, 高度)：")
# print(results.pandas().xywh)  # 输出预测的边界框坐标
# print("\n分类标签名称：")
# print(results.names)  # 输出类别名称
# print("\n置信度分数：")
# print(results.pandas().confidence)  # 输出置信度分数
#
# # 如果需要批量预测，可以指定图片文件夹
# batch_source = 'ultralytics/assets/images/'  # 批量图片路径
# batch_results = model.predict(source=batch_source, save=True, save_dir=output_dir)
# print(f"\n批量预测完成！结果保存到 {output_dir}")
