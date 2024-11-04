import pandas as pd
import matplotlib.pyplot as plt

# 读取导出的 CSV 数据
data = pd.read_csv("latest_dreamer_run.csv")

# 绘制训练损失和验证损失随迭代次数变化的图表
plt.figure(figsize=(10, 5))

# 假设数据中有 'epoch', 'navigation', 和 'val_loss' 数据
if '_step' in data.columns and 'navigation' in data.columns:
    plt.plot(data['navigation'],
             label='Training Loss', color='blue')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.savefig('training_validation_loss.png')
else:
    print("Data does not contain required columns: 'epoch', 'navigation', 'val_loss'.")
