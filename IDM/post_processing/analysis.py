import matplotlib.pyplot as plt
import pandas as pd

def draw_distribution(df):
    # 绘制柱状图

    # 为每列创建柱状图
    n_groups = 4  # 列的数量
    names = ['w', 's', 'a', 'd']
    fig, axes = plt.subplots(n_groups, 1, figsize=(10, 8))  # 创建一个子图网格

    for i, name in enumerate(names):
        counts = df[name].value_counts().sort_index()
        # 绘制柱状图
        counts.plot(kind='bar', ax=axes[i], title=f'Value counts for {name}')
        axes[i].set_xticklabels(counts.index, rotation=45)  # 旋转x轴标签以便阅读

    # 调整子图间距
    plt.tight_layout()
    plt.show()

    # 设置X轴的标签
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Frequency Distribution')

    # 显示图例
    plt.legend(names)

    # 显示图表
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('../data/labels_interval-1_dirty-5.0.csv')
    draw_distribution(df)