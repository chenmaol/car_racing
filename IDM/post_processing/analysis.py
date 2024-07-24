import matplotlib.pyplot as plt
import pandas as pd

def draw_distribution(df):
    f1 = df['feature_1'].values
    f2 = df['feature_2'].values

    # 绘制柱状图
    kwargs = dict(histtype='stepfilled', alpha=0.3, bins=40)
    plt.hist(f1, **kwargs)
    plt.hist(f2, **kwargs)

    # 设置X轴的标签
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Frequency Distribution')

    # 显示图例
    plt.legend(['f1', 'f2'])

    # 显示图表
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('../data/labels_interval-3_dirty-5.0.csv')
    draw_distribution(df)