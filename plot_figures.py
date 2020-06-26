#! C:\Users\93715\Anaconda3\python.exe
# *-* coding:utf8 *-*
"""
@author: LiuZhe
@license: (C) Copyright SJTU ME
@contact: LiuZhe_54677@sjtu.edu.cn
@file: plot_figures.py
@time: 2020/6/15 11:24
@desc: 人生苦短，我用python
"""
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 导入中文字体
font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=15)


def plot_figure(ind_var, dep_var, path, x_label: str, y_label: str, title: str, num: int = 1, is_show=True, ):
    # 画出原始图像
    # plt.figure(num=num)
    fig, ax = plt.subplots()
    ax.plot(ind_var, dep_var)

    ax.set(xlabel=x_label, ylabel=y_label)
    plt.title(title, fontproperties=font)
    ax.grid()

    # plt.rcParams['savefig.dpi'] = 300  # 图片像素
    # plt.rcParams['figure.dpi'] = 300  # 分辨率

    fig.savefig(path, dpi=600, format='svg')
    # fig.savefig("./images/primitiveFunction.png")
    if is_show:
        plt.show()


def num_plot(independent_variable, *dependent_variables, **kwargs):
    # path: str, x_label: str, y_label: str, title: str,is_show: bool = True):
    # 画图
    fig, ax = plt.subplots()
    i = 0
    for dependent_variable in dependent_variables:
        if kwargs.get('label') is not None:
            ax.plot(independent_variable, dependent_variable, label=kwargs['label'][i])
            i = i + 1
        else:
            ax.plot(independent_variable, dependent_variable)

    # 设置坐标轴和标题
    ax.set(xlabel=kwargs["x_label"], ylabel=kwargs["y_label"])
    if kwargs["title"] is not None:
        plt.title(kwargs["title"], fontproperties=font)
    # 网格显示
    ax.grid()
    # 标签 自动合适位置
    if kwargs.get('label') is not None:
        ax.legend(loc='best')
    # 设置像素和分辨率
    # plt.rcParams['savefig.dpi'] = 300  # 图片像素
    # plt.rcParams['figure.dpi'] = 300  # 分辨率
    # 保存图片
    plt.tight_layout()
    fig.savefig(kwargs["path"], dpi=600, format='svg')
    # 显示
    if kwargs["is_show"]:
        plt.show()
