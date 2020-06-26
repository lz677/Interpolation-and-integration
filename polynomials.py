#! C:\Users\93715\Anaconda3\python.exe
# *-* coding:utf8 *-*
"""
@author: LiuZhe
@license: (C) Copyright SJTU ME
@contact: LiuZhe_54677@sjtu.edu.cn
@file: polynomials.py
@time: 2020/6/23 9:12
@desc: 人生苦短，我用python
"""
import numpy as np
import sympy as sp
import solving_equation


class Polynomials(object):
    def __init__(self, n: int, x: np.ndarray, y: np.ndarray, y_diff: np.ndarray):
        # 初始化插值点
        self.n = n
        self.nodes_num = self.n + 1  # 插值节点数
        self.x = x  # 插值节点
        self.y = y  # 插值节点对应的y值
        self.y_diff = y_diff  # 插值节点的导数值

        # 拉格朗日多项式初始化
        # 插值基函数 初始化为全1列表
        self.l_k = [1 for i in range(self.nodes_num)]
        # 拉格朗日多项式 初始化为0
        self.L_n = 0
        # w_n+1 TODO:修改初始化值
        self.w = 1

        # 牛顿插值多项式初始化
        # 初始化多项式系数
        self.coefficients = np.zeros(self.nodes_num)
        # 多项式初始化
        self.P_n = 0
        # 插值表初始化
        self.v = np.zeros((self.nodes_num, self.nodes_num))
        self.p_np = 0

        # 多段线性初始化多项式
        self.i_h = [0] * self.n  # 每一段的多项式函数
        self.I_n = 0  # 整体多项式函数
        # self.I_n_num = 0  # 整体多项式函数

        # Hermite插值多项式初始化
        self.h_i_h = [0] * self.n  # 每一段的多项式函数
        self.H_I_n = 0  # 整体多项式函数
        # self.H_I_n_num = 0  # 整体多项式函数

    def lagrange_polynomials(self, x):
        """
        拉格朗日插值多项式
        :param x:   自变量 可以是离散值 也可以是解析解自变量符号
        """
        # 插值基函数
        for k in range(self.nodes_num):
            for i in range(self.nodes_num):
                if i != k:
                    self.l_k[k] = self.l_k[k] * (x - self.x[i]) / (self.x[k] - self.x[i])
        #     print(sp.latex(sp.N(sp.expand(self.l_k[k]),6)))
        # print('*'*100)
        # 拉格朗日多项式Ln(x) 以及 w_n+1
        for k in range(self.nodes_num):
            self.L_n += self.y[k] * self.l_k[k]
            self.w *= (x - self.x[k])

    def newton(self, x):
        """
        求解牛顿插值多项式的系数和插值多项式
        直接修改属性：参数coefficients和插值多项式P_n
        :param x: 自变量
        """
        # v = np.zeros((self.nodes_num, self.nodes_num))
        # 将y值赋值给v的第一列
        for j in range(self.nodes_num):
            self.v[j, 0] = self.y[j]
        # 求解均差表 保存到v中， 从上到下保存
        for i in range(1, self.nodes_num):  # 第i列
            for j in range(self.nodes_num - i):
                self.v[j, i] = (self.v[j + 1, i - 1] - self.v[j, i - 1]) / (self.x[j + i] - self.x[j])

        # 将系数赋值给系数数组
        for i in range(self.nodes_num):
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            # print(self.v[:, i])
            self.coefficients[i] = self.v[0, i]
        # print('*****插商表*****:')

        # np.set_printoptions(precision=6, suppress=True)

        # with open('./chashangbiao.txt', 'a+') as f:
        #     f.write(self.v)
        # print(self.v)
        # 定义自变量
        for i in range(self.nodes_num):
            p_i = 1
            for j in range(i):
                p_i *= x - self.x[j]

            self.P_n += self.coefficients[i] * p_i
        self.p_np = sp.lambdify(x, self.P_n, "numpy")

    def linear(self, x):
        """
        多段线性插值
        :param x: 自变量
        """
        # cond = []
        # 计算每一段的插值函数 共10段
        for k in range(self.n):
            # 第k段条件
            # cond.append((self.x_k[k] < x) & (self.x_k[k + 1] >= x))
            # 第k段表达式
            self.i_h[k] = ((x - self.x[k + 1]) / (self.x[k] - self.x[k + 1]) * self.y[k] +
                           (x - self.x[k]) / (self.x[k + 1] - self.x[k]) * self.y[k + 1])
            # print(self.i_h[k])
            # self.I_n_num += self.i_h[k] * ((self.x_k[k] < x) & (self.x_k[k + 1] >= x))

        self.I_n = sp.Piecewise((self.i_h[0], x < self.x[1]), (self.i_h[1], x < self.x[2]),
                                (self.i_h[2], x < self.x[3]), (self.i_h[3], x < self.x[4]),
                                (self.i_h[4], x < self.x[5]), (self.i_h[5], x < self.x[6]),
                                (self.i_h[6], x < self.x[7]), (self.i_h[7], x < self.x[8]),
                                (self.i_h[8], x < self.x[9]), (self.i_h[9], x < self.x[10]))

    def hermite(self, x):
        """
        多段Hermite插值 式(5.3)
        :param x: 自变量
        """
        # 计算每一段的两点三次插值函数共10段
        for k in range(self.n):
            # 第k段条件
            # cond.append((self.x_k[k] < x) & (self.x_k[k + 1] >= x))
            # 第k段表达式
            self.h_i_h[k] = (((x - self.x[k + 1]) / (self.x[k] - self.x[k + 1])) ** 2 *
                             (1 + 2 * (x - self.x[k]) / (self.x[k + 1] - self.x[k])) * self.y[k] +
                             ((x - self.x[k]) / (self.x[k + 1] - self.x[k])) ** 2 *
                             (1 + 2 * (x - self.x[k + 1]) / (self.x[k] - self.x[k + 1])) * self.y[k + 1] +
                             ((x - self.x[k + 1]) / (self.x[k] - self.x[k + 1])) ** 2 * (x - self.x[k]) * self.y_diff[
                                 k] +
                             ((x - self.x[k]) / (self.x[k + 1] - self.x[k])) ** 2 *
                             (x - self.x[k + 1]) * self.y_diff[k + 1])
            # 多项式
            self.H_I_n = sp.Piecewise((self.h_i_h[0], x < self.x[1]), (self.h_i_h[1], x < self.x[2]),
                                      (self.h_i_h[2], x < self.x[3]), (self.h_i_h[3], x < self.x[4]),
                                      (self.h_i_h[4], x < self.x[5]), (self.h_i_h[5], x < self.x[6]),
                                      (self.h_i_h[6], x < self.x[7]), (self.h_i_h[7], x < self.x[8]),
                                      (self.h_i_h[8], x < self.x[9]), (self.h_i_h[9], x < self.x[10]))
