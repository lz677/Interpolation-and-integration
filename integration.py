#! C:\Users\93715\Anaconda3\python.exe
# *-* coding:utf8 *-*
"""
@author: LiuZhe
@license: (C) Copyright SJTU ME
@contact: LiuZhe_54677@sjtu.edu.cn
@file: integration.py
@time: 2020/6/24 9:20
@desc: 人生苦短，我用python
"""

import numpy as np
import sympy as sp
import math


class Integration(object):
    def __init__(self, n: int, x: np.ndarray, y: np.ndarray):
        # 初始化插值点
        self.n = n
        self.nodes_num = self.n + 1  # 插值节点数
        self.x = x  # 插值节点
        self.y = y  # 插值节点对应的y值
        # self.y_diff = y_diff  # 插值节点的导数值

        # # 拉格朗日
        # 拉格朗日等节点插值积分项
        self.L_A_k = np.zeros(self.nodes_num)
        self.L_A_k_sum = 0
        self.L_I_n = 0  # 积分

        # 代数精度
        self.L_m = 0
        # 积分余项
        self.R_f_ac_poly = 0  # 插值余项的积分
        self.R_f_ac = 0  # 积分余项
        self.R_f_et_poly = 0  # 求积余项估计

        # # newton-cotes
        self.C_k = np.zeros(self.nodes_num)
        self.nc_I_n = 0

    def lagrange(self, l_k, x, L_n, f):
        # 求解 A_k I_n 属于理论推导部分
        for i in range(self.nodes_num):
            self.L_A_k[i] = sp.integrate(l_k[i], (x, -1, 1))
            self.L_A_k_sum += self.L_A_k[i]
            self.L_I_n += self.L_A_k[i] * self.y[i]

        # 代数精度
        self.L_m = self.__degree_of_precision(self.L_A_k)

        # 积分余项
        # 准确的方式
        self.R_f_ac_poly = 1 / 5 * sp.atan(5 * x) - sp.integrate(L_n)
        self.R_f_ac = self.R_f_ac_poly.subs(x, 1) - self.R_f_ac_poly.subs(x, -1)
        # 估计的方式 L_A_k_sum在__degree_of_precision修改
        K = (1 / math.factorial(self.L_m + 1) * (
                1 / (self.L_m + 2) * (1 ** (self.L_m + 2) - (-1) ** (self.L_m + 2)) - self.L_A_k_sum))
        # 求解对应的m+1阶导数
        f_d = f
        for i in range(self.L_m + 1):
            f_d = sp.diff(f_d)
        self.R_f_et_poly = K * f_d

    def newton_cotes(self):
        t = sp.Symbol('t')
        nc = [1 for i in range(self.nodes_num)]
        for k in range(self.nodes_num):
            for j in range(self.nodes_num):
                if j != k:
                    nc[k] *= t - j
            self.C_k[k] = (-1) ** (self.n - k) / (
                    self.n * math.factorial(k) * math.factorial(self.n - k)) * sp.integrate(nc[k], (t, 0, 10))
            self.nc_I_n += 2 * self.C_k[k] * self.y[k]
        #     print(nc[k])
        # print('*' * 100)
        pass

    def __degree_of_precision(self, A_k, m=0) -> int:
        """
        利用定义求解代数精度
        :param m: 初试满足精度
        :param A_k: 求积公式求得的A_K系数
        :return: m 代数精度
        """
        while True:
            A_k_sum = 0
            for i in range(self.nodes_num):
                A_k_sum += A_k[i] * self.x[i] ** m
            right_value = 1 / (m + 1) * (1 ** (m + 1) - (-1) ** (m + 1))
            if sp.Abs(A_k_sum - right_value) < 0.00001:
                m += 1
            else:
                m += -1
                break
        # 修改self.L_A_k_sum 用于后续K计算
        self.L_A_k_sum = A_k_sum
        return m
