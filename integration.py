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
    def __init__(self, n: int, x: np.ndarray, y: np.ndarray, y_diff: np.ndarray):
        # 初始化插值点
        self.n = n
        self.nodes_num = self.n + 1  # 插值节点数
        self.x = x  # 插值节点
        self.y = y  # 插值节点对应的y值
        self.y_diff = y_diff  # 插值节点的导数值

        # # 拉格朗日
        # 拉格朗日等节点插值积分项
        self.L_A_k = np.zeros(self.nodes_num)
        self.L_A_k_poly = [1 for i in range(self.nodes_num)]
        self.L_A_k_a = [1 for i in range(self.nodes_num)]
        self.L_A_k_b = [1 for i in range(self.nodes_num)]
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
        self.nc_I_n_poly = 0

        self.a = sp.Symbol('a')
        self.b = sp.Symbol('b')

        # 分段线性插值
        self.h = 2 / self.n
        self.T_n = 0

        # hermite
        self.H_T_n = 0

        # Gauss
        self.xk1 = np.zeros(self.nodes_num)
        self.xk2 = np.zeros(self.nodes_num)
        self.yk1 = np.zeros(self.nodes_num)
        self.yk2 = np.zeros(self.nodes_num)
        self.G_T_n = 0

    def lagrange(self, l_k, x, L_n, f):
        # 求解 A_k I_n 属于理论推导部分
        for i in range(self.nodes_num):
            self.L_A_k_a[i] = sp.integrate(l_k[i]).subs(x, self.a)
            self.L_A_k_b[i] = sp.integrate(l_k[i]).subs(x, self.b)
            self.L_A_k_poly[i] = self.L_A_k_b[i] - self.L_A_k_a[i]
            # print(self.L_A_k_poly[i])
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
            self.C_k[k] = (-1) ** (sp.Integer(self.n - k)) / (
                    self.n * math.factorial(sp.Integer(k)) * math.factorial(sp.Integer(self.n - k))) * sp.integrate(nc[k], (t, sp.Integer(0), sp.Integer(10)))
            print("k:", self.C_k[k])
            self.nc_I_n += 2 * self.C_k[k] * self.y[k]

        #     print(nc[k])
        # print('*' * 100)

    def linear(self):
        # 梯形积分 T_n = h/2*[f(a)+2*sum f(x_k)+f(b)] k = 1,2,...,n-1
        for k in range(1, self.n):
            self.T_n += 2 * self.y[k]
        self.T_n = self.h / 2 * (self.y[0] + self.T_n + self.y[self.n])

    def hermite(self):
        for k in range(1, self.n):
            self.H_T_n += 2 * self.y[k]
        self.H_T_n = self.h / 2 * (self.y[0] + self.H_T_n + self.y[self.n]) + self.h ** 2 / 12 * (
                self.y_diff[0] - self.y_diff[self.n])

    def gauss(self, x):
        for i in range(self.n):
            self.xk1[i] = -sp.sqrt(3) / 30 + (self.x[i] + self.x[i + 1]) / 2
            self.xk2[i] = sp.sqrt(3) / 30 + (self.x[i] + self.x[i + 1]) / 2
            self.yk1[i] = 1 / (1 + 25 * x ** 2).subs(x, self.xk1[i])
            self.yk2[i] = 1 / (1 + 25 * x ** 2).subs(x, self.xk2[i])
            self.G_T_n += 1 / 10 * (self.yk1[i] + self.yk2[i])
            # print('$%d$ & $%.1f$ & $%.1f$ & $%.6f$ & $%.6f$ & $%.6f$ & $%.6f$ \\\\' % (
            # i, self.x[i], self.x[i + 1], self.xk1[i], self.xk2[i], self.yk1[i], self.yk2[i]))

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
