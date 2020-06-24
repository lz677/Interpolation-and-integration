#! C:\Users\93715\Anaconda3\python.exe
# *-* coding:utf8 *-*
"""
@author: LiuZhe
@license: (C) Copyright SJTU ME
@contact: LiuZhe_54677@sjtu.edu.cn
@file: error.py
@time: 2020/6/23 15:43
@desc: 人生苦短，我用python
"""
import numpy as np
import sympy as sp
import math

import polynomials
import solving_equation


class InterpolationError(polynomials.Polynomials):
    def __init__(self, n: int, x: np.ndarray, y: np.ndarray, y_diff: np.ndarray):
        super().__init__(n, x, y, y_diff)

        self.f = 1 / (1 + 25 * x ** 2)
        # 拉个朗日
        # 插值余项
        self.L_R_n = sp.Symbol('x')
        # f^{(11)}
        self.f_diff_11 = sp.Symbol('x')
        # 精确的无穷范数
        self.L_max_nor_acc = 0
        # 估计的无穷范数
        self.L_max_nor_est = 0
        # 2范数
        self.L_2_nor = 0
        # 步长
        self.h = 0.2

    def error_lagrange(self, x_0, x, interpolation, w, n=100, delta=10 ** (-6), is_print=False):
        """
        :param is_print: true 显示记录迭代过程的每个值
        :param x_0: 初始值
        :param x: 自变量符号
        :param interpolation: 插值余项
        :param n: 求零点迭代次数
        :param delta: 求零点迭代精度
        """
        self.L_R_n = interpolation  # 插值余项
        # 求解精确解
        # 求插值余项导数的零点 从而求插值余项最大值
        se = solving_equation.SolvingEquation(x_0, x, sp.diff(self.L_R_n), n, delta)
        # 采用牛顿法 true 显示记录迭代过程的每个值
        se.newton(is_print)
        # 最大范数赋值
        self.L_max_nor_acc = sp.Abs(self.L_R_n.subs(x, se.x_0))

        # 2范数方式估计误差
        f_nor_2 = self.L_R_n ** 2
        x_k = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
        x_k_half = np.zeros(n)
        y_k_half = np.zeros(n)
        y_nor = np.zeros(n + 1)
        # 辛普森积分
        # 求中间节点及值
        for k in range(self.n):
            # 求x_{k+1/2}
            x_k_half[k] = x_k[k] + 1 / 2 * self.h
            # 求对应的 f(x_{k+1/2})
            y_k_half[k] = f_nor_2.subs(x, x_k_half[k])
            y_nor[k] = f_nor_2.subs(x, x_k[k])
        y_nor[self.n] = f_nor_2.subs(x, x_k[self.n])
        # 求积分
        self.L_2_nor = 0
        for k in range(1, self.n):
            self.L_2_nor += 2 * y_nor[k] + 4 * y_k_half[k]
        self.L_2_nor = sp.sqrt(self.h / 6 * (y_nor[0] + 4 * y_k_half[0] + self.L_2_nor + y_nor[n]))

        # 估算范围用公式
        # TODO: 估计最大值
        # f_d = self.f
        # for i in range(self.n + 1):
        #     f_d = sp.diff(f_d)
        # self.f_diff_11 = f_d
        # sol_equ = solving_equation.SolvingEquation(0.02, x, sp.diff(self.f_diff_11))
        # sol_equ.newton(False)
        # self.L_max_nor_est = f_d.subs(x, sol_equ.x_0) / math.factorial(n + 1) * w
        # # # 误差估计求导方式
        # # print(f)
        # f_d = f
        # for i in range(n + 1):
        #     f_d = sp.diff(f_d)
        # # print(sp.latex(sp.N(f_d, 5)))
        # # print(f_d)
        # # print(sp.diff(f_d))
        # # sp.plot(f_d, (x, -0.05, 0.05), title=r'$f^{(n+1)}$')
        # # sp.plot(sp.diff(f_d), (x, -1, 1), title=r'$f^{(n+2)}$')
        # # 0.0242843966444520
        # sol_equ = solving_equation.SolvingEquation(0.02, x, sp.diff(f_d), 100)
        # sol_equ.newton(False)
        # print(pl.w)
        # print(math.factorial(n + 1))
        # max_f = f_d.subs(x, sol_equ.x_0) / math.factorial(n + 1) * pl.w
        # sp.plot(max_f, (x, -1, 1))
