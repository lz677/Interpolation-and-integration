#! C:\Users\93715\Anaconda3\python.exe
# *-* coding:utf8 *-*
"""
@author: LiuZhe
@license: (C) Copyright SJTU ME
@contact: LiuZhe_54677@sjtu.edu.cn
@file: solving_equation.py
@time: 2020/6/23 14:01
@desc: 人生苦短，我用python
"""
import numpy as np
import sympy as sp


class SolvingEquation(object):
    def __init__(self, x_0, x: sp.Symbol, f: sp.frac, n: int = 100, delta: float = 10 ** (-6)):
        # 初始化
        self.x = x  # 函数符号'x'
        self.f = f  # 函数表达式
        self.f_diff = sp.diff(f)  # 函数表达式的导数
        self.x_0 = x_0  # 初试近似值 x0
        self.f_0 = self.f.subs(self.x, self.x_0)  # f0
        self.f_0_diff = self.f_diff.subs(self.x, self.x_0)  # f'0
        self.N = n  # 最大迭代次数100
        self.delta = delta  # 迭代精度

    def newton(self, is_print: bool = False):
        n = 0
        x_1 = self.x_0
        while n < self.N and self.f_diff.subs(self.x, x_1) != 0:
            x_1 = self.x_0 - self.f_0 / self.f_0_diff
            f_1 = self.f.subs(self.x, x_1)
            f_1_diff = self.f_diff.subs(self.x, x_1)
            if self.__delta(self.x_0, x_1) < self.delta:
                self.x_0 = x_1
                # TODO: 记录值
                if is_print:
                    print(f"此满足精度，x0={self.x_0}")
                break
            else:
                self.x_0 = x_1
                self.f_0 = f_1
                self.f_0_diff = f_1_diff
                if is_print:
                    print(f"不满足精度，x0={self.x_0}")

    def __delta(self, x_0: float, x_1: float) -> float:
        return sp.Abs(x_1 - x_0) if sp.Abs(x_0) < 1 else sp.Abs(x_1 - x_0) / sp.Abs(x_1)


if __name__ == '__main__':
    x = sp.Symbol('x')
    # print(np.fabs(0.0))
    se = SolvingEquation(0.0, x, (x - 1) * (x - 2), 100)
    se.newton()
