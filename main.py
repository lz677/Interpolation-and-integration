#! C:\Users\93715\Anaconda3\python.exe
# *-* coding:utf8 *-*
"""
@author: LiuZhe
@license: (C) Copyright SJTU ME
@contact: LiuZhe_54677@sjtu.edu.cn
@file: main.py
@time: 2020/6/23 9:27
@desc: 人生苦短，我用python
"""

import numpy as np
import sympy as sp
import math
import polynomials
import solving_equation
import integration
import error

n = 10  # 插值最大节点下标
nodes = n + 1  # 插值点数
x_k = np.zeros(nodes)  # 等步长插值节点
y_k = np.zeros(nodes)  # 插值节点对应的y值
y_k_diff = np.zeros(nodes)  # 插值节点对应的y'值
x_k_c = np.zeros(nodes)  # 切比雪夫插值零点
y_k_c = np.zeros(nodes)  # 切比雪夫插值零点对应y值
x = sp.Symbol('x')  # x为解析符号
f = 1 / (1 + 25 * x ** 2)  # 原函数表达式
f_diff = f.diff(x)
h = 0.2

for k in range(nodes):
    # 等值节点 x0,x1 ...... x9,x10
    x_k[k] = -1 + 2 / 10 * k
    # 等值节点的函数值
    # y_k[k] = 1 / (1 + 25 * x_k[k] ** 2)
    y_k[k] = f.subs(x, x_k[k])
    # 等值节点的导数值
    y_k_diff[k] = f_diff.subs(x, x_k[k])
    # 切比雪夫插值节点 x0,x1 ...... x9,x10
    x_k_c[k] = np.cos((2 * k + 1) * np.pi / (2 * (n + 1)))
    # 切比雪夫插值节点对应的函数值
    y_k_c[k] = 1 / (1 + 25 * x_k_c[k] ** 2)

la_debug = 0
new_debug = 1
linear_print_plot = 0
hermitr_print_plot = 0


def main(is_plot=False):
    # TODO: 求插值多项式
    # 等节点
    poly = polynomials.Polynomials(n, x_k, y_k, y_k_diff)
    er = error.InterpolationError(n, x_k, y_k, y_k_diff)
    ing = integration.Integration(n, x_k, y_k)
    # 切比雪夫零点
    poly_c = polynomials.Polynomials(n, x_k_c, y_k_c, y_k_diff)
    er_c = error.InterpolationError(n, x_k_c, y_k_c, y_k_diff)
    ing_c = integration.Integration(n, x_k_c, y_k_c)
    if la_debug:
        # # 拉格朗日等节点插值多项式
        poly.lagrange_polynomials(x)
        # 插值余项
        lagrange_e_r_n = f - poly.L_n
        # 估计误差 无穷范数和最大值范数
        er.error_lagrange(0.95, x, lagrange_e_r_n, w=poly.w)
        if is_plot:
            # sp.plot(lagrange_e_r_n, (x, -1, 1), title=r'$Lagrange polynomials\' error$')
            # sp.plot(sp.diff(lagrange_e_r_n), (x, -1, 1), title=r"$R_n\'$")
            print('拉格朗日等节点插值余项：', )
            print(er.L_R_n)
            # print(sp.latex(sp.N(sp.expand(lagrange_e_r_n), 5)))
            # print(sp.latex(sp.N(sp.diff(sp.expand(lagrange_e_r_n)), 5)))
            print('拉格朗日等节点插值余项最大范数:', er.L_max_nor_acc)
            print('拉格朗日等节点插值余项2范数:', er.L_2_nor)
            print('*' * 100)
        # 积分
        ing.lagrange(poly.l_k, x, poly.L_n, f)
        if is_plot:
            # for i in range(n + 1):
            # print(sp.latex(sp.N(sp.expand(poly.l_k[i]), 5)))
            # print(sp.latex(sp.N(sp.expand(sp.integrate(poly.l_k[i])), 5)))
            print("实际积分为：", sp.integrate(poly.L_n, (x, -1, 1)))
            print("积分为：", ing.L_I_n)
            print("代数精度:", ing.L_m)
            print("积分余项:", ing.R_f_ac)
            print("手推插值余项的积分：")
            print(sp.latex(sp.N(ing.R_f_ac_poly, 6)))
            print("电脑推插值余项的积分：")
            print(sp.latex(sp.expand(sp.N(sp.integrate(lagrange_e_r_n), 6))))
            print('*' * 100)

        # # 拉格朗日切比雪夫零点
        poly_c.lagrange_polynomials(x)
        # 插值余项
        la_c_r_n = f - poly_c.L_n
        # 估计误差 无穷范数和最大值范数
        er_c.error_lagrange(0.3, x, la_c_r_n, w=poly.w, is_print=False)
        if is_plot:
            # sp.plot(r_n**2, (x, -1, 1), title=r'$Lagrange polynomials\' error$')
            print('拉格朗日切比雪夫插值余项：最大范数', er_c.L_max_nor_acc)
            print('拉格朗日切比雪夫插值余项：2范数', er_c.L_2_nor)
            print('*' * 100)
        # 积分
        ing_c.lagrange(poly_c.l_k, x, poly_c.L_n, f)
        if is_plot:
            # for i in range(n + 1):
            # print(sp.latex(sp.N(sp.expand(poly.l_k[i]), 5)))
            # print(sp.latex(sp.N(sp.expand(sp.integrate(poly.l_k[i])), 5)))
            print("实际积分为：", sp.integrate(poly_c.L_n, (x, -1, 1)))
            print("积分为：", ing_c.L_I_n)
            print("代数精度:", ing_c.L_m)
            print("积分余项:", ing_c.R_f_ac)
            print("手推插值余项的积分：")
            print(sp.latex(sp.N(ing_c.R_f_ac_poly, 6)))
            print("电脑推插值余项的积分：")
            print(sp.latex(sp.expand(sp.N(sp.integrate(la_c_r_n), 6))))
            print('*' * 100)

    if new_debug:
        # 牛顿等节点插值
        poly.newton(x)
        newton_r_n = f - poly.P_n
        er.error_lagrange(0.95, x, newton_r_n, w=poly.w)
        if is_plot:
            # sp.plot(newton_r_n ** 2, (x, -1, 1), title=r'$Lagrange polynomials\' error$')
            print('牛顿等节点插值余项：最大范数', er.L_max_nor_acc)
            print('牛顿等节点插值余项：2范数', er.L_2_nor)
            print("*" * 100)
        # 积分 Newton-Cotes
        ing.newton_cotes()
        if is_plot:
            print("牛顿科特斯积分：", ing.nc_I_n)
            print("科特斯系数")
            print(ing.C_k)
            print(ing.C_k.sum())
            print("*" * 100)

        # 牛顿切比雪夫节点插值
        poly_c.newton(x)
        newton_c_r_n = f - poly_c.P_n
        er_c.error_lagrange(0.3, x, newton_c_r_n, w=poly.w)
        if is_plot:
            # sp.plot(newton_c_r_n**2, (x, -1, 1), title=r'$Lagrange polynomials\' error$')
            print('牛顿切比雪夫插值余项：最大范数', er_c.L_max_nor_acc)
            print('牛顿切比雪夫插值余项插值余项：2范数', er_c.L_2_nor)
            print("*" * 100)

    # 分段线性插值
    # poly.linear(x)

    # # Herimite插值
    # poly.hermite(x)

    # if is_plot:
    #     if la_debug:
    #         # 拉格朗日等节点插值多项式
    #         print(sp.expand(poly.L_n))
    #         sp.plot(f, poly.L_n, (x, -1, 1), title="L_E")
    #
    #         # 拉格朗日切比雪夫零点
    #         print(sp.expand(poly_c.L_n))
    #         sp.plot(f, poly_c.L_n, (x, -1, 1), title="L_C")
    #     if new_debug:
    #         # 牛顿等节点插值
    #         print(sp.expand(poly.P_n))
    #         sp.plot(f, poly.P_n, (x, -1, 1), title="N_E")
    #
    #         # 牛顿切比雪夫节点插值
    #         print(sp.expand(poly_c.P_n))
    #         sp.plot(f, poly_c.P_n, (x, -1, 1), title="N_C")
    #     if linear_print_plot:
    #         # 分段线性插值
    #         print(sp.expand(poly.I_n))
    #         sp.plot(f, poly.I_n, (x, -1, 1), title="Linear")
    #     if hermitr_print_plot:
    #         # Herimite插值
    #         print(sp.expand(poly.H_I_n))
    #         sp.plot(f, poly.H_I_n, (x, -1, 1), title="Hermite")


if __name__ == '__main__':
    main(True)
