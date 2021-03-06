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
import plot_figures

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
x_np = np.linspace(-1, 1, 20001)
f_np = sp.lambdify(x, f, "numpy")(x_np)
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
# print(y_k)
kexi = sp.Symbol(r'\eta')
# print(sp.latex(sp.diff(f).subs(x, kexi)))
# print(sp.latex(sp.diff(sp.diff(f)).subs(x, kexi)))
# print(sp.latex(sp.diff(sp.diff(sp.diff(f))).subs(x, kexi)))
# print(sp.latex(sp.diff(sp.diff(sp.diff(sp.diff(sp.diff(sp.diff(f)))))).subs(x, kexi)))
# print(sp.latex(sp.diff(sp.diff(sp.diff(sp.diff(f)))).subs(x, kexi)))
# print('x_k_C')
# print(x_k_c)
# print('y_k_C')
# print(y_k_c)
# for i in range(5):
#     print('$%d$ & $%.1f$ & $%.6f$ & $%.6f$ & $%d$ & $%.1f$ & $%.6f$ & $%.6f$ \\\\' % (
#     i, x_k[i], y_k[i], y_k_diff[i], i + 6, x_k[i + 6], y_k[i + 6], y_k_diff[i + 6]))
# print('$%d$ & $%.1f$ & $%.6f$ & $%.6f$ & & &  & \\\\' % (
#     5, x_k[5], y_k[5], y_k_diff[5]))
# f_d2 = sp.diff(sp.diff(f))
# f_d3 = sp.diff(f_d2)
# f_d4 = sp.diff(f_d3)
# f_d5 = sp.diff(f_d4)
# print(sp.latex(f_d4))
# print(sp.latex(f_d5))
# print(sp.latex(sp.solve(f_d5,x)))
# sp.plot(f_d2, (x, -1, 1))
# sp.plot(f_d3, (x, -1, 1))
la_debug = 0
new_debug = 0
linear_debug = 0
hermite_debug = 0
Gauss_debug = 0


def calc_cotes(n_=10, is_print=False):
    """
    打印科特斯系数
    :param is_print:
    :param n_:
    :return:
    """
    t = sp.Symbol('t')
    omega = 1
    n_ = sp.Integer(n_)
    for j in range(n_ + 1):
        omega *= (t - j)
    cotes = []
    for k in range(n_ + 1):
        k = sp.Integer(k)
        _ = (-1) ** (n_ - k) / (n_ * math.factorial(k)) / math.factorial(n_ - k) * sp.integrate(omega / (t - k),
                                                                                                (t, 0, n_))
        cotes.append(_)
        if is_print:
            print('$ ' + sp.latex(_) + ' $', end=' & ')
    return cotes


def main(is_plot=False):
    # TODO: 求插值多项式
    # 等节点
    poly = polynomials.Polynomials(n, x_k, y_k, y_k_diff)
    er = error.InterpolationError(n, x_k, y_k, y_k_diff)
    ing = integration.Integration(n, x_k, y_k, y_k_diff)
    # # 切比雪夫零点
    poly_c = polynomials.Polynomials(n, x_k_c, y_k_c, y_k_diff)
    er_c = error.InterpolationError(n, x_k_c, y_k_c, y_k_diff)
    ing_c = integration.Integration(n, x_k_c, y_k_c, y_k_diff)

    if la_debug:
        # # 拉格朗日等节点插值多项式
        poly.lagrange_polynomials(x)

        # 插值余项
        lagrange_e_r_n = f - poly.L_n
        # 估计误差 无穷范数和最大值范数
        er.error_lagrange(0.95, x, lagrange_e_r_n, w=poly.w)
        if is_plot:
            # sp.plot(lagrange_e_r_n, (x, -1, 1))
            # sp.plot(sp.diff(lagrange_e_r_n), (x, -1, 1), title=r"$R_n\'$")
            # for i in range(nodes):
            #     print("l%d" % i)
            #     print(sp.latex(sp.N(sp.expand(poly.l_k[i]), 6)))
            kexi = sp.Symbol(r'\xi')
            print('w:', sp.latex(sp.N(sp.expand(poly.w), 6)))
            f_d = f
            for i in range(nodes):
                f_d = sp.diff(f_d)
            print('插值余项')
            # lagrange_e_r_n_np_d = sp.lambdify(x, sp.diff(lagrange_e_r_n), "numpy")(x_np)
            # lagrange_e_r_n_np = sp.lambdify(x, lagrange_e_r_n, "numpy")(x_np)
            # plot_figures.num_plot(x_np, lagrange_e_r_n_np, path="C:\\Users\\93715\\Desktop\\数值分析期末大作业\\figure\\larn.svg",
            #                       x_label=r'$x$',
            #                       y_label=r'$R_n(x)$', title='拉格朗日插值余项', is_show=True,)
            # plot_figures.num_plot(x_np, lagrange_e_r_n_np_d, path="C:\\Users\\93715\\Desktop\\数值分析期末大作业\\figure\\larnd.svg",
            #                       x_label=r'$x$',
            #                       y_label=r'$R’_n(x)$', title='拉格朗日插值余项导数', is_show=True, )
            # print(sp.latex(sp.N(f_d, 6)))
            # print(sp.latex(sp.N(f_d.subs(x, kexi), 6)))
            # print(sp.latex(sp.N(sp.diff(f_d.subs(x, sp.Symbol('\eta'))), 6)))
            # print(sp.latex(sp.N(sp.expand(poly.L_n), 6)))
            print('拉格朗日等节点插值余项：', )
            print(er.L_R_n)
            # print(sp.latex(sp.N(sp.expand(lagrange_e_r_n), 5)))
            print('R\'n')
            # print(sp.latex(sp.N(sp.diff(sp.expand(lagrange_e_r_n)), 6)))
            # sp.plot(sp.diff(lagrange_e_r_n), (x, -1, 1))
            print('拉格朗日等节点插值余项最大范数:', er.L_max_nor_acc)
            print('拉格朗日等节点插值余项2范数:', er.L_2_nor)

            print('*' * 100)
        # 积分
        ing.lagrange(poly.l_k, x, poly.L_n, f)
        if is_plot:
            ing.newton_cotes()
            # for i in range(n + 1):
            # print(sp.latex(sp.N(sp.expand(poly.l_k[i]), 5)))
            # print(sp.latex(sp.N(sp.expand(sp.integrate(poly.l_k[i])), 5)))
            print("A_k")
            print(ing.L_A_k)
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
        if 0:
            # sp.plot(r_n**2, (x, -1, 1), title=r'$Lagrange polynomials\' error$')

            np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

            print('w')
            print(sp.latex(sp.N(sp.expand(poly_c.w), 6)))
            print('L_n_c')
            print(sp.latex(sp.N(sp.expand(poly_c.L_n), 6)))

            # print(sp.latex(sp.N(sp.expand(poly.l_k[i]), 6)))
            f_d = f
            for i in range(nodes):
                f_d = sp.diff(f_d)
            print('插值余项')
            kexi = sp.Symbol(r'\xi')
            print(sp.latex(sp.N(f_d, 6)))
            print(sp.latex(sp.N(f_d.subs(x, kexi), 6)))
            print('R\'n')
            print(sp.latex(sp.N(sp.diff(sp.expand(la_c_r_n)), 6)))

            # la_c_r_n_d = sp.lambdify(x, sp.diff(la_c_r_n), "numpy")(x_np)
            # la_c_r_n = sp.lambdify(x, la_c_r_n, "numpy")(x_np)
            # plot_figures.num_plot(x_np, la_c_r_n, path="C:\\Users\\93715\\Desktop\\数值分析期末大作业\\figure\\lacrn.svg",
            #                       x_label=r'$x$',
            #                       y_label=r'$R_n(x)$', title='拉格朗日插值余项', is_show=True,)
            # plot_figures.num_plot(x_np, la_c_r_n_d, path="C:\\Users\\93715\\Desktop\\数值分析期末大作业\\figure\\lacrnd.svg",
            #                       x_label=r'$x$',
            #                       y_label=r'$R’_n(x)$', title='拉格朗日插值余项导数', is_show=True, )
            print('拉格朗日切比雪夫插值余项：最大范数', er_c.L_max_nor_acc)
            print('拉格朗日切比雪夫插值余项：2范数', er_c.L_2_nor)
            print('*' * 100)
        # 积分
        ing_c.lagrange(poly_c.l_k, x, poly_c.L_n, f)
        if 0:
            L_n_np = sp.lambdify(x, poly_c.L_n, "numpy")(x_np)
            # print(p_n_np)
            # print(f_np)
            # plot_figures.num_plot(x_np, f_np, L_n_np,
            #                       path="C:\\Users\\93715\\Desktop\\数值分析期末大作业\\figure\\LandFNumc.svg",
            #                       x_label=r'$x$',
            #                       y_label=r'$f(x),L_n(x)$', title='原函数与拉格朗日插值多项式', is_show=True,
            #                       label=[r'$Primitive\ Function$', r'$Lagrange\ Polynomials$'])
            # for i in range(n + 1):
            # print(sp.latex(sp.N(sp.expand(poly.l_k[i]), 5)))
            # print(sp.latex(sp.N(sp.expand(sp.integrate(poly.l_k[i])), 5)))
            # for i in range(nodes):
            #     print("l%d" % i)
            #     print(sp.latex(ing_c.L_A_k_poly[i]))
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

        # p_n_np = sp.lambdify(x, poly.P_n, "numpy")(x_np)
        # # print(p_n_np)
        # # print(f_np)
        # plot_figures.num_plot(x_np, f_np, p_n_np, path="C:\\Users\\93715\\Desktop\\数值分析期末大作业\\figure\\NandFNumc.svg",
        #                       x_label=r'$x$',
        #                       y_label=r'$f(x),P_n(x)$', title='原函数与牛顿插值多项式', is_show=True,
        #                       label=[r'$Primitive\ Function$', r'$Newton\ Polynomials$'])

        if is_plot:
            print('插值多项式')
            print(sp.latex(sp.N(sp.expand(poly.P_n), 6)))
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
            # p_n_np = sp.lambdify(x, poly_c.P_n, "numpy")(x_np)
            # # print(p_n_np)
            # # print(f_np)
            # plot_figures.num_plot(x_np, f_np, p_n_np,
            #                       path="C:\\Users\\93715\\Desktop\\数值分析期末大作业\\figure\\NandFNumc.svg",
            #                       x_label=r'$x$',
            #                       y_label=r'$f(x),P_n(x)$', title='原函数与牛顿插值多项式', is_show=True,
            #                       label=[r'$Primitive\ Function$', r'$Newton\ Polynomials$'])
            # sp.plot(newton_c_r_n**2, (x, -1, 1), title=r'$Lagrange polynomials\' error$')
            print('插值多项式')
            print(sp.latex(sp.N(sp.expand(poly.P_n), 6)))
            # print(sp.latex(sp.N(sp.expand(poly.P_n), 6)))
            print('牛顿切比雪夫插值余项：最大范数', er_c.L_max_nor_acc)
            print('牛顿切比雪夫插值余项插值余项：2范数', er_c.L_2_nor)
            print("*" * 100)

    if linear_debug:
        # 分段线性插值
        poly.linear(x)
        print(sp.latex(sp.N(poly.I_n, 6)))
        if is_plot:
            ing.linear()
            print("线性插值函数积分：", ing.T_n)
        #     I_n_np = sp.lambdify(x, poly.I_n, "numpy")(x_np)
        #     plot_figures.num_plot(x_np, f_np, I_n_np,
        #                           path="C:\\Users\\93715\\Desktop\\数值分析期末大作业\\figure\\LinearandF.svg",
        #                           x_label=r'$x$',
        #                           y_label=r'$f(x),I_n(x)$', title='原函数与分段线性插值函数', is_show=True,
        #                           label=[r'$Primitive\ Function$', r'$Linear\ Splines$'])
    if hermite_debug:
        # Hermite插值
        poly.hermite(x)
        if is_plot:
            ing.hermite()
            print("线性插值函数积分：", ing.H_T_n)
            I_n_np = sp.lambdify(x, poly.H_I_n, "numpy")(x_np)
            plot_figures.num_plot(x_np, f_np, I_n_np,
                                  path="C:\\Users\\93715\\Desktop\\数值分析期末大作业\\figure\\HandF2.svg",
                                  x_label=r'$x$',
                                  y_label=r'$f(x),I_n(x)$', title='原函数与分段Hermite插值函数', is_show=True,
                                  label=[r'$Primitive\ Function$', r'$Hermite\ Splines$'])

    if Gauss_debug:
        ing.gauss(x)
        print("高斯积分：", ing.G_T_n)

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
    cotes = calc_cotes(is_print=False)
    a = sp.symbols("a")
    b = sp.symbols("b")
    K = 0

    # for i in range(nodes):
    #     K += cotes[i] * (a + (b - a) / n * i) ** 12
    # f = (b ** 13 - a ** 13) / sp.Integer(13) - (b - a) * K
    # print(136500000000 * sp.apart(f, a) / 26927 )
    # print('ss')
    # print(sp.expand(f))
    # print(f.subs({a: -1, b: 1})/math.factorial(12))
    # print(-26927/7981410937500000)
    # # print(164924164/533203125)
    # # print(sp.simplify(sp.expand(f)))
    # # a = sp.Integer(0)
    # # x_k_f = [1/26,1,1,1,1,1,1,1,1,1,1]
    # sum_k = 0
    # for i in range(nodes):
    #     # sum_k += sp.Integer(2) * cotes[i] * sp.Integer(x_k[i]) ** 12
    #     sum_k += 2 * cotes[i] * x_k[i] ** 12
    # print("a", (sp.Integer(2)/sp.Integer(13)-sum_k)/sp.Integer(math.factorial(12)))
    # print(2**13*26927/136500000000/math.factorial(12))
    # print(26927/136500000000*2/math.factorial(12))