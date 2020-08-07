#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt

# 多項式
def polynomial(w, x):
    d = len(w)
    return sum([w[k] * (x ** k) for k in range(d)])
    
# 最小二乗法による損失関数
def loss(w, x, y):
    n = len(x)
    return sum([(y[i] - polynomial(w, x[i])) ** 2 for i in range(n)]) / 2

# 目的関数の勾配
def grad(w, x, y):
    n = len(x)
    d = len(w)
    return np.array([-sum([(y[i] - polynomial(w, x[i])) * (x[i] ** k) for i in range(n)]) for k in range(d)])

if __name__ == "__main__":
    # y = sin(2pix) + (誤差項)
    # 誤差項は 平均0 分散1 の正規分布によって生成
    n = 20
    x = np.array([i / n for i in range(n+1)])
    y = np.array([math.sin(2 * math.pi * i) + np.random.randn() for i in x])

    # ハイパーパラメータ
    d = 10
    alpha = 0.01
    n_epoch = 10000

    # 多項式係数の初期条件
    w = np.array([np.random.randn() for i in range(d+1)])

    # 勾配法による更新
    for i in range(n_epoch):
        w -= alpha * grad(w, x, y) 

        # 損失関数の推移を標準出力によって表示
        if (i % 100 == 0):
            l = loss(w, x, y)
            print("step: {step}, loss: {loss}".format(step=i, loss=l))


    # 散布図
    plt.figure(figsize=(15, 15))
    plt.scatter(x, y)

    # 多項式曲線
    _y = [polynomial(w, i) for i in x]

    # 多項式回帰の結果を表示
    plt.plot(x, _y, color="red")
    plt.show()
