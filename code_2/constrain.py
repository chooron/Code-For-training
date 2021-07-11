# 问题约束
import numpy as np

supply = [250, 200, 180]
need = [80, 70, 90, 80, 120, 70, 100, 90]
loss = [10, 8, 5, 10, 10, 8, 5, 8]


def constrain_1(x):
    x = x.reshape(3, 8)
    return -np.sum(x, axis=0)[0] + need[0]


def constrain_2(x):
    x = x.reshape(3, 8)
    return -np.sum(x, axis=0)[1] + need[1]


def constrain_3(x):
    x = x.reshape(3, 8)
    return -np.sum(x, axis=0)[2] + need[2]


def constrain_4(x):
    x = x.reshape(3, 8)
    return -np.sum(x, axis=0)[3] + need[3]


def constrain_5(x):
    x = x.reshape(3, 8)
    return -np.sum(x, axis=0)[4] + need[4]


def constrain_6(x):
    x = x.reshape(3, 8)
    return -np.sum(x, axis=0)[5] + need[5]


def constrain_7(x):
    x = x.reshape(3, 8)
    return -np.sum(x, axis=0)[6] + need[6]


def constrain_8(x):
    x = x.reshape(3, 8)
    return -np.sum(x, axis=0)[7] + need[7]


def constrain2_1(x):
    x = x.reshape(3, 8)
    return np.sum(x, axis=0)[0] - 0.75 * need[0]


def constrain2_2(x):
    x = x.reshape(3, 8)
    return np.sum(x, axis=0)[1] - 0.75 * need[1]


def constrain2_3(x):
    x = x.reshape(3, 8)
    return np.sum(x, axis=0)[2] - 0.75 * need[2]


def constrain2_4(x):
    x = x.reshape(3, 8)
    return np.sum(x, axis=0)[3] - 0.75 * need[3]


def constrain2_5(x):
    x = x.reshape(3, 8)
    return np.sum(x, axis=0)[4] - 0.75 * need[4]


def constrain2_6(x):
    x = x.reshape(3, 8)
    return np.sum(x, axis=0)[5] - 0.75 * need[5]


def constrain2_7(x):
    x = x.reshape(3, 8)
    return np.sum(x, axis=0)[6] - 0.75 * need[6]


def constrain2_8(x):
    x = x.reshape(3, 8)
    return np.sum(x, axis=0)[7] - 0.75 * need[7]


def constrain_A(x):
    x = x.reshape(3, 8)
    return -np.sum(x, axis=1)[0] + supply[0]


def constrain_B(x):
    x = x.reshape(3, 8)
    return -np.sum(x, axis=1)[1] + supply[1]


def constrain_C(x):
    x = x.reshape(3, 8)
    return -np.sum(x, axis=1)[2] + supply[2]
