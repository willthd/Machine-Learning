import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([1, 1])
    b = 1
    tmp = np.sum(x*w) - b
    if tmp >= 1:
        return 1
    else:
        return 0

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([1, 1])
    b = 1
    tmp = np.sum(x*w) - b
    if tmp < 1:
        return 1
    else:
        return 0

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([1, 1])
    b = 0.5
    tmp = np.sum(x*w) - b
    if tmp >= 0:
        return 1
    else:
        return 0

def XOR(x1, x2):
    d1 = NAND(x1, x2)
    d2 = OR(x1, x2)
    return AND(d1, d2)


data = np.array([[0,0],[0,1],[1,0],[1,1]])

for i in range(4):
    if i == 0:
        for value in data:
            print(AND(value[0], value[1]))
    elif i == 1:
        for value in data:
            print(NAND(value[0], value[1]))
    elif i == 2:
        for value in data:
            print(OR(value[0], value[1]))
    else:
        for value in data:
            print(XOR(value[0], value[1]))
    print("-------")