import numpy as np

def power(x, times = 20):
    tmp = x
    for i in range(times):
        tmp = tmp @ x
    return tmp

if __name__ == '__main__':
    a = np.random.rand(5, 5)
    print(a)
    u, s, v = np.linalg.svd(a)
    b = a/np.max(s)
    print(power(a))
    print(power(b))