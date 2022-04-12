import numpy as np
def to_stochastic(a):
    for i in range(a.shape[0]):
        a[i]/= np.sum(a[i])
    return a

if __name__ == '__main__':
    a = np.zeros([3,3,3])
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i==j and j==k:
                    a[i,j,k] = 1

    b = np.arange(1, 10).reshape(3,3)
    c = np.arange(10, 19).reshape(3,3)

    print(b)
    print(c)
    print(np.einsum('qp, ij, pjk -> qik', b, c, a))
