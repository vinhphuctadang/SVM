import matplotlib.pyplot as plt
import numpy as np
import quadprog as qp
def main():
    X = np.array([
        [1., 1.],
        [2., 2.],
        [3., 3.]
    ])

    m, n= X.shape
    c = 10
    epsilon = 0.3

    G = np.diag([1.] * n + [1e-6]*(2*m+1))
    a = np.array([0.] * (n+1) + [-c*1.0] * (2*m))

    I = np.diag([1.0]*m)
    b = np.array([1.] * m).reshape((m,1))

    C = np.vstack([
        np.hstack([-X, -b, I, np.zeros((m, m))]),
        np.hstack([X, b, np.zeros((m, m)), I]),
        np.hstack([np.zeros((m,n+1)), I, np.zeros((m, m))]),
        np.hstack([np.zeros((m,n+1)), np.zeros((m, m)), I]),
    ])

    h = np.array([-epsilon*1.0] * m + [epsilon*1.0] * m + [0.] * (2*m))

    print('G',G)
    print('a',a)
    print('C',C)
    print('h',h)
    solution = qp.solve_qp(G, a, C.T, h)

    wb = solution(solution[0])

    plt.scatter(X[:,0], X[:,1], 'bo')
    plt.show()
    pass

if __name__ == '__main__':
    main()
