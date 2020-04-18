import numpy as np
import quadprog as qp
import matplotlib.pyplot as plt

def line(a, b, c, L=0, H=3): # draw a line ax+by+c=0 within range of x in [L,H]
    x1 = L
    y1 = (-c-a*x1)/b
    x2 = H
    y2 = (-c-a*x2)/b
    plt.plot([x1, x2], [y1, y2])

X = np.matrix([
    [2, 2],
    [3, 1],
    [1, 1]
])
Y = np.array([1, 1, -1])

plt.plot(X[0:2,0], X[0:2,1], 'bo')
plt.plot(X[2:3,0], X[2:3,1], 'rx')

G = np.matrix([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1E-6]
])
a = np.array([0., 0., 0.])
C = np.matrix([
    [2., 2., -1.],
    [3., 2., -1.],
    [-1., -1., 1.]
]).T
b = np.array([1., 1., 1.])

sol = qp.solve_qp(G, a, C, b)
wb = sol[0]
print(wb)

# wb[0] * x1 + wb[1] * x2 - wb[2] = 0
line(wb[0], wb[1], -wb[2])
# draw 2 'support' vector (much like: 'line')

# wb[0] * x1 + wb[1] * x2 - wb[2] = 1
line(wb[0], wb[1], -wb[2]-1)

# wb[0] * x1 + wb[1] * x2 - wb[2] = -1
line(wb[0], wb[1], -wb[2]+1)

# New coming element:
A = (0, 0)
if wb[0]*A[0]+wb[1]*a[1]-wb[2] >= 0:
    print(A,'belongs to positive class')
    plt.plot(A[0], A[1], 'bo')
else:
    print(A,'belongs to negative class')
    plt.plot(A[0], A[1], 'bx')
plt.show()
