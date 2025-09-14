import sympy as sp

# index set I = {0, ..., 5}
I = list(range(0, 6))
n = len(I)

x = sp.IndexedBase('x')

# A:  -1 on diagonal, neighbors = 1/2,
# first row right neighbor = 1, last row left neighbor = 1/2
A = sp.Matrix.zeros(n, n)
for r, i in enumerate(I):          # r = 0..n-1, i = 0..5
    for c, j in enumerate(I):      # c = 0..n-1, j = 0..5
        if 0 < r < n-1:  # interior row
            A[r, c] = (sp.Rational(1, 2) if (j == i+1 or j == i-1)
                        else (-1 if j == i else 0))
        elif r == 0:     # first row
            A[r, c] = 1 if (j == i+1) else (-1 if j == i else 0)
        else:            # last row (r == n-1)
            A[r, c] = (sp.Rational(1, 2) if (j == i-1)
                        else (-1 if j == i else 0))

print("Matrix A:")
sp.pprint(A)

# RHS: constant -1 for each i in I
b_vec = sp.Matrix([-1] * n)

# unknown vector ordered by I
x_vec = sp.Matrix([x[i] for i in I])

# solve A x = b symbolically
sol_vec = A.LUsolve(b_vec)

print(sol_vec)
