import sympy as sp
import matplotlib.pyplot as plt
import numpy as np


P = sp.Matrix.zeros(9, 9) 

P[0, 4] = sp.Rational(1, 2); P[0, 6] = sp.Rational(1, 2)
P[1, 3] = sp.Rational(1, 2); P[1, 4] = sp.Rational(1, 2)
P[2, 3] = sp.Rational(1, 2); P[2, 4] = sp.Rational(1, 2)        # I know I could have truncated this into a 5x5 matrix...
P[3, 0] = sp.Rational(1, 2); P[3, 4] = sp.Rational(1, 2)
P[4, 0] = sp.Rational(1, 2); P[4, 6] = sp.Rational(1, 2)
P[5, 3] = sp.Rational(1, 2); P[5, 6] = sp.Rational(1, 2)
P[6, 3] = sp.Rational(1, 2); P[6, 8] = sp.Rational(1, 2)
P[7, 8] = sp.Rational(1, 1)
P[8, 8] = sp.Rational(1, 1)

print("Matrix P:")
sp.pprint(P)


# Part (i)

a = [0]*20 

for n in range(20):
    P_n = P**n
    a[n] = P_n[0, 8]

print(a)

# --- plot ---

x = list(range(1, 21))

plt.figure(figsize=(7, 5))
plt.plot(x, a, marker='o')

plt.xlabel("Number of turns")
plt.ylabel("Probability of winning")
plt.title("robability of winning vs. Number of turns")

plt.xticks(range(1, 21))
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# Part (ii)

u = sp.Matrix([i for i in range(1, 10)])   # initial distribution

print(u)

b = [0]*20

for n in range(20):
    b[n] = (P**n * u)[1]

print(b)

plt.figure(figsize=(7, 5))
plt.plot(x, b, marker='o')

plt.xlabel("Number of turns")
plt.ylabel("Average square number")
plt.title("Average square number vs. Number of turns")

plt.xticks(range(1, 21))
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# Part (iii)

I = sp.eye(9)

b = sp.ones(9, 1)

A = I - P 

absorb = [8]  # the absorbing state is the  8th (counting from 0)

for i in absorb:
    # Replace row i with the equation c_i = 0
    A[i, :] = sp.zeros(1, A.cols)
    A[i, i] = 1
    b[i] = 0

c = A.LUsolve(b)    # solving (I - P)c = id with c[8] = 0
print("Mean first passage time to reach 9 for 0:", c[0])
