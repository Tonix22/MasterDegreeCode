import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[3, 4], [5, 6]])
r1 = a+1j*b

c = np.array([[1, 2], [3, 4]])
d = np.array([[3, 4], [5, 6]])
r2 = c+1j*d

r3 = r1@r2
print(r3)

print("********")

r4 = (a@c-b@d)+1j*(a@d+b@c)
print(r4)