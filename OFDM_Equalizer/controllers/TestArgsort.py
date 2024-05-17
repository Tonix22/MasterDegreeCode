import numpy as np

# Create a complex array
complex_array = np.array([complex(1, 2), complex(2, -1), complex(-1, 1)])

# Attempt to find the minimum
try:
    result = np.min(complex_array)
    print("Minimum:", result)
except TypeError as e:
    print("Error:", str(e))
