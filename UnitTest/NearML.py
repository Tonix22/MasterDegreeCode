import numpy as np

def recursive_near_ml(yp, R, conste, nt):
    QRM = len(conste)  # Number of constellation points

    # Base case: Compute and return the last symbol directly
    if nt == 1:
        a_est = yp[0, 0] / R[0, 0]
        min_index = np.argmin(np.abs(a_est - conste)**2)
        return np.array([conste[min_index]], dtype=np.complex_)

    # Recursive case: Compute for current layer and recurse for the rest
    s_est = np.zeros(nt, dtype=np.complex_)  # Initialize the symbol estimate vector
    best_metric = np.inf

    for idx in range(QRM):
        # Estimate symbol for current layer
        a_est = yp[nt-1, 0] / R[nt-1, nt-1]
        sest = conste[idx]
        metric = np.abs(a_est - sest)**2

        # Update the received vector for the next recursion
        new_yp = yp.copy()
        new_yp[:nt, 0] -= sest * R[:nt, nt-1]

        # Recursive call to process the remaining layers
        s_est_next = recursive_near_ml(new_yp[:nt-1], R[:nt-1, :nt-1], conste, nt - 1)

        # Combine current estimate with results from recursive call
        current_metric = metric + np.linalg.norm(new_yp[:nt-1] - R[:nt-1, nt-1] * sest)**2
        if current_metric < best_metric:
            best_metric = current_metric
            s_est = np.concatenate((s_est_next, [sest]))  # Append the current symbol estimate

    # Return the estimated symbol vector, maintaining the correct order
    return s_est

# Example usage:
# Define your system parameters
yp = np.array([[1+1j], [2+2j], [3+3j]])  # Example received vector
R = np.array([[1, 0.1, 0], [0, 1, 0.1], [0, 0, 1]])  # Example upper triangular matrix
conste = np.array([-1, 1])  # Example constellation for BPSK
nt = 3  # Number of layers

# Call the function
s_est = recursive_near_ml(yp, R, conste, nt)
print("Estimated symbols:", s_est)
