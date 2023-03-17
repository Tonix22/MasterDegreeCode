import torch
import numpy as np
import sys

def Near_ML(yp, R, conste, index):
    # Preprocessing for Near_ML detection
    row, nt = R.shape
    QRM = len(conste)

    # Initialization parameters
    M = 4
    s_est = torch.zeros(nt, dtype=torch.complex128)
    sest3 = torch.zeros(nt, dtype=torch.complex128)
    sest2 = torch.zeros(QRM)
    
    acu = 0
    nodos = 0

    parent_yp = torch.zeros((nt, QRM), dtype=torch.complex128)
    parent_yp2 = torch.zeros((nt, M), dtype=torch.complex128)
    parent_yp2t = torch.zeros((nt, M), dtype=torch.complex128)
    parent_node = torch.zeros((1, 2 * QRM), dtype=torch.complex128)
    parent_node2 = torch.zeros((nt, 2 * M), dtype=torch.complex128)
    parent_node2t = torch.zeros((1, 2 * M), dtype=torch.complex128)
    vector = torch.zeros(QRM * M)
    pos = torch.zeros(QRM * M)
    x1 = torch.zeros((48, 2 * M * QRM), dtype=torch.complex128)
    distc = torch.zeros((M * QRM), dtype=torch.complex128)
    distc2 = torch.zeros(QRM)
    ordtotal = torch.zeros(M * QRM)

    # Detection at level nt
    a_est = yp[nt - 1] / R[nt - 1, nt - 1]
    sest2 = torch.abs(a_est - conste) ** 2
    dist, ord = torch.sort(sest2)
    row = 0
    for p in range(QRM):
        parent_node[:, row:row + 2] = torch.tensor([dist[p], conste[ord[p]]], dtype=torch.complex128).unsqueeze(0)
        parent_yp[:, p] = yp
        row += 2

    indice = 0
    dmin = sys.float_info.max + 1j * sys.float_info.max
    skip = 0
    row = 0

    for n in range(QRM):
        # Estimation of the nt-1 levels
        for k in range(nt - 1, 0, -1):
            if k == nt - 1:
                distp = parent_node[0, row]
                sest = parent_node[0, row + 1]
                rm = parent_yp[:, n]
                rm = (rm - sest * R[:, k])
                a_est = rm[k - 1] / R[k - 1, k - 1]
                sest2 = (torch.abs(a_est - conste) ** 2)
                distc2 = distp + sest2
                
                # Convert the tensor to a NumPy array
                distc2_np = distc2.numpy()
                # Get the sorted indices using NumPy's argsort function
                ord2 = np.argsort(distc2_np)
                # Get the sorted values using the sorted indices
                dist2 = distc2_np[ord2]
                # Convert the indices and sorted values back to PyTorch tensors if needed
                ord2  = torch.from_numpy(ord2)
                dist2 = torch.from_numpy(dist2)
                
                col = 0
                for p in range(M):
                    parent_node2[k, col:col + 2] = parent_node[0, row:row + 2]
                    parent_node2[k - 1, col:col + 2] = torch.tensor([dist2[p], conste[ord2[p]]], dtype=torch.complex128).unsqueeze(0)
                    parent_yp2[:, p] = rm
                    col += 2
                row += 2
            else:
                acu = 0
                col = 0
                for p in range(M):
                    distp = parent_node2[k, col]
                    sest = parent_node2[k, col + 1]
                    rm = parent_yp2[:, p]
                    rm = (rm - sest * R[:, k])
                    a_est = rm[k - 1] / R[k - 1, k - 1]
                    sest2 = (torch.abs(a_est - conste) ** 2)
                    distc[acu:acu + QRM] = distp + sest2
                    vector[acu:acu + QRM] = torch.ones(QRM) * p
                    pos[acu:acu + QRM] = torch.arange(QRM)
                    parent_yp2[:, p] = rm
                    acu += QRM
                    col += 2
                
                # Convert the tensor to a NumPy array
                distc_np = distc.numpy()
                # Get the sorted indices using NumPy's argsort function
                ord3 = np.argsort(distc_np)
                # Get the sorted values using the sorted indices
                dist3 = distc_np[ord3]
                # Convert the indices and sorted values back to PyTorch tensors if needed
                ord3 = torch.from_numpy(ord3)
                dist3 = torch.from_numpy(dist3)

                if dist3[0].item().real > dmin.real and dist3[0].item().imag > dmin.imag:
                    skip = 1
                    break
                
                if skip == 0:
                    parent_node2t = parent_node2.clone()
                    parent_yp2t = parent_yp2.clone()
                    col = 0
                    for p in range(M):
                        parent_node2[k - 1, col:col + 2] = torch.tensor([dist3[p], conste[int(pos[ord3[p].item()].item())]], dtype=torch.complex128).unsqueeze(0)
                        if vector[ord3[p]] != p:
                            temp = int((2 * vector[ord3[p].item()]))
                            parent_node2[k:nt, col:col + 2] = parent_node2t[k:nt, temp:temp+1]
                            parent_yp2[:, p] = parent_yp2t[:, int(vector[ord3[p].item()].item())]
                        col += 2
        if skip == 0:
            # Store the M best of the n-th iteration
            x1[:, indice:indice + (2 * M)] = parent_node2
            indice += (2 * M)
            dtotal = x1[0, 0:2:indice - 1]
            dmin = complex(np.min(dtotal.numpy()))
            nodos += M

        # Check if a new tree is opened
        if n < QRM - 1:
            distp = parent_node[0, row]
            if distp.item().real > dmin.real and distp.item().imag > dmin.imag:
                break
            else:
                skip = 0

    # Determine the vector with the minimum distance
    dtotal = x1[0, 0:2:indice - 1]
    contador = 0
    #ordtotal = torch.zeros(((indice - 1) // 2), dtype=torch.int64)
    for k in range(1, indice - 1, 2):
        ordtotal[contador] = k
        contador += 1

    # Convert the tensor to a NumPy array
    dtotal_np = dtotal.numpy()

    # Get the sorted indices using NumPy's argsort function
    ordmin = np.argsort(dtotal_np)

    # Get the sorted values using the sorted indices
    dminf = dtotal_np[ordmin]

    # Convert the indices and sorted values back to PyTorch tensors if needed
    ordmin = torch.from_numpy(ordmin)
    dminf  = torch.from_numpy(dminf)
    
    sest3 = x1[:, int(ordtotal[ordmin[0].item()].item())]

    for k in range(nt):
        s_est[index[k]] = sest3[k]
        
    return s_est

