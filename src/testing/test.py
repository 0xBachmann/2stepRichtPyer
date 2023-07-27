import numpy as np

N_pad = 5
M = 6
D = 3
g = np.random.random((M, M, M))

u = np.random.random((M, N_pad, N_pad, D))
sums = np.zeros((M, N_pad, N_pad, D, D))
for m in range(M):
    for n in range(M):
        for q in range(M):
            for d1 in range(D):
                for d2 in range(D):
                    sums[q, ..., d1, d2] += u[m, ..., d1] * u[n, ..., d2] * g[m, n, q]

            # assert np.allclose(np.einsum("...i,...j,->...ij", u[m], u[n], g[m, n, q]), sums[q])
            # sums[q] += np.einsum("...i,...j,->...ij", u[m], u[n], g[m, n, q])  # + missing?

assert np.allclose(np.einsum("m...d,n...b,mnq->q...db", u, u, g), sums)
