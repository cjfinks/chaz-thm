from decomp_svd import subspace_angles
import numpy as np
import itertools
import operator as op
from scipy.misc import comb as choose

def friedrichs(U,V):
    """
    Calculates the Friedrichs angle between subspaces U and V (in radians).

    The Friedrichs angle is the smallest nonzero principle angle.

    Alternatively, it is the (s+1)th principal angle when dim(U \cap V) = s.
    """
    angles = subspace_angles(U,V)
    nz_angles = angles > 0  # np.finfo(float).eps
    if np.any(nz_angles):
        min_nz_angle = np.sort(angles[nz_angles])[0]
        return min_nz_angle
    else:
        return 0.

def xi(A, G):
    """
    Calculates xi({A_S}_{S \in G})

    (Assumes A is injective on H, so the intersection of the span is the span of the intersection.)
    """
    if len(G) == 1:
        return 1 # by definition
    Gs = itertools.permutations(G)
    prods = []
    for Gseq in Gs: # loop through all orderings of elements in G
        prod = 1.
        for i, Si in enumerate(Gseq[:-1]):
            cap_Sj = list(set.intersection(*map(set, Gseq[i+1:])))
            if cap_Sj: # otherwise they intersect only at 0 and sin(theta) = 1, since every subspace forms pi/2 angle with 0 (weird)
                theta = friedrichs(A[:,Si], A[:,cap_Sj])
                prod *=  np.sin(theta)**2
        prods.append(prod)
    return 1 - np.sqrt(1 - np.max(prods))

def C2(A, H, r):
    """
    Ouputs the constant C2

    A: n x m dictionary matrix (2d numpy array)
    H: hypergraph (list of lists in range(m))
    r: regularity of H

    Loops over subsets of H of size r+1
    """
    xi_Gs = [] # xi is in (0,1]
    for G in itertools.combinations(H, r + 1): # loop over subsets of H of size r
        xi_Gs.append( xi(A, G) )
    max_A = np.max( np.linalg.norm(A, axis=0) )
    return len(H) * max_A / min(xi_Gs)

def L_k(M, k, n_samples = None):
    vals = []
    if n_samples is None:
        supports = itertools.combinations(range(M.shape[1]), k)
    else:
        supports = [np.random.choice(M.shape[1],k) for i in range(n_samples)]
    for S in supports:
        u, s, v = np.linalg.svd(M[:,S])
        vals.append(s[-1])
    return np.percentile(vals, 1 ) #min(np.array(vals)[np.array(vals) > 1e-14]) # #min(vals)

def C1_denom(A, Xs, H, num_rand_ksets = None):
    """
    Ouputs the denominator in the def of constant C1

    ************
    ASSUMES: first (k-1)(m choose k) are supported on H[0],
             second (k-1)(m choose k) are supported on H[1],
             ...
             so that N = |H|(k-1)(m choose k)
    ************

    A: n x m dictionary matrix (2d numpy array)
    X: m x N matrix of N k-sparse codes
    H: hypergraph (list of lists in range(m))
    k: sparsity k of support sets
    """
    k = len(H[0])
    m = A.shape[1]
    min_val = np.inf
    for S, X in zip(H, Xs):
        AX = np.dot(A[:,S], X)
        lk = L_k(AX, k, n_samples = num_rand_ksets)
        min_val = min(min_val, lk)
    return min_val

def cyclic_hypergraph(m, k):
    H = []
    for i in range(m - k + 1):
        H.append(list(range(i, i + k)))
    for i in range(1, k):
        j = k - i
        H.append(list(range(m - j, m)) + list(range(0, k - j)))
    return H

if __name__ == '__main__':
    #np.random.seed(0)
    """
    H = [ # cyclic order
        [0,1,2],
        [1,2,3],
        [2,3,0],
        [3,0,1]
        ]
    r = 3
    """
    """ rows and cols of square grid """

    # SQUARE GRID
    r = 2
    Ks = [4, 8, 16]
    for k in Ks:
        m = k**2
        n = m
        square_grid = np.reshape(range(m),(k,k))
        H = [row.tolist() for row in square_grid] + [col.tolist() for col in square_grid.T]
        N_per_support = int((k - 1) * choose(m, k))
        N = len(H) * N_per_support
        num_trials = 100
        Cs = np.zeros((num_trials, 2))  # 0:C1, 1:C2
        #pcntiles = np.zeros((num_trials, 2))  # 0:C1, 1:C2
        for i in range(num_trials):
            print("Trial %d" % i)
            A = np.random.randn(n, m)
            A = np.dot( A, np.diag(1./np.linalg.norm(A, axis=0)) ) # normalize
            # Xs = [np.random.randn(k, N_per_support) for S in H]
            # Xs = [np.dot(X, np.diag(1. / np.linalg.norm(X, axis=0))) for X in Xs] # normalize
            Cs[i, 1] = C2(A, H, r)
            # Cs[i, 0] = Cs[i,1] / C1_denom(A, Xs, H, num_rand_ksets = 10)
            #Cs[i,0] = Cs[i,1] / L_k( np.dot(A[:,H[0]], Xs[0]), k, n_samples = 10 )
           # print('%1.3f' % c2)
           # if (i > 1000) & (i % 100 == 0):
           #     print(i)
           #     pcntiles[i, 1] = np.percentile(Cs[:i, 1], 95)

        import matplotlib.pyplot as pp
        pp.ion()
        pp.figure()
        C2s = Cs[:, 1]
        pp.hist(C2s, bins=40)
        pp.show()
        pp.title('Grid C2s (m=%d, trials=%d)' % (m, num_trials))
        # plt.savefig('Grid_C2_m%d_nt%d.pdf' % (m, num_trials))

#     # CYLIC
#     Ks = [4, 8] # , 16, 32]
#     for k in Ks:
#         r = k
#         m = k ** 2
#         n = m
#         H = cyclic_hypergraph(m, k)
#         N_per_support = int((k - 1) * choose(m, k))
#         N = len(H) * N_per_support
#         num_trials = 5
#         Cs = np.zeros((num_trials, 2))  # 0:C1, 1:C2
#         #pcntiles = np.zeros((num_trials, 2))  # 0:C1, 1:C2
#         for i in range(num_trials):
#             print("Trial %d" % i)
#             A = np.random.randn(n, m)
#             A = np.dot( A, np.diag(1./np.linalg.norm(A, axis=0)) ) # normalize
#             # Xs = [np.random.randn(k, N_per_support) for S in H]
#             # Xs = [np.dot(X, np.diag(1. / np.linalg.norm(X, axis=0))) for X in Xs] # normalize
#             Cs[i, 1] = C2(A, H, r)
#             # Cs[i, 0] = Cs[i,1] / C1_denom(A, Xs, H, num_rand_ksets = 10)
#             #Cs[i,0] = Cs[i,1] / L_k( np.dot(A[:,H[0]], Xs[0]), k, n_samples = 10 )
#            # print('%1.3f' % c2)
#            # if (i > 1000) & (i % 100 == 0):
#            #     print(i)
#            #     pcntiles[i, 1] = np.percentile(Cs[:i, 1], 95)
#
#         import matplotlib.pyplot as pp
#         pp.ion()
#         pp.figure()
#         C2s = Cs[:, 1]
#         pp.hist(C2s, bins=40)
#         pp.show()
#         pp.title('Cyclic C2s (m=%d)' % m)
#         # plt.savefig('Cyclic_C2_m%d.pdf' % m)
#
#     # pp.figure()
#     # C1s = Cs[:, 0]
#     # pp.hist(C1s, bins=100)
#     # pp.show()
#     # pp.title('C1s')
#
#     # pp.figure();
#     # pp.plot(pcntiles[:, 1])
#     # pp.title('C2s 95 Percentiles over trials')  # looks funny for some reason
#
# ### NOTE $$$
# # What is RIP of n by (k-1)* mCk matrix?
