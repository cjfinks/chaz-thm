#from decomp_svd import subspace_angles
from scipy.linalg.decomp_svd import subspace_angles
import numpy as np
import itertools
import operator as op
from scipy.special import comb as choose

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

def square_hypergraph(m, k):
    assert m == k**2
    square_grid = np.reshape(range(m),(k,k))
    H = [row.tolist() for row in square_grid] + [col.tolist() for col in square_grid.T]
    return H


if __name__ == '__main__':
    import matplotlib.pyplot as pp
    np.random.seed(0)

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

    ks = range(4,9)
    fig, ax = pp.subplots(len(ks), 1, figsize=(20,10), sharex=True)
    fig.add_subplot(111, frameon=False)
    pp.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    pp.ylabel("Histogram of $C_2(\mathbf{A}, \mathcal{H})$")
    for i, k in enumerate(ks):
        print("k=%d" % k)
        # SQUARE
        m = k**2
        n = (3*m)//4
        H = square_hypergraph(m, k)
        r = 2 # since SIP satisfied by (row,col) pair

        # CIRCLE
        #k = 3 # choose something less than m
        #H = cyclic_hypergraph(m, k)
        #r = k

        N_per_support = int((k - 1) * choose(m, k, exact=True)) + 1
        N = len(H) * N_per_support
        num_trials = 1000
        c2s = np.zeros(num_trials)
        for t in range(num_trials):
            print("Trial %d" % t, end='\r')
            A = np.random.randn(n, m)
            A = np.dot( A, np.diag(1./np.linalg.norm(A, axis=0)) ) # normalize
            c2s[t] = C2(A, H, r)

        c2s = np.sort(c2s[c2s < np.inf])
        c2max = 1000 #c2s[int(0.95*len(c2s))]
        ax[i].hist(c2s[c2s<c2max], bins=100)
        ax[i].legend(["$m=%d$" % m], loc='upper right', handlelength=0, handletextpad=0)
        pp.savefig('C2.pdf')
        pp.show()
        # TODO: do relative histogram so as to share y axis