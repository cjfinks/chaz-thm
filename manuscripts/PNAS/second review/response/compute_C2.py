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
    nz_angles = angles > np.finfo(float).eps
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

def cyclic_hypergraph(m, k):
    H = []
    for i in range(m - k + 1):
        H.append(list(range(i, i + k)))
    for i in range(1, k):
        j = k - i
        H.append(list(range(m - j, m)) + list(range(0, k - j)))
    return H

if __name__ == '__main__':
    # SQUARE GRID
    num_trials = 100
    r = 2
    Ks = [4, 8, 16]
    for k in Ks:
        m = k**2
        n = m;
        square_grid = np.reshape(range(m),(k,k))
        H = [row.tolist() for row in square_grid] + [col.tolist() for col in square_grid.T]
        N_per_support = int((k - 1) * choose(m, k))
        N = len(H) * N_per_support

        C2s = np.zeros(num_trials)
        for i in range(num_trials):
            print("Trial %d" % i)
            A = np.random.randn(n, m)
            A = np.dot( A, np.diag(1./np.linalg.norm(A, axis=0)) ) # normalize
            C2s[i] = C2(A, H, r)

        import matplotlib.pyplot as pp
        pp.ion()
        pp.figure()
        pp.hist(C2s, bins=40)
        pp.show()
        pp.title('Grid C2s (m=%d, trials=%d)' % (m, num_trials))
        pp.savefig('Grid_C2_m%d_nt%d.pdf' % (m, num_trials))
