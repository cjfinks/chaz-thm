from decomp_svd import subspace_angles
import numpy as np
import itertools

def friedrichs(U,V):
    """
    Calculates the Friedrichs angle between subspaces U and V (in radians).

    The Friedrichs angle is the smallest nonzero principle angle.

    Alternatively, it is the (s+1)th principal angle when dim(U \cap V) = s.
    """
    angles = subspace_angles(U,V)
    nz_angles = angles > 1e-10 #> np.finfo(float).eps
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

#def powerset(s):
#    return [[x for j, x in enumerate(s) if (i >> j) & 1] for i in range(2**len(s))]

def C_2(A, H, r):
    """
    Ouputs the constant C2

    A: n x m dictionary matrix (2d numpy array)
    H: hypergraph (list of lists in range(m))
    r: regularity of H
    """
    xi_Gs = [] # xi is in (0,1]
    for G in itertools.combinations(H, r): # loop over subsets of H of size r
        xi_Gs.append( xi(A, G) )
    max_A = np.max( np.linalg.norm(A, axis=0) )
    return len(H) * max_A / min(xi_Gs)

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
    H = [ # rows and cols of square array
        [0,1,2,3],
        [4,5,6,7],
        [8,9,10,11],
        [12,13,14,15],
        [0,4,8,12],
        [1,5,9,13],
        [2,6,10,14],
        [3,7,11,15]
        ]
    r = 2
    c2s = []
    pcntiles = []
    for i in range(10000):
        #A = np.eye(9,9)\
        A = np.random.randn(16,16)
        A = np.dot( A, np.diag(1./np.linalg.norm(A,axis=0)) ) # normalize
        c2 = C_2(A, H, r)
        #print('%1.3f' % c2)
        c2s.append(c2)
        if (i > 1000) & (i % 100 == 0):
            print(i)
            pcntiles.append( np.percentile(c2s,95) )

    import matplotlib.pyplot as pp
    pp.ion()
    c2s = np.array(c2s)
    pp.hist(c2s[c2s<1000], bins=40)
    pp.show()
    pp.figure();
    pp.plot(pcntiles)
