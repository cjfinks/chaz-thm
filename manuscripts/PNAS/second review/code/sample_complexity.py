""" 
Compute the probability that for every S in H, there are at least k vectors x_i supported there which share a support in B 
"""
# TODO: Try alternative formula for bounded pigeonhole

import numpy as np
from scipy.misc import comb

def pigeonhole_bounded_alt(num_balls, num_bins, max_balls_per_bin):
    # TODO: supposed to work better for large N
    pass

def pigeonhole_bounded(num_balls, num_bins, max_balls_per_bin):
    """ 
    Returns the number of ways to allocate num_balls to num_bins without exceeding max_balls_per_bin in any bin

    Source URL:
    https://math.stackexchange.com/questions/553960/extended-stars-and-bars-problemwhere-the-upper-limit-of-the-variable-is-bounded
    """
    lim = int(min(num_bins, 1.*num_balls / (max_balls_per_bin+1)))
    val = 0
    for q in range(lim+1):
        sgn = (-1)**q
        term_1 = comb(num_bins, q)
        term_2 = comb(num_balls - q*(max_balls_per_bin + 1) + num_bins - 1, num_bins - 1)
        val += sgn * term_1 * term_2
    return val

def pigeonhole(num_balls, num_bins):
    """ 
    Returns the number of ways to allocate num_balls to num_bins.
    """
    return comb(num_balls + num_bins - 1, num_bins - 1)

def prob_atleast_k(num_balls, num_bins, k):
    """
    Returns the probability that some bin contains at least k balls
    """
    bnded = pigeonhole_bounded( num_balls, num_bins, max_balls_per_bin = k-1)
    tot = pigeonhole( num_balls, num_bins )
    return 1 - bnded / tot

"""
There are N_supp vectors supported on each S in H. These are the balls.
There are mCk k-sparse supports wrt. the n x m matrix B. These are the bins.
For what N_supp does the proof hold with a given probability?

We plot this as a function of m for fixed k.
Assume H is the set of contiguous length-k intervals of [m] arranged in cyclic order, so that |H| = m.
"""
k = 3 # Fixed sparsity, i.e. H is k-uniform
m_min = k+1 # since k < m required for H to satisfy SIP
m_max = 20 # Max dictionary size
prob = 0.999 # With what probability do we want the proof to work
N_probabilistic = [] # Number of samples for proof to work with probability prob
N_deterministic = [] # Number of samples for deterministic proof, i.e. with absolute certainty
for m in range(m_min, m_max+1):
    print(m)
    prob_each_supp_req = np.exp(np.log(prob) / m) # with what probability do we need the pigeonholing for each S in H to succeed so that pigeonholing for all S in H succeeds with probability prob (recall |H| = m)?
    N_supp_min = k # need at least k
    N_supp_max = int((k-1) * comb(m,k) + 1) # with N_supp_max per support or more, probability is 1
    N_deterministic.append( N_supp_max * m )
    for N_supp in range(N_supp_min, N_supp_max+1): 
        prob_supp = prob_atleast_k(num_balls = N_supp, num_bins = comb(m,k), k = k)
        if prob_supp >= prob_each_supp_req:
            N_probabilistic.append( N_supp * m ) # total number of data points required is N_supp * |H|
            break
    #N_supp_pcntile = next(i for i, prob in enumerate(probs) if prob > supp_pcntile) 

import matplotlib.pyplot as pp
#pp.plot(range(k+1,m_max), [Np / Nd for Np, Nd in zip(N_prob, N_det)])
fix, axes = pp.subplots(1,2)
axes[0].plot(range(m_min,m_max+1), N_probabilistic)
axes[0].plot(range(m_min,m_max+1), N_deterministic)
axes[1].plot(range(m_min,m_max+1), [Np / Nd for Np, Nd in zip(N_probabilistic, N_deterministic)])

"""
What is the probability that for every S in H, at least k out of N_supp vectors supported there have the same support wrt. B? 
"""
m = 16
k = 2
N_supp_min = 0
N_supp_max = int((k-1) * comb(m,k)) # with N_supp_max + 1 per support or more, probability is 1
probs = []
for N_supp in range(N_supp_min, N_supp_max+1):
    prob_supp = prob_atleast_k(num_balls = N_supp, num_bins = comb(m,k), k = k)
    probs.append( prob_supp ** m )

Ns = range(N_supp_min*m, m * (N_supp_max+1), m) # N_supp per support in H, and there are m supports in H
pp.figure(); pp.plot(Ns, probs)
