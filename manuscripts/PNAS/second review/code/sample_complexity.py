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

if __name__ == '__main__':
    import matplotlib.pyplot as pp
    pp.ion()

    """
    Given N_supp samples per support, what is the probability that for every S in H,
    at least k out of N_supp vectors supported there have the same support wrt. B?
    """
    m = 16
    k = 2
    num_supps = m # cyclic order hypergraph
    samples_per_supp_deterministic = int((k-1) * comb(m,k)) + 1 # with N_deterministic per support or more, probability is 1 (deterministic case)
    probs = []
    for samples_per_supp in range(samples_per_supp_deterministic):
        prob_supp = prob_atleast_k(num_balls = samples_per_supp, num_bins = comb(m,k), k = k) # probability of success
        probs.append( prob_supp ** num_supps ) # must succeed for every S in H

    Ns = range(0, num_supps * samples_per_supp_deterministic, m) # N_supp per support in H, and there are m supports in H
    fig, ax = pp.subplots(1,1)
    ax.plot(Ns, probs)
    ax.set_title('Probability that proof works vs. sample size (m=16, k=3)')
    ax.set_xlabel('Sample size (up to deterministic sufficient number of samples)')
    ax.set_ylabel('Prob( proof works )')
    pp.show()

    """
    There are N_supp vectors supported on each S in H. These are the balls.
    There are mCk k-sparse supports wrt. the n x m matrix B. These are the bins.
    For what N_supp does the proof hold with a given probability?

    We plot this as a function of m for fixed k.
    Assume H is the set of contiguous length-k intervals of [m] arranged in cyclic order, so that |H| = m.
    """
    k = 3 # Fixed sparsity, i.e. H is k-uniform
    num_supps = m # cyclic order hypergraph
    m_min = k+1 # since k < m required for H to satisfy SIP
    m_max = 20 # Max dictionary size
    prob = 0.999 # With what probability do we want the proof to work
    samples_probabilistic = [] # Number of samples for proof to work with probability prob
    samples_deterministic = [] # Number of samples for deterministic proof, i.e. with absolute certainty
    for m in range(m_min, m_max+1):
        print(m)
        prob_each_supp_req = np.exp(np.log(prob) / m) # with what probability do we need the pigeonholing for each S in H to succeed so that pigeonholing for all S in H succeeds with probability prob (recall |H| = m)?
        samples_per_supp_min = k # need at least k
        samples_per_supp_deterministic = int((k-1) * comb(m,k) + 1) # with N_supp_max per support or more, probability is 1
        samples_deterministic.append( samples_per_supp_deterministic * num_supps )
        for samples_per_supp in range(samples_per_supp_min, samples_per_supp_deterministic + 1):
            prob_supp = prob_atleast_k(num_balls = samples_per_supp, num_bins = comb(m,k), k = k)
            if prob_supp >= prob_each_supp_req:
                samples_probabilistic.append( samples_per_supp * num_supps ) # total number of data points required is N_supp * |H|
                break
        #N_supp_pcntile = next(i for i, prob in enumerate(probs) if prob > supp_pcntile)

    #pp.plot(range(k+1,m_max), [Np / Nd for Np, Nd in zip(N_prob, N_det)])
    fig, ax = pp.subplots(1,1)
    ax.plot(range(m_min,m_max+1), samples_deterministic)
    ax.plot(range(m_min,m_max+1), samples_probabilistic)
    ax.legend(['100% guarantee', '99.9% guarantee'])
    ax.set_title('Sufficient sample size for probabilistic and deterministic guarantees')
    ax.set_xlabel('Number of dictionary elements m (k=3)')
    ax.set_ylabel('Sufficient sample size')

    #axes[1].plot(range(m_min,m_max+1), [Np / Nd for Np, Nd in zip(N_probabilistic, N_deterministic)])
