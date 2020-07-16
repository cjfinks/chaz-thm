"""
Compute the probability that for every S in H, there are at least k vectors x_i supported there which share a support in B
"""
# TODO: Try alternative formula for bounded pigeonhole

import numpy as np
from scipy.special import comb

COMB_EXACT = True

def pigeonhole_bounded_large_N(N, n, r):
    """
    Supposed to be better for large N

    Source URL:
    https://math.stackexchange.com/questions/553960/extended-stars-and-bars-problemwhere-the-upper-limit-of-the-variable-is-bounded
    """
    val = 0
    lo = int(np.ceil( (N+n) / (r+1) ))
    hi = int(n)
    for q in range(lo, hi+1):
        sgn = (-1)**(n-q)
        term_1 = comb(n, q, exact=COMB_EXACT)
        term_2 = comb(q*(r+1) - 1 - N, n - 1, exact=COMB_EXACT)
        val += sgn * term_1 * term_2
    return val

def pigeonhole_bounded(N, n, r):
    """
    Returns the number of ways to allocate N indistinguishable balls to n distinct bins with at most r balls per bin

    Source URL:
    https://math.stackexchange.com/questions/553960/extended-stars-and-bars-problemwhere-the-upper-limit-of-the-variable-is-bounded
    """
    val = 0
    lim = int(min(n, 1.*N / (r+1)))
    for q in range(lim+1):
        sgn = (-1)**q
        term_1 = comb(n, q, exact=COMB_EXACT)
        term_2 = comb(N - q*(r + 1) + n - 1, n - 1, exact=COMB_EXACT)
        val += sgn * term_1 * term_2
    return val

def pigeonhole(N, n):
    """
    Returns the number of ways to allocate N indistinguishable balls to n distinct bins.
    """
    return comb(N + n - 1, n - 1, exact=COMB_EXACT)

def prob_atleast_k_balls_in_some_bin(num_balls, num_bins, k):
    """
    Returns the probability that some bin contains at least k balls
    """
    num_ways_all_less_than_k = pigeonhole_bounded(num_balls, num_bins, k-1) # number of ways to allocate balls so that every bin has fewer than k balls
    #bnded = pigeonhole_bounded_large_N( num_balls, num_bins, k-1) # number of ways to allocate balls so that bin contains k or more balls
    num_ways = pigeonhole(num_balls, num_bins) # number of ways to allocate balls
    #if (num_ways_all_less_than_k / num_ways < 0) or (num_ways_all_less_than_k / num_ways > 1):
        #import pdb; pdb.set_trace()
    return 1 - num_ways_all_less_than_k / num_ways # probability that at least one of them has k or more

if __name__ == '__main__':
    import matplotlib.pyplot as pp
    import os
    try:
        os.mkdir('./figures')
    except:
        pass
    pp.ion()

    """
    Given N_supp samples per support, what is the probability that for every S in H,
    at least k out of N_supp vectors supported there have the same support wrt. B?
    """

    ms = [m for m in range(3, 12)]
    fig, ax = pp.subplots(len(ms),len(ms))
    fig.tight_layout()
    for i, m in enumerate(ms):
        for j, k in enumerate(ms):
            ax[i][j].xaxis.set_visible(False)
            ax[i][j].yaxis.set_visible(False)
            ax[i][j].set_visible(False)

    for i, m in enumerate(ms):
        ks = [k for k in range(2, m)]
        for j, k in enumerate(ks):
            print("%d" % k)
            samples_per_supp_deterministic = int((k-1) * comb(m,k)) + 1 # with N_deterministic per support or more, probability is 1 (deterministic case)
            probs = []
            for samples_per_supp in range(0, samples_per_supp_deterministic+1):
                prob_supp = prob_atleast_k_balls_in_some_bin(num_balls = samples_per_supp, num_bins = comb(m,k), k = k) # probability of success
                num_supps = m # assume H is cyclic order hypergraph, i.e. |H| = m
                probs.append( prob_supp ** num_supps ) # must succeed for every S in H

            #Ns = range(0, num_supps * samples_per_supp_deterministic, num_supps) # deterministic N is samples_per_supp_deterministic * n_supps, and there are m supports
            Ns = np.arange(0, samples_per_supp_deterministic+1) / samples_per_supp_deterministic  # relative to deterministic sample complexity
            ax[i][j].plot(Ns, probs)
            ax[i][j].plot(Ns, probs)
            ax[i][j].set_visible(True)
            ax[i][j].xaxis.set_visible(False)
            ax[i][j].yaxis.set_visible(False)
            ax[i][j].spines['right'].set_visible(False)
            ax[i][j].spines['top'].set_visible(False)

    ax[-1][0].xaxis.set_visible(True)
    for i, m in enumerate(ms):
        ax[i][0].yaxis.set_visible(True)
        ax[i][0].set_ylabel('$m$=%d' % m)
        

    ax[-1][0].yaxis.set_visible(True)
    for j, k in enumerate(ks):
        ax[-1][j].xaxis.set_visible(True)
        ax[j][j].set_title('$k$=%d' % k)
    
    #pp.title('Probability of successful recovery for sample // sizes below deterministic sample complexity')
    #pp.set_xlabel('Fraction of deterministic sample size')
    #pp.set_ylabel('Pr(success)')
    #ax.legend(["m=%d" % m for m in ms])
    pp.savefig('./figures/prob_vs_samples.png')