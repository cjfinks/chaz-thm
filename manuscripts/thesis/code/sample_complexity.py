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

    #H = 'cyclic'
    H = 'square'

    """
    Given N_supp samples per support, what is the probability that for every S in H,
    at least k out of N_supp vectors supported there have the same support wrt. B?
    """

    if H == 'cyclic':
        ms = [m for m in range(3, 10+1)]
    elif H == 'square':
        ms = [k**2 for k in range(2, 4+1)] # square
    
    fig, ax = pp.subplots(len(ms), 1, sharex=True, sharey=True, figsize=(11,8.5))
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    pp.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    pp.ylabel("Probability of successful recovery")
    #pp.ylabel("common Y")
    fig.tight_layout()

    for i, m in enumerate(ms):
        if H == 'cyclic':
            ks = [k for k in range(2, m)]
        elif H == 'square':
            ks = [int(np.sqrt(m))]
        for k in ks:
            print("%d" % k)
            samples_per_supp_deterministic = int((k-1) * comb(m,k)) + 1 # with N_deterministic per support or more, probability is 1 (deterministic case)
            probs = []
            for samples_per_supp in range(0, samples_per_supp_deterministic+1):
                prob_supp = prob_atleast_k_balls_in_some_bin(num_balls = samples_per_supp, num_bins = comb(m,k), k = k) # probability of success
                if H == 'cyclic':
                    num_supps = m # assume H is cyclic order hypergraph, i.e. |H| = m
                elif H == 'square':
                    num_supps = 2 * k
                probs.append( prob_supp ** num_supps ) # must succeed for every S in H

            #Ns = range(0, num_supps * samples_per_supp_deterministic, num_supps) # deterministic N is samples_per_supp_deterministic * n_supps, and there are m supports
            Ns = np.arange(0, samples_per_supp_deterministic+1) / samples_per_supp_deterministic  # relative to deterministic sample complexity
            ax[i].plot(Ns, probs, color=str(np.linspace(0,1,m+1)[k]))
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        #ax[i].set_ylabel('$m$=%d' % m)
        ax[i].legend(["$m=%d$" % m], loc='upper left', handlelength=0, handletextpad=0)
        ax[i].set_xlim([0,1])
        ax[i].set_ylim([0,1])
        ax[i].set_yticks([0,1])
        ax[i].set_xticks(np.linspace(0,1,6))

    ax[-1].set_xlabel('Number of samples relative to the deterministic sample complexity')
    pp.savefig('./figures/prob_vs_samples.png')


