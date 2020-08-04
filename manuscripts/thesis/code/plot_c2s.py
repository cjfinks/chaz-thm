import matplotlib.pyplot as pp
import numpy as np

s = 1
ks = range(5,10+1,s)

#fig, ax = pp.subplots(len(ks), 1, figsize=(20,10), sharex=True)
#if len(ks) == 1:
    #ax = [ax]
#fig.add_subplot(111, frameon=False)
#pp.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#pp.ylabel("Pr[$C_2(\mathbf{A}, \mathcal{H}) = x]$")
#pp.xlabel("x")

fig, ax = pp.subplots(1,1)#,figsize=(10,20))
ax = [ax]
pp.ylabel("Pr[$C_2(\mathbf{A}, \mathcal{H}) = x]$")
pp.xlabel("x")


# set bounds
c2s = np.load("c2s_k=%d_2x.npy" % ks[0])
p = 0.3#0.95
c2max = c2s[int(p*len(c2s))]

for i, k in enumerate(ks[::-1]):
    m = k**2
    #c2s = np.load("c2s_k=%d.npy" % k)
    c2s = np.load("c2s_k=%d_2x.npy" % k)
    bins = np.linspace(0, c2max, 151)
    hist, bins = np.histogram(c2s[np.logical_and(c2s>0, c2s<c2max)], bins)
    hist = hist / hist.sum()
    j=0
    ax[j].bar(bins[:-1], hist, width=bins[1]-bins[0], color=str(np.linspace(0,1,(ks[-1]-ks[0])//s+2)[i]))
    #ax[j].legend(["$m=%d, k=%d$" % (m,k)], loc='upper left', handlelength=0, handletextpad=0)
    ax[j].spines['right'].set_visible(False)
    ax[j].spines['top'].set_visible(False)
    ax[j].set_ylim([0, 0.3])
    ax[j].set_xlim([0,c2max])
    ax[j].set_yticks([0, 0.1, 0.2, 0.3])
    # TODO: do relative histogram so as to share y axis

ax[j].legend(["$m=%d^2$" % k for k in ks[::-1]], loc='upper right')
pp.savefig('C2_2x.pdf')
pp.show()