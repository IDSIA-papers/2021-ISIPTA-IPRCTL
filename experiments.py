import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.patches as mpatches
from tikzplotlib import save as tikz_save
from iprctl import Chain, contaminate

##############################################################
n = 6
pmc = Chain("Messages", n, ["start", "delivery", "try", "lost"])
transitions = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 0.9, 0, 0.1], [0, 0, 1, 0]])
pmc.set_t(transitions)

# Example 1 (Table 1)
print(":: Table 1: Hitting Probs, index is t-1, each col is a different starting state")
p = pmc.hitting([0, 0, 0, 1], verbose=True)

# Example 2 (Figure 2, left)
print(":: Plotting Figure 2 (top) as hitting1.png (x index is t-1)")
upper = {}
lower = {}
alpha_max = 0.05
alpha_min = 0.30
levels_max = 0.03
levels_min = 0.01
n_steps = 3
alphas = np.linspace(alpha_max, alpha_min, n_steps)
levels = np.linspace(levels_max, levels_min, n_steps)
imc = Chain("IMessages", n, ["start", "delivery", "try", "lost"])

for k, level in enumerate(levels):
	imc.set_t(contaminate(transitions, level))
	upper[level] = imc.hitting([0, 0, 0, 1])
	lower[level] = imc.hitting([0, 0, 0, 1], 'min')

k = 0
fig = plt.figure()
plt.plot(p.index, p[pmc.states[k]], color='black', label=pmc.states[k])
plt.axhline(y=0.25, color='k', linestyle='dashed')
for level, a in zip(levels, alphas):
	plt.fill_between(lower[level].index, lower[level][pmc.states[k]], upper[level][pmc.states[k]], color='black', alpha=a)
plt.xlabel("$t$")
plt.ylabel("$h_{\{\mathrm{lost}\}}^{\leq t}(\mathrm{start})$")
leg = []
leg.append(mpatches.Patch(color='black', label='$\epsilon = 0.00$'))
leg.append(mpatches.Patch(color='0.4', label='$\epsilon = 0.01$'))
leg.append(mpatches.Patch(color='0.6', label='$\epsilon = 0.02$'))
leg.append(mpatches.Patch(color='0.8', label='$\epsilon = 0.03$'))
plt.legend(handles=leg)
plt.savefig("hitting1.png", dpi=250, bbox_inches="tight",pad_inches=0.02)
tikz_save('hitting1.tex')


##############################################################


# Example 2 (Figure 2, right)
print(":: Plotting Figure 2 (bot) as hitting2.png (x index is t-1)")

n = 100
pmc = Chain("Messages", n, ["start", "delivery", "try", "lost"])
pmc.set_t(transitions)
p = pmc.hitting([0, 0, 0, 1], verbose=False)
upper = {}
lower = {}
alpha_max = 0.05
alpha_min = 0.30
levels_max = 0.03
levels_min = 0.01
n_steps = 3
alphas = np.linspace(alpha_max, alpha_min, n_steps)
levels = np.linspace(levels_max, levels_min, n_steps)
imc = Chain("IMessages", n, ["start", "delivery", "try", "lost"])

for k, level in enumerate(levels):
    imc.set_t(contaminate(transitions, level))
    upper[level] = imc.hitting([0, 0, 0, 1])
    lower[level] = imc.hitting([0, 0, 0, 1], 'min')

k = 0
fig = plt.figure()
plt.plot(p.index, p[pmc.states[k]], color='black', label=pmc.states[k])
plt.axhline(y=0.25, color='k', linestyle='dashed')
for level, a in zip(levels, alphas):
    plt.fill_between(lower[level].index, lower[level][pmc.states[k]], upper[level][pmc.states[k]], color='black', alpha=a)
plt.xlabel("$t$")
plt.ylabel("$h_{\{\mathrm{lost}\}}^{\leq t}(\mathrm{start})$")
plt.yticks(np.arange(0, 1.01, step=0.25))
leg = []
leg.append(mpatches.Patch(color='black', label='$\epsilon = 0.00$'))
leg.append(mpatches.Patch(color='0.4', label='$\epsilon = 0.01$'))
leg.append(mpatches.Patch(color='0.6', label='$\epsilon = 0.02$'))
leg.append(mpatches.Patch(color='0.8', label='$\epsilon = 0.03$'))
plt.legend(handles=leg)
plt.axhline(y=1, color='k', linestyle='dashed')
plt.savefig("hitting2.png", dpi=250, bbox_inches="tight",pad_inches=0.02)
tikz_save('hitting2.tex')

# Application (Section 6)
print(":: Plotting Figure 4 as cumul.png")
gamma = [0.0175, 0.0354, 0.0281]
nu = [0.00031, 0.00187, 0.00149]
delta = [0.0012, 0.0013, 0.0018]
a = [99, 57, 94]
l = [25, 82, 78]
dep = 3
n = 365*3 #20

geriatric_mc = Chain("Geriatric", n, ['acute','long','dismissed'])
geriatric_mc.set_rew([100, 50, 0])
cumul = {}
total = [0, 0, 0]

for dep in range(3):
    tr = np.array([[1-gamma[dep]-nu[dep], nu[dep], gamma[dep]], [0, 1-delta[dep], delta[dep]], [0, 0, 1]])
    geriatric_mc.set_t(tr)
    cumul[dep] = geriatric_mc.cumulative([0,0,1])
    total[dep] = a[dep]*cumul[dep]['acute'][n]+l[dep]*cumul[dep]['long'][n]
    print('Tot = %2.5f' % total[dep])

geriatric_imc = Chain("IGeriatric", n, ['acute','long','dismissed'])
geriatric_imc.set_rew([100, 50, 0])
itr = np.array([[[1-max(gamma)-max(nu), min(nu), min(gamma)], [1-min(gamma)-min(nu), max(nu), max(gamma)]], [[0, 1-max(delta), min(delta)], [0, 1-min(delta), max(delta)]], [[0, 0, 1],[0, 0, 1]]])
geriatric_imc.set_t(itr)
cumul['lower'] = geriatric_imc.cumulative([0,0,1], opt='min')
cumul['upper'] = geriatric_imc.cumulative([0,0,1], opt='max')
totall = sum(a) * cumul['lower']['acute'][n] + sum(l) * cumul['lower']['long'][n]
totalu = sum(a) * cumul['upper']['acute'][n] + sum(l) * cumul['upper']['long'][n]

totals_precise = []
totals_lower = []
totals_upper = []

for _ in range(n+1):
    t = 0
    for dep in range(3):
        t += a[dep] * cumul[dep]['acute'][_] + l[dep] * cumul[dep]['long'][_]
    if _ % 30 == 0:
        totals_precise.append(t)
        totals_lower.append(sum(a) * cumul['lower']['acute'][_] + sum(l) * cumul['lower']['long'][_])
        totals_upper.append(sum(a) * cumul['upper']['acute'][_] + sum(l) * cumul['upper']['long'][_])

fig = plt.figure()
plt.plot(range(len(totals_precise)), totals_precise, color='black', label='ciao')
plt.fill_between(range(len(totals_precise)), totals_lower, totals_upper, color='black', alpha=.2)
plt.xlabel("$t$")
plt.ylabel("$cost$")
# plt.yticks(np.arange(0, 1.01, step=0.25))
plt.savefig("cumul.png", dpi=250, bbox_inches="tight",pad_inches=0.02)
tikz_save('cumul.tex')

# Last experiment

gamma = [0.0175, 0.0354, 0.0281]
nu = [0.00031, 0.00187, 0.00149]
delta = [0.0012, 0.0013, 0.0018]
deps = 3
n = 365

geriatric_mc = Chain("Geriatric", n, ['acute','long','dismissed'])
geriatric_imc = Chain("IGeriatric", n, ['acute','long','dismissed'])
geriatric_mc.set_rew([100, 50, 0])
geriatric_imc.set_rew([100, 50, 0])
itr = np.array([[[1-max(gamma)-max(nu), min(nu), min(gamma)], [1-min(gamma)-min(nu), max(nu), max(gamma)]], [[0, 1-max(delta), min(delta)], [0, 1-min(delta), max(delta)]], [[0, 0, 1],[0, 0, 1]]])
geriatric_imc.set_t(itr)


print(":: Computing Data in Table 3")
cumul = {}
for dep in range(3):
    tr = np.array([[1-gamma[dep]-nu[dep], nu[dep], gamma[dep]], [0, 1-delta[dep], delta[dep]], [0, 0, 1]])
    geriatric_mc.set_t(tr)
    cumul[dep] = geriatric_mc.cumulative([0,0,1])

cumul['lower'] = geriatric_imc.cumulative([0,0,1], opt='min')
cumul['upper'] = geriatric_imc.cumulative([0,0,1], opt='max')

print(cumul[0]['acute'][365])
print(cumul[1]['acute'][365])
print(cumul[2]['acute'][365])
print(cumul['lower']['acute'][365])
print(cumul['upper']['acute'][365])
print('--')
print(cumul[0]['long'][365])
print(cumul[1]['long'][365])
print(cumul[2]['long'][365])
print(cumul['lower']['long'][365])
print(cumul['upper']['long'][365])
