"""Figure 2: the moderation slope beta1 (Delta_CLS ~ F4) flips from wrong-signed-null
(collapsed-policy experiment #18) to the H1-predicted POSITIVE direction once a present,
scene-adaptive multimodal treatment (RL) is supplied. Error bars = +/- cluster SE."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# committed results (ADR-033/034 for #18; ADR-043/044 for the RL re-test)
labels = ["#18 contrast\n(collapsed, r0)", "#18 contrast\n(collapsed, r1)",
          "RL re-test\nN=200", "RL re-test\nN=600", "RL re-test\nN=800"]
beta = [-0.0064, -0.0013, 0.054, 0.045, 0.035]
se   = [0.0062, 0.0068, 0.049, 0.028, 0.023]
pone = [None, None, 0.13, 0.055, 0.063]      # one-sided p for H1 (RL only)
colors = ["#c0392b", "#c0392b", "#27ae60", "#27ae60", "#27ae60"]

fig, ax = plt.subplots(figsize=(9, 5.2))
x = range(len(labels))
ax.axhspan(0, 0.09, color="#27ae60", alpha=0.06)
ax.axhline(0, color="k", lw=1)
ax.errorbar(x, beta, yerr=se, fmt="o", ms=9, capsize=5, lw=2,
            ecolor="gray", mfc="white", mec="k", zorder=3)
for xi, (b, c) in enumerate(zip(beta, colors)):
    ax.plot(xi, b, "o", ms=9, color=c, zorder=4)
for xi, p in enumerate(pone):
    if p is not None:
        ax.annotate(f"1-sided p={p}", (xi, beta[xi] + se[xi] + 0.004), ha="center", fontsize=8)
ax.text(0.5, -0.022, "collapsed policy:\nslope null & WRONG-SIGNED", ha="center", color="#c0392b", fontsize=9)
ax.text(3.0, 0.082, "present scene-adaptive treatment:\nslope POSITIVE (H1 direction)", ha="center",
        color="#27ae60", fontsize=9)
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel(r"moderation slope $\beta_1$  (Δ$_{CLS}$ ~ F4)")
ax.set_title("Making multimodality PRESENT flips the interaction-criticality moderation positive\n"
             "(diagnostic null was treatment-absence, not a true negative)", fontsize=11, fontweight="bold")
ax.set_ylim(-0.04, 0.11)
plt.tight_layout()
out = "/scratch/patodia.pa/av-policy-lab/slope_flip_figure.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)
