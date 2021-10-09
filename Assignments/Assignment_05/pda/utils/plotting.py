from matplotlib import pyplot as plt
from scipy.stats import chi2
import numpy as np


def plot_NEES(NEES, dof=2, confidence=0.95):
    fig, ax = plt.subplots(1)
    ax.plot(NEES, label="NEES")
    conf_lower = chi2.ppf((1-confidence)/2, df=2)
    conf_upper = chi2.ppf(1 - (1-confidence)/2, df=2)
    n_total = len(NEES)
    n_below = len([None for value in NEES if value < conf_lower])
    n_above = len([None for value in NEES if value > conf_upper])
    frac_inside = (n_total - n_below - n_above)/n_total
    frac_below = n_below/n_total
    frac_above = n_above/n_total

    ax.hlines([conf_lower, conf_upper], 0, len(NEES), "r", ":",
              label=f"{confidence:2.1%} confidence interval")
    ax.legend()
    ax.set_title(f"NEES\n {frac_inside:2.1%} "
                 f"inside {confidence:2.1%} confidence interval "
                 f"({frac_below:2.1%} below, {frac_above:2.1%} above)")

    ax.set_yscale('log')
    ax.set_ylabel('NEES')
    ax.set_xlabel('K')
    fig.tight_layout()
