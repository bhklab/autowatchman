import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from damply import dirs

data = pd.read_csv(dirs.RESULTS / 'PMCC_AutoWATChmAN' / 'pred_results.csv')


metric = 'auc'
x = data['type']
y_mean = data[f'train_{metric}_score']
ci_lower = data[f'train_{metric}_ci_min']
ci_upper = data[f'train_{metric}_ci_max']

val_mean = data[f"val_{metric}_score"]
val_ci_lower = data[f"val_{metric}_ci_min"]
val_ci_upper = data[f"val_{metric}_ci_max"]

# y_err = [y_mean - ci_lower, ci_upper - y_mean]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:cyan", "tab:olive"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout='tight')

# plt.figure(figsize=(8,4), layout='tight')
for xi, yi, cil, ciu, c in zip(x, y_mean, ci_lower, ci_upper, colors):
    axes[0].errorbar(xi, yi, yerr=[[yi - cil], [ciu - yi]], fmt='o', color=c, ecolor=c, capsize=6, elinewidth=2, markersize=8)
    # plt.errorbar(xi, yi, yerr=[[yi - cil], [ciu - yi]], fmt='o', color=c, ecolor=c, capsize=6, elinewidth=2, markersize=8)
axes[0].set_ylim(0.25,1.05)
# rotate xlabels
axes[0].tick_params(axis='x', labelrotation=45)
axes[0].set_xlabel('Model Type')
axes[0].set_ylabel(f'Train {metric.upper()}')

# plt.figure(figsize=(8,4), layout='tight')
for xi, yi, cil, ciu, c in zip(x, val_mean, val_ci_lower, val_ci_upper, colors):
    axes[1].errorbar(xi, yi, yerr=[[yi - cil], [ciu - yi]], fmt='o', color=c, ecolor=c, capsize=6, elinewidth=2, markersize=8)

# plt.errorbar(x, y_mean, yerr=[y_mean - ci_lower, ci_upper - y_mean], fmt='o', color=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:cyan"], capsize=4)
axes[1].set_ylim(0.25,1.05)
# rotate xlabels
axes[1].tick_params(axis='x', labelrotation=45)
axes[1].set_xlabel('Model Type')
axes[1].set_ylabel(f'Validation {metric.upper()}')

plt.show()