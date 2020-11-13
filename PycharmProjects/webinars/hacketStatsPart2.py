import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#/colorbrewer2.org/
#pd.set_option('display.max_columns', None)
#https://github.com/justinbois/dc_stat_think for functions

df = pd.read_csv(
    "https://github.com/datacamp/Hacker-Stats-in-Python-Live-Training/blob/master/data/gfmt_sleep.csv?raw=True",
    na_values="*",
)
pd.set_option('display.max_columns', None)
df['insomnia'] = df.sci <= 16



#ecdf function
def ecdf(data):
    x = sorted(data)
    y = np.arange(1, len(x) + 1) / len(data)
    return x, y

def boostrap_replicate_1d(data, fun):
    bs_sample = np.random.choice(data, len(data))
    bs_replicate = fun(bs_sample)
    return bs_replicate

def draw_bs_reps(data, fun, size=10):
    bs_replicates = np.empty(size)
    for i in range(size):
        bs_replicates[i] = boostrap_replicate_1d(data, fun)

    return bs_replicates

def plugin_summary(data, func, ptiles=(2.5, 97.5), n_bs_reps=10000, label=None):
    """Compute and store ECDF, plug-in estimate, and confidence
       intervals in a dictionary."""
    # Initialize output dictionary
    summary = {}

    # Store data and settings
    summary['data'] = data
    summary['func'] = func
    summary['ptiles'] = ptiles
    summary['n_bs_reps'] = n_bs_reps
    summary['label'] = label

    # Compute ECDF x and y values
    summary['ecdf_x'], summary['ecdf_y'] = ecdf(data)

    # Compute plug-in estimate
    summary['estimate'] = func(data)

    # Compute bootstrap confidence interval
    summary['bs_reps'] = draw_bs_reps(data, func, size=n_bs_reps)
    summary['conf_int'] = np.percentile(summary['bs_reps'], ptiles)

    return summary


# Initialize list of plug-in summaries
summaries = []

# Iterate through groups and instantiate conf intervals
for label, group in df.groupby('insomnia'):
    summaries.append(
        plugin_summary(
            group['percent correct'].dropna().values,
            np.mean,
            label=label
        )
    )

# Adjust label names to be descriptive
for i, _ in enumerate(summaries):
    summaries[i]['label'] = (
        'insomniac' if summaries[i]['label'] else 'normal'
    )

'''   SAME AS LAST FUNC
for e in summaries:
    e['label'] = 'insomniac' if e['label'] else 'normal''' #SAME AS LAST FOR

def plot_conf_ints(summaries, palette=None):
    """Plot confidence intervals with estimates."""
    # Set a nice color palette
    if palette is None:
        palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
    elif type(palette) == str:
        palette = [palette]

    labels = [ci["label"] for ci in summaries][::-1]
    estimates = [ci["estimate"] for ci in summaries][::-1]
    conf_intervals = [ci["conf_int"] for ci in summaries][::-1]
    palette = palette[: len(labels)][::-1]

    # Set up axes for plot
    fig, ax = plt.subplots(figsize=(5, len(labels) / 2))

    # Plot estimates as dots and confidence intervals as lines
    for i, (label, est, conf_int, palette) in enumerate(
        zip(labels, estimates, conf_intervals, palette)
    ):
        #color = palette[i % len(palette)]
        ax.plot(
            [est],
            [label],
            marker=".",
            linestyle="none",
            markersize=10,
            color=palette,
        )

        ax.plot(conf_int, [label] * 2, linewidth=3, color=palette)

    # Make sure margins look ok
    ax.margins(y=0.25 if len(labels) < 3 else 0.125)

    return ax

'''# Dictionary connecting category to color
colors = {"normal": "#1f77b3", "insomniac": "#ff7f0e"}

# Plot the ECDFs
for s in summaries:
    _ = plt.plot(
        s['ecdf_x'],
        s['ecdf_y'],
        marker='.',
        linestyle='none',
        label=s['label'],
        color=colors[s['label']]
    )

_ = plt.xlabel('percent correct')
_ = plt.ylabel = 'ECDF'
_ = plt.legend()
plt.show()

plot_conf_ints(summaries)
plt.show()''' # Graphs for the first part solution from the first part not including formulas


#Information confidence when correct/in insomniacs/norm

conf_corr_insom = df.loc[df['insomnia'], 'confidence when correct'].dropna().values
conf_incorr_insom = df.loc[df['insomnia'], 'confidence when incorrect'].dropna().values
conf_corr_normal  = df.loc[~df['insomnia'], 'confidence when correct'].dropna().values
conf_incorr_normal = df.loc[~df['insomnia'], 'confidence when incorrect'].dropna().values


conf_corr_normal = plugin_summary(conf_corr_normal, np.mean, label='normal when correct')
conf_incorr_normal = plugin_summary(conf_incorr_normal, np.mean, label='normal when incorrect')
conf_corr_insom = plugin_summary(conf_corr_insom, np.mean, label='insom when correct')
conf_incorr_insom = plugin_summary(conf_incorr_insom, np.mean, label='insom when incorrect')

kwargs = {'marker': '.', 'linestyle': 'none'}
'''
plt.plot(conf_corr_normal['ecdf_x'], conf_corr_normal['ecdf_y'], c='r', **kwargs)
plt.plot(conf_incorr_normal['ecdf_x'], conf_incorr_normal['ecdf_y'], c='r',fillstyle='none', **kwargs)
plt.plot(conf_corr_insom['ecdf_x'], conf_corr_insom['ecdf_y'], c='b', **kwargs)
plt.plot(conf_incorr_insom['ecdf_x'], conf_incorr_insom['ecdf_y'], c='b',fillstyle='none', **kwargs)
plt.xlabel('confidence')
plt.ylabel('ecdf')
plt.show()


plot_conf_ints([conf_corr_normal, conf_incorr_normal,
                conf_corr_insom,conf_incorr_insom],
                palette =['#1f78b4', '#a6cee3', '#ff7f00', '#fdbf6f'])
#/colorbrewer2.org/ FOR COLORS
plt.xlabel('confidence')
plt.show()
''' #graphs part 1



'''
# part 2, Performing hypothesis significance test on 
# Confidence correct and confidence incorrect for normal and insomniac  

def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""
    # Concatenate the data sets
    data = np.concatenate((data1, data2))

    # Permute the concatenated array
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""
    return np.mean(data_1) - np.mean(data_2)

def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""
    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(
            data_1, data_2
        )

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates


test_stat = diff_of_means(
    conf_corr_normal['data'], conf_corr_insom['data']
)
print('t_s_c ', test_stat)
perm_reps = draw_perm_reps(
    conf_corr_normal['data'], conf_corr_insom['data'],
    diff_of_means, size=1000
)
p_value = np.sum(perm_reps <= test_stat) / len(perm_reps)
print('p1 ', p_value)



# Compute difference of means from original data
test_stat = diff_of_means(
    conf_incorr_normal['data'], conf_incorr_insom['data']
)
# Acquire permutation replicates
perm_reps = draw_perm_reps(
    conf_incorr_normal['data'], conf_incorr_insom['data'],
    diff_of_means, size=1000
)
# Compute p-value
p_value = np.sum(perm_reps <= test_stat) / len(perm_reps)
print('p2 ', p_value)'''  # part 2 NHST

'''
sns.pairplot(df[['sci', 'psqi', 'ess']],  diag_kind=None, height=2)
plt.show()

# Extract all three sleep/drowsiness metrics as NumPy arrays
sci = df['sci'].dropna().values
psqi = df['psqi'].dropna().values
ess = df['ess'].dropna().values

# Compute Pearson correlation between each pair
rho_sci_psqi = np.corrcoef(sci, psqi)[0, 1]
rho_sci_ess = np.corrcoef(sci, ess)[0, 1]
rho_psqi_ess = np.corrcoef(psqi, ess)[0, 1]
# Print the result
print('plug-in SCI-PSQI correlation:', rho_sci_psqi)
print('plug-in SCI-ESS correlation: ', rho_sci_ess)
print('plug-in PSQI-ESS correlation:', rho_psqi_ess)'''#Part 3. 1 scattered plot pair bootstrap  confidence interval


'''
def draw_pair_bs_reps(data1, data2, func, size=10):
    pair_bs_reps = np.empty(size)
    size1 = np.arange(len(data1))
    for i in range(size):

        inds = np.random.choice(size1, len(size1))
        data11 = data1[inds]
        data22 = data2[inds]

        pair_bs_reps[i] = func(data11, data22)

    return pair_bs_reps

def pearson(x, y):
    r = np.corrcoef(x,y)
    return r[0,1]

def ci_pearson(pair_bs_reps):
    ci_pearsons = np.percentile(pair_bs_reps, [2.5, 97.5])
    return ci_pearsons

bs_pair_rep2 = draw_pair_bs_reps(sci, psqi, pearson, 10000)
bs_pair_rep3 = draw_pair_bs_reps(sci, ess, pearson, 10000)
bs_pair_rep4 = draw_pair_bs_reps(psqi, ess, pearson, 10000)

print(ci_pearson(bs_pair_rep2))
print(ci_pearson(bs_pair_rep3))
print(ci_pearson(bs_pair_rep4))''' #My formula for pair boostrap and Pearson correlation

'''
def draw_bs_pairs(x, y, func, size=1):
    """Perform pairs bootstrap for single statistic."""
    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_replicates[i] = func(bs_x, bs_y)

    return bs_replicates

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix
    corr_mat = np.corrcoef(x, y)

    return corr_mat[0, 1]


# Pairs bootstrap replicates of correlation
sci_psqi_reps = draw_bs_pairs(sci, psqi, pearson_r, size=10000)
sci_ess_reps = draw_bs_pairs(sci, ess, pearson_r, size=10000)
psqi_ess_reps = draw_bs_pairs(psqi, ess, pearson_r, size=10000)

# Pairs bootstrap confidence intervals
sci_psqi_conf_int = np.percentile(sci_psqi_reps, [2.5, 97.5])
sci_ess_conf_int = np.percentile(sci_ess_reps, [2.5, 97.5])
psqi_ess_conf_int = np.percentile(psqi_ess_reps, [2.5, 97.5])

print(sci_psqi_conf_int, sci_ess_conf_int, psqi_ess_conf_int)
#Creates dictionaries to be able to use a confidence interval plot function
sci_psqi = dict(label="SCI–PSQI", estimate=rho_sci_psqi, conf_int=sci_psqi_conf_int)
sci_ess = dict(label="SCI–ESS", estimate=rho_sci_ess, conf_int=sci_ess_conf_int)
psqi_ess = dict(label="PSQI–ESS", estimate=rho_psqi_ess, conf_int=psqi_ess_conf_int)

plot_conf_ints([sci_psqi, sci_ess, psqi_ess],
               palette=['#e41a1c','#377eb8','#4daf4a'])
plt.show()''' #Part 3. 2 scattered plot pair bootstrap  confidence interval
'''
plt.plot(df["sci"], df["percent correct"], marker=".",linestyle="none", alpha=0.5)
plt.xlabel("SCI")
plt.ylabel("percent correct")
plt.show()
'''
sci = df['sci'].dropna().values
pcorr = df['percent correct'].dropna().values

#slope, intercept = np.polyfit(sci, pcorr, 1)


def draw_bs_pairs_linreg(x, y, size=1):

    ind = np.arange(len(x))
    bs_slopes = np.empty(size)
    bs_interc = np.empty(size)
    for i in range(size):
        per_ind = np.random.choice(ind, len(ind))
        x_perm , y_perm = x[per_ind], y[per_ind]
        bs_slopes[i], bs_interc[i] = np.polyfit(x_perm, y_perm, 1)
    return bs_slopes, bs_interc

def calc_percen(data):
    return np.percentile(data, [2.5, 97.5])




bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(sci, pcorr, size=10000)
conf_int_slope = calc_percen(bs_slope_reps)
conf_int_interc = calc_percen(bs_intercept_reps)

def plot_bs_pairs_linreg_reps(slopes, intercepts, x_values):
    fig, ax = plt.subplots()
    x = np.array([0, max(x_values)])
    counter = 0
    for i, (slope, intercept) in enumerate(zip(slopes, intercepts)):
        if i % 33 == 0:

            ax.plot(x, intercept + slope * x, linewidth=0.5, alpha=0.2, c='tomato')
            counter += 1
    print(counter)
    return ax
plot_bs_pairs_linreg_reps(bs_slope_reps, bs_intercept_reps, sci)
plt.plot(sci, pcorr, marker=".",linestyle="none", alpha=0.5)
plt.xlabel("SCI")
plt.ylabel("percent correct")
plt.show()


def draw_perm_reps_slope(x, y, size=1):
    slp_reps = np.empty(size)

    for i in range(size):
        x_perm = np.random.permutation(x)
        slp_rep, _ = np.polyfit(x_perm, y, 1)
        slp_reps[i] = slp_rep
    return slp_reps

test_stat_s, _ = np.polyfit(sci, pcorr, 1)

nhst_slopes = draw_perm_reps_slope(sci, pcorr, 10000)

p_slope = sum(nhst_slopes >= test_stat_s) / 10000
print('p_slope ', p_slope)



