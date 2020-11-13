import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


'''x = np.linspace(125, 205, 200)
cdf = scipy.stats.norm.cdf(x, 165, 9)
print(len(x))
plt.plot(x, cdf)
plt.margins(x=0, y=.015)
#plt.show()

np.random.seed(3252)
heights = np.random.normal(165, 9, 100)
#print(heights)

def ecdf(data):
    x = sorted(data)
    y = np.arange(1, len(x) + 1) / len(data)
    return x, y

hieght, perc = ecdf(heights)
plt.plot(x, cdf)
plt.plot(hieght, perc, marker='.', linestyle='none')

plt.show()'''   #ecdf function
def ecdf(data):
    x = sorted(data)
    y = np.arange(1, len(x) + 1) / len(data)
    return x, y

df = pd.read_csv(
    "https://github.com/datacamp/Hacker-Stats-in-Python-Live-Training/blob/master/data/gfmt_sleep.csv?raw=True",
    na_values="*",
)
pd.set_option('display.max_columns', None)
df['insomnia'] = df.sci <= 16
print(df.info())

pcorr_normal = df.loc[~df['insomnia'], 'percent correct'].values
pcorr_insom = df.loc[df['insomnia'], 'percent correct'].values

x_normal, y_normal = ecdf(pcorr_normal)
x_insom, y_insom = ecdf(pcorr_insom)

plt.plot(x_normal, y_normal, marker='.', linestyle='none', label='norm', color='blue' )
plt.plot(x_insom, y_insom,  marker='.', linestyle='none', label='ins', color='red')
plt.margins(x=0)
plt.legend()
plt.show()

'''sns.catplot(x='insomnia', y='percent correct', data=df, kind='violin')
plt.show()
''' #catplot

pcorr_normal_mean = np.mean(pcorr_normal)
pcorr_insom_mean = np.mean(pcorr_insom)
print('mean nor', pcorr_normal_mean)
print('mean inso', pcorr_insom_mean)


def boostrap_replicate_1d(data, fun):
    bs_sample = np.random.choice(data, len(data))
    bs_replicate = fun(bs_sample)
    return bs_replicate

def draw_bs_reps(data, fun, size=10):
    bs_replicates = np.empty(size)
    for i in range(size):
        bs_replicates[i] = boostrap_replicate_1d(data, fun)

    return bs_replicates
bs_reps_normal = draw_bs_reps(pcorr_normal, np.mean, 10000)
bs_reps_insom =  draw_bs_reps(pcorr_insom, np.mean, 10000)

ci_nor = np.percentile(bs_reps_normal, [2.5, 97.5])
ci_ins = np.percentile(bs_reps_insom, [2.5, 97.5])
print(ci_nor, ci_ins)


def plot_conf_ints(categories, estimates, conf_ints, palette=None):
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
    palette = palette[: len(categories)][::-1]

    # Set up axes for plot
    fig, ax = plt.subplots(figsize=(5, len(categories) / 2))

    # Plot estimates as dots and confidence intervals as lines
    for i, (cat, est, conf_int) in enumerate(
        zip(categories[::-1], estimates[::-1], conf_ints[::-1])
    ):
        color = palette[i % len(palette)]
        ax.plot(
            [est],
            [cat],
            marker=".",
            linestyle="none",
            markersize=10,
            color=color,
        )

        ax.plot(conf_int, [cat] * 2, linewidth=3, color=color)

    # Make sure margins look ok
    ax.margins(y=0.25 if len(categories) < 3 else 0.125)

    return ax
'''ax = plot_conf_ints(
    ["normal sleepers", "insomniacs"],
    [pcorr_normal_mean, pcorr_insom_mean],
    [ci_nor, ci_ins],
)
_ = ax.set_xlabel("percent correct")
plt.show()''' # plot ci

def plot_ci(categories, cis, means):
    for i, (category, ci, mean) in enumerate(zip(categories, cis, means), 1):
        plt.plot(ci, [i * 1/3, i * 1/3], linewidth=3, label=categories[i-1])
        plt.plot(mean, i  * 1/3, marker='d', c='g')
    mini = round(min(min(cis[0]), min(cis[1])))  - 1
    maxi = round(max(max(cis[0]), max(cis[1])))  + 1
    plt.legend()
    '''#plt.xlim(mini, maxi)
    #plt.xticks([*np.arange(mini, maxi+1)])
    #plt.ylim([0, len(means) + 1])
    #plt.yticks(np.arange(1, len(categories) + 1) * 1/3, categories)
    # '''#plot customization
    plt.show()
plot_ci(['normal', 'insomniac'], [ci_nor, ci_ins],
       [pcorr_normal_mean, pcorr_insom_mean])


diff_mean = bs_reps_normal - bs_reps_insom
sns.distplot(diff_mean)
plt.show()

diff_mean_x, diff_mean_y = ecdf(diff_mean)
plt.plot(diff_mean_x, diff_mean_y,
         marker='.', linestyle='none', color='red')
plt.show()
print('mean diff of means:', np.mean(diff_mean))
print('mean diff of mean:', pcorr_normal_mean - pcorr_insom_mean)

sns.catplot('percent correct', 'insomnia', hue='gender', data=df ,kind='point', join=False, orient='h')
plt.show()
#pointplots do the job too


# practice bs by gender normal
'''m_pcorr_normal = df.loc[(df['insomnia']) & (df['gender'] == 'm'), 'percent correct'].dropna().values
f_pcorr_normal = df.loc[(df['insomnia']) & (df['gender'] == 'f'), 'percent correct'].dropna().values
print('h:', len(m_pcorr_normal))
print('f: ', len(f_pcorr_normal))

x_m_pcorr_normal, y_m_pcorr_normal = ecdf(m_pcorr_normal)
x_f_pcorr_normal, y_f_pcorr_normal = ecdf(f_pcorr_normal)

plt.plot(x_m_pcorr_normal, y_m_pcorr_normal, marker='.', linestyle='none', label='hombre', color='blue' )
plt.plot(x_f_pcorr_normal, y_f_pcorr_normal,  marker='.', linestyle='none', label='mujer', color='red')
plt.margins(x=0)
plt.legend()
plt.show()


bs_m_pcorr_normal = draw_bs_reps(m_pcorr_normal, np.mean, 10000)
ci_bs_m_pcorr_normal = np.percentile(bs_m_pcorr_normal, [2.5, 97.5])

bs_f_pcorr_normal = draw_bs_reps(f_pcorr_normal, np.mean, 10000)
ci_bs_f_pcorr_normal = np.percentile(bs_f_pcorr_normal, [2.5, 97.5])

diff_mean_f_m_pcorr_normal = bs_m_pcorr_normal - bs_f_pcorr_normal
ci_diff_mean_f_m_pcorr_normal = np.percentile(diff_mean_f_m_pcorr_normal, [2.5, 97.5])
print(ci_diff_mean_f_m_pcorr_normal)
p = sum(diff_mean_f_m_pcorr_normal < 0)  / 10000
print('p:', p)
plot_ci(['m', 'f'], [ci_bs_m_pcorr_normal, ci_bs_f_pcorr_normal],
        [np.mean(bs_m_pcorr_normal), np.mean(bs_f_pcorr_normal)])
plt.show()

print(df.gender.value_counts())'''