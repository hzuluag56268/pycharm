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

#print(df.info())

ess = df['ess'].dropna().values
age = df['age'].dropna().values
pcor = df['percent correct'].dropna().values
psqi = df['psqi'].dropna().values
sci =  df['sci'].dropna().values

def ci_and_slope_pearson(x, y, labelx, labely, size=10000):
    bs_slopes = np.empty(size)
    inds = np.arange(len(x))

    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_slopes[i], _ = np.polyfit(x[bs_inds], y[bs_inds], 1)

    ci = 'ci: {}'.format(np.percentile(bs_slopes, [2.5, 97.5]))
    slope = 'slope: {}'.format(np.polyfit(x, y, 1)[0])

    return  'x= {}, y= {}'.format(labelx, labely) + '\n'  +slope + '\n' + ci + '\n'

for x, labelx, labely in zip([sci, ess, psqi], ['sci', 'ess', 'psqi'], ['pcor'] * 3):

    ci_slp = ci_and_slope_pearson( x, pcor, labelx, labely)
    print(ci_slp)
'''
for x, label in zip([sci, ess, psqi], ['sci', 'ess', 'psqi']):
    fig, ax = plt.subplots(figsize=(4,5))
    ax = sns.regplot(x, pcor)
    ax.set_xlabel(label)
    ax.set_ylabel('pcor')

plt.show()'''


ci_age = ci_and_slope_pearson(age, pcor, 'pcor', 'age')
print(ci_age)
sns.lmplot('percent correct', 'age', df, hue='insomnia')
plt.show()