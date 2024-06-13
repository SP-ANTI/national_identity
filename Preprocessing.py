import matplotlib.pyplot as plt
import pandas as pd
import pyreadstat
import seaborn as sns
import scipy

names = ['CASEID', 'YEAR_SDNO', 'COUNTRY',
         'v5', 'v7', 'v8', 'v9', 'v15', 'v16', 'v18', 'v20', 'v21', 'v22', 'v23',
         'v25', 'v26', 'v27', 'v28', 'v29', 'v32', 'v42', 'v43', 'v44', 'v45', 'v48',
         'TOPBOT', 'DEGREE', 'AGE', 'SEX', 'MAINSTAT']

df = pd.read_spss('ZA5960_v1-0-0.sav', convert_categoricals=False)[names]
df['COUNTRY'] = pd.read_spss('ZA5960_v1-0-0.sav', convert_categoricals=True)['COUNTRY'].astype(str)
print(df['COUNTRY'])
df_sup = pd.read_spss('ZA5961_v1-0-0.sav', convert_categoricals=False)[['CASEID', 'CLASS95', 'YEAR_SDNO']]
df.info()
df_sup.info()
df_sup = df_sup[df_sup['YEAR_SDNO'] == 1995].drop('YEAR_SDNO', axis=1)
df_sup['CLASS95'] = (df_sup['CLASS95'] + 1) // 2
print(df_sup['CLASS95'].unique())
df['TOPBOT'] = df['TOPBOT'].apply(lambda x: None if pd.isna(x) else 1 if x < 4 else 2 if x < 8 else 3)
df['MAINSTAT'] = df['MAINSTAT'].apply(
    lambda x: None if pd.isna(x) else 4 if (4 < x < 8) else 3 if x == 1 else 1 if x == 2 else 2)

print(df['TOPBOT'].unique())
df = df.merge(df_sup, on='CASEID', how='left')
print(df)
df.info()
df_sup.info()
df.loc[df['YEAR_SDNO'] == 1995, 'TOPBOT'] = df.loc[df['YEAR_SDNO'] == 1995, 'CLASS95']
df.drop('CLASS95', axis=1, inplace=True)
print(df.head())
df.info()

ex = df[['TOPBOT', 'MAINSTAT', 'DEGREE']].copy()
df['ECO'] = 2
df.loc[(df['TOPBOT'] == 1) | (df['TOPBOT'] == 2) & (
        df['MAINSTAT'].isin([2, 3]) & (df['DEGREE'] < 3) | (df['MAINSTAT'] == 1)), 'ECO'] = 1
df.loc[(df['TOPBOT'] == 3) | (df['TOPBOT'] == 2) & df['MAINSTAT'].isin([2, 3]) & (df['DEGREE'] >= 5), 'ECO'] = 3
ex['ECO'] = df['ECO'].copy()
print(ex.head(20))


def recode(x):
    l = max(x.unique())
    return abs(x - l) + 1


print(df['v16'].head(10))
recode_list = ['v5', 'v7', 'v8', 'v9', 'v15', 'v16', 'v18', 'v20', 'v21', 'v22', 'v23',
               'v25', 'v26', 'v27', 'v28', 'v29', 'v32', 'v42', 'v44']
for name in recode_list:
    df[name] = recode(df[name])
print(df['v16'].head(10))
df.dropna(inplace=True)
df.to_csv('dataset.csv', index=False)

anti = ['v48', 'v42', 'v43', 'v44', 'v45']
blind = ['v15', 'v32', 'v16']
ethn = ['v5', 'v7', 'v8', 'v9']
cult = ['v18', 'v25', 'v26', 'v28']
polit = ['v20', 'v21', 'v22', 'v23', 'v29']
years = df['YEAR_SDNO'].unique()

show = False
if show:
    for p in [anti, blind, ethn, cult, polit]:
        for y in years:
            sns.heatmap(df.loc[df['YEAR_SDNO'] == y, p].corr(method='spearman'), annot=True, cmap='cividis')
            plt.show()
        print(df.groupby('YEAR_SDNO')[p].mean())
        print(df.groupby('YEAR_SDNO')[p].std())
else:
    print(df.groupby('YEAR_SDNO')[anti].mean())
    print(df.groupby('YEAR_SDNO')[anti].std())

table = df.groupby('YEAR_SDNO')[anti].mean()
table.loc[2013.0] = table.loc[2013.0] - table.loc[1995.0]
table.loc[2003.0] = table.loc[2003.0] - table.loc[1995.0]

for p in anti:
    print(p)
    r2003 = 'less' if table.loc[2003, p] < 0 else 'greater'
    r2013 = 'less' if table.loc[2013, p] < 0 else 'greater'
    print(2003, scipy.stats.ttest_ind(df.loc[df['YEAR_SDNO'] == 2003, p], df.loc[df['YEAR_SDNO'] == 1995, p], alternative=r2003)[1])
    print(2013, scipy.stats.ttest_ind(df.loc[df['YEAR_SDNO'] == 2013, p], df.loc[df['YEAR_SDNO'] == 1995, p], alternative=r2013)[1])

print('Изменение среднего')
print(table)
