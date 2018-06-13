from __future__ import absolute_import,division,print_function
import matplotlib as  mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df = pd.read_csv('redcard.csv')
print(df.shape)

print(df.describe())

all_columns = df.columns.tolist()
all_columns


np.mean(df.groupby('playerShort').height.mean())

player_index = 'PlayerShort'
player_cols = ['birthday','height','weight','position','photoID','rater1','rater2']

all_cols_unique_players = df.groupby('playerShort').agg({col:'nunique' for col in player_cols})

all_cols_unique_players[all_cols_unique_players > 1 ].dropna()

def get_subgroup(dataframe,g_index,g_columns):
    g = dataframe.groupby(g_index).agg({col:'nunique' for col in g_columns})
    if g[g > 1].dropna().shape[0] != 0:
        print('it doesn\'t have all unique values ')
    return dataframe.group(g_index).agg({col:'max' for col in g_columns})

def save_subgroup(dataframe,g_index,subgroup_name,prefix = 'raw_'):
    save_subgroup_filename = ''.join([prefix,subgroup_name,'.csv.gz'])]
    dataframe.to_csv(save_subgroup_filename,compression = 'gzip',encoding = 'UTF-8')
    test_df  = pd.read_csv(save_subgroup_filename,compression='gzip',index_col = g_index,encoding='UTF-8')
    if dataframe.equals(test_df):
        print('Test-passed:we recover the equivalent subgroup dataframe')
    else:
        print('Warning--equivalence test!!! Double check')




club_index = 'club'
club_cols = ['leagueCountry']
clubs = get_subgroup(df,club_index,club_cols)

clubs['leagueCountry'].value_counts()

save_subgroup(clubs,club_index,'clubs')



referee_index = 'refNum'
referee_cols = ['refCountry']
referees = get_subgroup(df,referee_index,referee_cols)

referees.refCountry.nunique()

country_index = 'refCountry'
country_cols = ['Alpha_3','meanIAT','nIAT','meanExp','nExp','seExp']
countries = get_subgroup(df,country_index,country_cols)

rename_columns = {'Alpha_3':'countryName'}
countries = countries.rename(columns = rename_columns)




dyad_index = ['refNum','playerShort']
dyad_cols = ['games','victories','ties','defeats','goals','yellowCards','yellowReds','redCards']
dyad = get_subgroup(df,g_index = dyad_index,g_columns=dyad_cols)


def load_subgroup(filename,index_col = [0]):
    return pd.read_csv(filename,compression = 'gzip',index_col=index_col)

import missingno as msno
msno.matrix(player.sample(500),figsize=(16,7),width_ratios=(15,1))


fig,ax = plt.subplot(figsize = (12,8))
sns.heatmap(pd.crosstab(players.rater1,players.rater2),cmap='Blues',annot=True,fmt='d',ax = ax)
ax.set_title('Correlation between Rater 1 and Rater 2\n')
fig.tight_layout()


##########################################################
players = load_subgroup('raw_players.csv.gz')
players.shape

msno.matrix(players.sample(500),figsize=(16,7),width_ratio = (15,1))

msno.heatmap(players.sample(500),figsize=(16,7))

print('all players:',len(players))
print('rater1 nulls:',len(players[players.rater1.isnull()]))
print('rater2 nulls:',len(players[players.rater2.isnull()]))
print('both nulls:',len(players[(players.rater1.isnull()) & (players.rater2.isnull())]))

pd.crosstab(players.rater1,players.rater2)

fig,ax = plt.subplots(figsize=(12,8))
sns.heatmap(pd.crosstab(players.rater1,players.rater2),cmap='Blues',annot=True,fmt='d',ax = ax)
ax.set_title('correlation between rater1 and rater2')
fig.tight_layout()

players['skintone'] = players[['rater1','rater2']].mean(axis = 1
                                                        )
sns.distplot(players.skintone,kde=False
             )


fig,ax = plt.subplots(figsize = (12,8))
players['positon'].value_counts(dropna=False).plot(kind = 'bar',ax = ax)
ax.set_xlabel('positons')
ax.set_ylabel('counts')
fig.tight_layout()



##############
#创建新特征
position_types = players.position.unique()

defense = ['Center Back','Attacking Midfielder','Right Fullback']
midfield = ['Right Midfielder','Center Midfielder','Left Midfielder']
forward = ['Attacking Midfielder','Left Winger','Right Winger','Center Forward']
keeper = 'Goalkeeper'

players.loc[players['position'].isin(defense),'position_agg'] = 'Defense'
players.loc[players['position'].isin(midfield),'position_agg'] = 'Midfield'
players.loc[players['position'].isin(forward),'position_agg'] = 'Forward'
players.loc[players['position'].eq(keeper),'position_agg'] = 'keeper'

fig,ax = plt.subplots(figsize = (12,8))
players['position_agg'].value_counts(dropna = False,ascending = True).plot(kind = 'barh',ax = ax)
ax.set_ylabel('position_agg')
ax.set_xlabel('counts')
fig.tight_layout()

#######################################
#找出特征之间联系
fig,ax = plt.subplots(figsize = (10,10))
pd.scatter_matrix(players[['height','weight','skintone']],alpha = 0.2,diagonal = 'hist',ax = ax)



fig,ax = plt.subplots(figsize = (10,10))
sns.regplot('weight','height',data = players,ax = ax)
ax.set_ylabel('height/cm')
ax.set_xlabel('weight/kg')
fig.tight_layout()


##############################################################
weight_categories = ['vlow_weight','low_weight','mid_weight','high_weight','vhigh_weight']
players['weight_class'] = pd.qcut(players['weight'],len(weight_categories),weight_categories)

height_categories = ['vlow_height','low_height','mid_height','high_height','vhigh_height']
players['height_class'] = pd.qcut(players['height'],len(height_categories),height_categories)

players['skintone_class'] = pd.qcut(players['skintone'],3)



###################################################
#漂亮方便的特征输出报告
import pandas_profiling
pandas_profiling.ProfileReport(players)


#############################
#特征重塑

players.birthday.head()

players['birth_date'] = pd.to_datetime(players.birthday,format = '%d.%m.%Y')
players['age_years'] = (pd.to_datetime('2013-01-01') - players['birth_date']).dt.days)/365.25




players_cleaned_variables = players.columns.tolist()
players_cleaned_variables

players_cleaned_variables = ['height','weight','position','photoID','rater1','rater2','skintone','position_agg','weight_class','height_class','skintone_class','birth_date','age_years']

pandas_profiling.ProfileReport(players[players_cleaned_variables])
players[players_cleaned_variables].to_csv('cleaned_players.csv.gz',compression = 'gzip')



###############################
agg_dyads = pd.read_csv('raw_dyads.csv.gz',compression = 'gzip',index_col=[0,1])

all(agg_dyads['games'] == agg_dyads.victories + agg_dyads.ties + agg_dyads.defeats)

agg_dyads['totalRedCards'] = agg_dyads['yellowReds'] + agg_dyads['redCards']
agg_dyads.rename(columns = {'redCards:'strictRedCards},inplace = True)

###############################################
clean_players = load_subgroup('clean_players.csv.gz')
temp = tidy_dyads.reset_index().set_index('playerShort').merge(clean_players,left_index = True,right_index = True)

tidy_dyads.groupby(level = 0).sum().sort_values('redcard',ascending=False).rename(columns = {'redcard':'total redcards given'})  #对于有多列索引的数据集，groupby（level = 0）表示对第一列索引进行groupby

tidy_dyads.groupby(level = 1).sum().sort_values('redcard',ascending=False).rename(columns = {'redcard':'total redcards received'})

total_ref_games = tidy_dyads.groupby(level = 0).size().sort_values(ascending=False)
total_player_games = tidy_dyads.groupby(level = 1).size().sort_values(ascending=False)

total_ref_given = tidy_dyads.groupby(level = 0).sum().sort_values(ascending=False,by = 'redcard')
total_player_received = tidy_dyads.groupby(level = 1).sum().sort_values(ascending=False,by = 'redcard')

sns.distplot(total_player_received,kde = False)

tidy_dyads.groupby(level = 1).sum().sort_values(ascending=False,by = 'redcard').head()

tidy_dyads.sum()
tidy_dyads.counts()
tidy_dyads.sum() / tidy_dyads.counts()

player_ref_game = (tidy_dyads.reset_index().set_index('playerShort').merge(clean_players,left_index = True,right_index = True))


bootstrap = pd.concat([player_ref_game.sample(replace = True,n = 1000).groupby('skintone').mean() for _ in range(100)])
player_ref_game.sample(replace = True,n = 10000).groupby('skintone').mean()

ax = sns.regplot(bootstrap.index.values,y = 'redcard',data = bootstrap,lowess = True,scatter_kws={'alpha':0.4},x_jitter=(0.125/4))
ax.set_xlabel('skintone')