import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

data = pd.read_csv('aquastat.csv.gzip',compression='gzip')

data.head()

data.shape

data.info()

data[['variable','variable_full']].drop_duplicates()

data.country.nunique()

countries = data.country.unique()

data.time_period.nunique()

time_periods = data.time_period.unique()
mid_periods = range(1960,2017,5)

data[data.variable == 'total_area'].value.isnull().sum()


##通过切片分析数据

def time_slice(df,time_period):
    df = df[df.time_period == time_period]

    df = df.pivot(index = 'country',columns = 'variable',values = 'value')
    return df

time_slice(data,time_periods[0])



def country_slice(df,country):
    df = df[df.country == country]

    df = df.pivot(index = 'variable',columns = 'time_period',values='value')

    df.index.name = country
    return df

country_slice(data,countries[40])

def variable_slice(df,variable):
    df = df[df.variable == variable]

    df.pivot(index = 'country',columns='time_period',values = 'value')
    return df

variable_slice(data,'total_pop')



def time_series(df,country,variable):
    series = df[(df.country == country) & (df.variable == variable)]
    series = series.dropna()[['year_measured','value']]
    series.year_measured = series.year_measured.astype(int)
    series.set_index('year_measured',inplace = True)
    series.columns = [variable]
    return series

time_series(data,'Belarus','total_pop')

data.region.unique()

simple_regions ={
    'World | Asia':'Asia',
    'Americas | Central America and Caribbean | Central America': 'North America',
    'Americas | Central America and Caribbean | Greater Antilles': 'North America',
    'Americas | Central America and Caribbean | Lesser Antilles and Bahamas': 'North America',
    'Americas | Northern America | Northern America': 'North America',
    'Americas | Northern America | Mexico': 'North America',
    'Americas | Southern America | Guyana':'South America',
    'Americas | Southern America | Andean':'South America',
    'Americas | Southern America | Brazil':'South America',
    'Americas | Southern America | Southern America':'South America',
    'World | Africa':'Africa',
    'World | Europe':'Europe',
    'World | Oceania':'Oceania'
}

data.region = data.region.apply(lambda x: simple_regions[x])

print(data.region.unique())

def subregion(data,region):
    return data[data.region ==  region]




#######################################################
##单变量分析
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import missingno as msno
import pandas_profiling


# Update matplotlib defaults to something nicer
mpl_update = {'font.size':16,
              'xtick.labelsize':14,
              'ytick.labelsize':14,
              'figure.figsize':[12.0,8.0],
              'axes.color_cycle':['#0055A7', '#2C3E4F', '#26C5ED', '#00cc66', '#D34100', '#FF9700','#091D32'],
              'axes.labelsize':20,
              'axes.labelcolor':'#677385',
              'axes.titlesize':20,
              'lines.color':'#0055A7',
              'lines.linewidth':3,
              'text.color':'#677385'}
mpl.rcParams.update(mpl_update)

data = pd.read_csv('aquastat.csv.gzip',compression = 'gzip')
data.region = data.region.apply(lambda x: simple_regions[x])

recent = time_slice(data,'2013-2017')
msno.matrix(recent,labels = True)

#######水资源缺失值情况
msno.matrix(variable_slice(data,'exploitable_total'),inline=False,sort='descending')
plt.xlabel('Time period')
plt.ylabel('Country')
plt.title('Missing total exploitable water resources data across countries and time periods \n \n \n \n')

#因为水资源那一变量缺失值太多所以舍弃这一变量
data = data.loc[~data.variable.str.contains('exploitable'),:]

################看全国降雨指数变量
msno.matrix(variable_slice(data,'national_rainfall_index'),inline = False,sort = 'descending')
plt.xlabel('Time period')
plt.ylabel('Country')
plt.title('Missing national rainfall index data across countries and time periods \n \n \n \n')

############################
north_america = subregion(data,'North America')
##指数完整性
msno.matrix(msno.nullity_sort(time_slice(north_america,'2013-2017'),sort = 'descending').T,inline = False)

##############################
#抽查巴哈马缺少哪些数据获得更多了解
msno.nullity_filter(country_slice(data,'Bahamas').T,filter='bottom',p = 0.1)


#####################################
#区域与缺失值之间的关系
#画地图
geo = r'world.json' #地图坐标值
null_data = recent['agg_to_gdp'].notnull()*1
map = folium.Map(location=[48,-102],zoom_start=2)
map.choropleth(geo_path = geo,data = null_data,columns = ['country','agg_to_gdp'],key_on='feature.properties.name',reset=True,fill_color='GnBu',fill_opacity=1,line_opacity=0.2,legend_name='Missing agricultural contribution to GDP data 2013-2017')
map

###############################################

fig,ax = plt.subplots(figsize = (16,16))
sns.heatmap(data.groupby(['time_period','variable']).value.count().unstack().T,ax = ax)
plt.xticks(rotation = 45)
plt.xlabel('Time period')
plt.ylabel('Variable')
plt.title('Number of countries with data reported for each variable over time')

###########################################
#profiling可视化
pandas_profiling.ProfileReport(time_slice(data,'2013-2017'))







###################################################
#单变量分析可视化

data = pd.read_csv('aquastat.csv.gzip')
data.region = data.region.apply(lambda x: simple_regions[x])
data = data.loc[~data.variable.str.contains('exploitable'),:]
data = data.loc[~(data.variable == 'national_rainfall_index')]

recent = time_slice(data,'2013-2017')

recent[['total_pop','urban_pop','rural_pop']].describe().astype(int)

recent.sort_values('rural_pop')[['total_pop','urban_pop','rural_pop']]

time_series(data,'Qatar','total_pop').join(time_series(data,'Qatar','urban_pop')).join(time_series(data,'Qatar','rural_pop'))

#############################
#检查数据分布情况
recent[['total_pop','urban_pop','rural_pop']].describe().astype(int)

#检查峰度与偏斜度
#左偏：中位值》均值
import scipy
from scipy import stats
recent[['total_pop','urban_pop','rural_pop']].apply(scipy.stats.skew)
recent[['total_pop','urban_pop','rural_pop']].apply(scipy.stats.kurtosis)

#画直方图
fig,ax = plt.subplots(figsize = (12,8))
ax.hist(recent.total_pop.values,bins = 50)
ax.set_xlabel('Total population')
ax.set_ylabel('Number of countries')
ax.set_title('Distribution of population of countries 2013-2017')

#因为直方图偏度较大，应用log变换
#对数变化可以解决数据不均匀（偏度、峰度）或者非正态化的问题
recent[['total_pop']].apply(np.log).apply(scipy.stats.skew)
recent[['total_pop']].apply(np.log).apply(scipy.stats.kurtosis)


########################################
plt.plot(time_series(data,'United State of America','total_pop'))
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('United States population over time')


#region
with sns.color_palette(sns.diverging_palette(220,280, s=85, l=25, n=23)):
    north_america = time_slice(subregion(data,'North America'),'1958-1962').sort_values('total_pop').index.tolist()
    for country in north_america:
        plt.plot(time_series(data,country,'total_pop'),label = country)
        plt.xlabel('Year')
        plt.ylabel('Population')
        plt.title('North American population over time')
    plt.legend(loc = 2,prop = {'size':10})

#上面很难看到国家人口增长情况，应该看每个国家人口最小值作为起始点的增长情况

with sns.color_palette(sns.diverging_palette(220,280,s=85,l=25,n=23)):
    for country in north_america:
        ts = time_series(data,country,'total_pop')
        ts['norm_pop'] = ts.total_pop / ts.total_pop.min() * 100
        plt.plot(ts['norm_pop'],label = country)
        plt.xlabel('Year')
        plt.ylabel('Percent increase in population')
        plt.title('Percent increase in population from 1960 in North American countries')
        plt.legend(loc = 2,prop = {'size':10})



##############################
#热力图可能显示效果更好
north_america_pop = variable_slice(subregion(data,'North America'),'total_pop')
north_america_norm_pop = north_america_pop.div(north_america_pop.min(axis=1), axis=0)*100
north_america_norm_pop = north_america_norm_pop.loc[north_america]

fig, ax = plt.subplots(figsize=(16, 12));
sns.heatmap(north_america_norm_pop, ax=ax, cmap=sns.light_palette((214, 90, 60), input="husl", as_cmap=True))
plt.xticks(rotation=45)
plt.xlabel('Time period')
plt.ylabel('Country, ordered by population in 1960 (<- greatest to least ->)')
plt.title('Percent increase in population from 1960')



###############################
#探索变量间关系
plt.scatter(recent.seasonal_variability,recent.gdp_per_capita)
plt.xlabel('Seasonal variability')
plt.ylabel('GDP per capita ($USD/person)')

def plot_scatter(df,x,y,xlabel = None,ylabel = None,title = None,logx = False,logy = False,by = None,ax = None):
    if not ax:
        fig,ax = plt.subplots(figsize = (12,10))
    colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
    if by:
        groups = df.groupby(by)
        for j,(name,group) in enumerate(groups):
            ax.scatter(group[x],group[y],color = colors[j],label = name)
        ax.legend()
    else:
        ax.scatter(df[x],df[y],color = colors[0])
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')

    ax.set_xlabel(xlabel if xlabel else x)
    ax.set_ylabel(ylabel if ylabel else y)
    if title:
        ax.set_title(title)
    return ax


###################################
#seaborn的散点图与条形图混合图
svr = [recent.seasonal_variability.min(),recent.seasonal_variabillity.max()]
gdpr = [(recent.gdp_per_capita.min()), recent.gdp_per_capita.max()]
gdpbins = np.logspace(*np.log10(gdpr), 25)



g = sns.JointGrid(x = 'seasonal_variability',y = 'gdp_per_capita',data = recent,ylim = gdpr)
g.ax_marg_x.hist(recent.seasonal_variability,range = svr)
g.ax_marg_y.hist(recent.gdp_per_capita,range = gdpr,bins = gdpbins,orientation='horizontal')
g.plot_joint(plt.hexbin,gridsize = 25)
ax = g.ax_joint
g.fig.set_figheight(8)
g.fig.set_figwidth(9)


###相关性
recent_corr = recent.corr().loc['gdp_per_capita'].drop(['gdp','gdp_per_capita'])
def conditional_bar(series,bar_colors = None,color_labels = None,figsize = (13,24),xlabel = None,by = None,ylabel = None,title = None):
    fig,ax = plt.subplots(figsize = figsize)
    if not bar_colors:
        bar_colors = mpl.rcParams['axes.prop_cycle'].by_key()['color'][0]
    plt.barh(range(len(series)),series.values,color = bar_colors)
    plt.xlabel('' if not xlabel else xlabel)
    plt.ylabel('' if not ylabel else ylabel)
    plt.yticks(range(len(series)),series.index.tolist())
    plt.title('' if not title else title)
    plt.ylim([-1,len(series)])
    if color_labels:
        for col, lab in color_labels.items():
            plt.plot([],linestyle = '',marker = 's',c = col,label = lab)
        lines,labels = ax.get_legend_handles_labels()
        ax.legend(lines[-len(color_labels.keys()):],labels[-len(color_labels.keys()):],loc = 'upper right')
    plt.close()
    return fig

bar_colors = ['#0055A7' if x else '#2C3E4F' for x in list(recent_corr.values < 0)]
color_labels = {'#0055A7':'Negative correlation', '#2C3E4F':'Positive correlation'}

conditional_bar(recent_corr.apply(np.abs), bar_colors, color_labels,
               title='Magnitude of correlation with GDP per capita, 2013-2017',
               xlabel='|Correlation|')







