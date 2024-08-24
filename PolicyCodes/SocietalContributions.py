#%% 0.1 导入基本的包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB,max_
import sys
import geopandas as gpd
import matplotlib as mpl
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.patches import Arrow
num_province=31
province_basicinfo=pd.read_excel('../数据整理/各省基本信息（英文版）.xlsx',index_col=1,nrows=num_province)
province_list=list(province_basicinfo.index)
plt.rcParams['font.size']=24
plt.rcParams['font.family']='Arial'

Exchange_rate=7 # 汇率

#%% 0.2 首先设定一些基本参数和函数
year_list=np.arange(2025,2061) # ref列表存储的是2023年开始的数据

# 全国的平均风电和光伏比例（参考链接：https://mp.weixin.qq.com/s/yoRzYcoAuPMeJ698fNVaoQ）
Wind_Accom_National_Rate_2024=0.959
Solar_Accom_National_Rate_2024=0.967

num_province=31
excel_name='../20240219整理文件/4.(测试2可相信）重新整合各省平衡表20240322.xlsx'
df_transfer_byyear_list=[]
for year in year_list:
    df_transfer_byyear=pd.read_excel(excel_name,sheet_name=str(year),usecols='A:AH',
        skiprows=[1,2],nrows=num_province,index_col=0).fillna(0.0)
    df_transfer_byyear[['BalanceCheck','Load','Hydro','Nuclear','Solar','Wind',
        'Others','Gen-Load','RenewableOut','RenewableIn','Gen-Load-RenewOut+RenewIn',
        'HydroOut','HydroIn','CoalFire','CoalFireOut','CoalFireIn','RawCoalFire(SelfGen)',
        'NewCoalFire(SelfGen)','RawCoalPurchase','NewCoalPurchase','MaxSellCoal','MaxSellCoal→SellPower','MaxBuyCoal→BuyPower','SellCoal→SellPower',
        'BuyCoal→BuyPower','CapacityUHV','RemainUHVForCoals','RenewableOut(UHV)','RenewableIn(UHV)']]/=10 #将亿千瓦时改变为TWh，方便成文 
    df_transfer_byyear_list.append(df_transfer_byyear)
df_datasource=pd.read_excel(excel_name,sheet_name='1.SourceData',usecols='A:D',index_col=0,nrows=7410)
df_datasource['Power']/=10 #将单位从亿千瓦时改为TWh
df_datasource_pivoted=df_datasource.reset_index().pivot_table(values='Power',index=['Province','Year'],columns='Type').fillna(0.0)
df_datasource_pivoted_aggbyyear=df_datasource_pivoted.groupby(by='Year').agg(sum).iloc[1:]

#%% 1.1 环境因素的影响
# 如果按照目前的新能源消纳比例，倒未来可能就不是这个数了(但是现在还都)
Total_Wind_Accom=pd.Series(df_datasource_pivoted_aggbyyear.loc[2023,'Wind']+np.arange(len(year_list)+2)*(df_datasource_pivoted_aggbyyear.loc[2024,'Wind']-df_datasource_pivoted_aggbyyear.loc[2023,'Wind']),index=np.insert(year_list,0,[2023,2024]))
Total_Solar_Accom=pd.Series(df_datasource_pivoted_aggbyyear.loc[2023,'Solar']+np.arange(len(year_list)+2)*(df_datasource_pivoted_aggbyyear.loc[2024,'Solar']-df_datasource_pivoted_aggbyyear.loc[2023,'Solar']),index=np.insert(year_list,0,[2023,2024]))
# Total_Wind_Accom_2024=df_datasource_pivoted_aggbyyear.loc[2024,'Wind']*Wind_Accom_National_Rate_2024
# Total_Solar_Accom_2024=df_datasource_pivoted_aggbyyear.loc[2024,'Solar']*Solar_Accom_National_Rate_2024
Wind_Accom_National_Rate_byyear_list=np.minimum(Total_Wind_Accom*Wind_Accom_National_Rate_2024/df_datasource_pivoted_aggbyyear['Wind'],0.98).iloc[2:]
Solar_Accom_National_Rate_byyear_list=np.minimum(Total_Solar_Accom*Solar_Accom_National_Rate_2024/df_datasource_pivoted_aggbyyear['Solar'],0.98).iloc[2:]
Renewable_Discard_National_byyear_list=(1-Wind_Accom_National_Rate_byyear_list)*df_datasource_pivoted_aggbyyear['Wind'].iloc[2:]+(1-Solar_Accom_National_Rate_byyear_list)*df_datasource_pivoted_aggbyyear['Solar'].iloc[2:]
# 发一度电会带来多少的废弃气体排放
# 数据出处：https://www.sciencedirect.com/science/article/pii/S1364032 117309206
CO2_kgperMWh=670
SOx_kgperMWh=0.25
NOx_kgperMWh=0.20
CO2_ton_byyear=CO2_kgperMWh*Renewable_Discard_National_byyear_list/1e3
SOx_ton_byyear=SOx_kgperMWh*Renewable_Discard_National_byyear_list/1e3
NOx_ton_byyear=NOx_kgperMWh*Renewable_Discard_National_byyear_list/1e3

# Ash的Content构成和Sludge(million tons)
Fly_Ash_2000=87
Bottom_Ash_2000=22
Boiler_Slag_2000=11
Total_Ash_2000=Boiler_Slag_2000+Fly_Ash_2000+Bottom_Ash_2000
Total_FGD_Sludge_2000=50

# 根据https://www.eia.gov/tools/faqs/faq.php?id=667&t=2 ，煤炭的发电率约为1.14 pounds/kWh,而1pound=0.4536kg；
# 而根据https://www.epa.gov/sites/default/files/2015-08/documents/coal-rtc.pdf ，ash的比例约为10.1%
Ash_Rate=0.1
Total_Ash_MillionTons=Renewable_Discard_National_byyear_list*1.14*1e9*0.4536/1e3*Ash_Rate/1e6
Bottom_Ash_MillionTons=Total_Ash_MillionTons*Bottom_Ash_2000/Total_Ash_2000
Fly_Ash_MillionTons=Total_Ash_MillionTons*Fly_Ash_2000/Total_Ash_2000
Boiler_Slag_MillionTons=Total_Ash_MillionTons*Boiler_Slag_2000/Total_Ash_2000
FGD_Sludge_MillionTons=Total_Ash_MillionTons/Total_Ash_2000*Total_FGD_Sludge_2000

#%% 1.2 社会福祉
# 看下煤电发电大省都是哪些
# label='SellCoal→SellPower'
# curr_year=2045
# for idx_year,year in enumerate(year_list):
#     if year==curr_year:
#         df_transfer_byyear=df_transfer_byyear_list[idx_year]
#         NewCoalFire=-df_transfer_byyear[[label]].loc[province_list]
#         NewCoalFire['GDP-PerCapita']=province_basicinfo['GDP-PerCapita']
#         print(NewCoalFire)# plt.bar(df_transfer_byyear.index,df_transfer_byyear['NewCoalFire(SelfGen)'])

# #np.corrcoef(NewCoalFire['GDP-PerCapita'],NewCoalFire['NewCoalFire(SelfGen)'])

# province_cn2en_dict=dict(zip(province_basicinfo['Province-CN'],province_basicinfo.index))
# file=r"../数据整理/中国省级地图GS（2019）1719号.geojson"
# china_main=gpd.read_file(file).rename(columns={'CNAME':'省份'}).set_index('省份').drop(index=['香港','澳门','台湾'])
# china_main['Province']=china_main.index.map(province_cn2en_dict)
# china_main=china_main.reset_index().set_index('Province').loc[province_list]
# china_main_crs=china_main.geometry.to_crs(epsg=2343).to_frame()
# china_main_crs=china_main_crs.merge(province_basicinfo[['Lat','Lon','Region']],left_index=True,right_index=True)
# china_main_crs['Lon-Center']=china_main_crs['geometry'].centroid.map(lambda t:t.x)
# china_main_crs['Lat-Center']=china_main_crs['geometry'].centroid.map(lambda t:t.y)
# china_main_crs['GDP-PerCapita']=province_basicinfo['GDP-PerCapita']
# province_coordinates=gpd.GeoDataFrame(province_basicinfo[['Lon','Lat']], geometry=gpd.points_from_xy(province_basicinfo['Lon'],province_basicinfo['Lat'],crs='EPSG:4326')) #EPSG就是经典的（经度，纬度）表示法
# province_coordinates_converted=province_coordinates.geometry.to_crs(epsg=2343) # 替换成epsg中经典的表示法
# xlim=(-2490128.154711554, 2968967.6930187442)
# ylim=(1.8e6,6.1e6)
# def get_xrel_yrel(ax_pos=mpl.transforms.Bbox([[0.12909373348498837,0.125],[0.8959062665150117,0.88]])):
#     df_xrel_yrel=pd.DataFrame(0.0,index=province_list,columns=['x_abs','y_abs','x_rel_fig','y_rel_fig','x_rel_ax','y_rel_ax'])
#     for province in province_list:
#         x,y=china_main_crs.loc[province,'Lon-Center'],china_main_crs.loc[province,'Lat-Center']
#         x_rel_fig=(x-xlim[0])/(xlim[1]-xlim[0])*(ax_pos.x1-ax_pos.x0)+ax_pos.x0 #省会相对于fig的位置（注意有别于相对于ax的位置）
#         y_rel_fig=(y-ylim[0])/(ylim[1]-ylim[0])*(ax_pos.y1-ax_pos.y0)+ax_pos.y0
#         x_rel_ax=(x-xlim[0])/(xlim[1]-xlim[0])
#         y_rel_ax=(y-ylim[0])/(ylim[1]-ylim[0])
#         df_xrel_yrel.loc[province,'x_abs']=x
#         df_xrel_yrel.loc[province,'y_abs']=y
#         df_xrel_yrel.loc[province,'x_rel_fig']=x_rel_fig
#         df_xrel_yrel.loc[province,'y_rel_fig']=y_rel_fig
#         df_xrel_yrel.loc[province,'x_rel_ax']=x_rel_ax
#         df_xrel_yrel.loc[province,'y_rel_ax']=y_rel_ax
#     return df_xrel_yrel

# fig,ax=plt.subplots(figsize=(10,8),dpi=100)
# ax.set_ylim(ylim)
# cbar_ax=fig.add_axes([0.8,0.2,0.03,0.4])
# china_main_crs.plot(ax=ax,edgecolor='black',facecolor='white')
# china_main_crs.plot('GDP-PerCapita',ax=ax,legend=True,cax=cbar_ax,edgecolor='black')
# # ax.annotate('CBS Power \n Proportion',xy=(0.62,0.065),xycoords='figure fraction')
# # ax.annotate('Subsidy',xy=(0.11,0.71),xycoords='figure fraction',color='#E07117',weight='bold')
# ax_pos=ax.get_position()
# max_newcoalfire_selfgen=NewCoalFire[label].max()
# for province in province_list:
#     if NewCoalFire.loc[province,label]>0.0:
#         df_xrel_yrel=get_xrel_yrel(ax_pos)
#         x_rel,y_rel=df_xrel_yrel.loc[province,'x_rel_fig'],df_xrel_yrel.loc[province,'y_rel_fig']
#         bar_ax=fig.add_axes([x_rel-0.02,y_rel-0.03,0.02,0.1])
#         bar_ax.set_ylim([0,max_newcoalfire_selfgen])
#         bar_ax.bar(np.arange(1),NewCoalFire.loc[province,label],color='orange',edgecolor='black')
#         bar_ax.set_axis_off()
# ax.axis('off') 
# fig.patch.set_alpha(0.0)
# ax.patch.set_alpha(0.0)
# fig.savefig('图片/废弃.pdf',format='pdf',bbox_inches='tight')

# #%%1.2.2用另一个角度去绘图，绘制GDP和火电传输的双y轴图（毕竟信息量没那么大）
# curr_year=2045
# label='SellCoal→SellPower'
# Exchange_Rate=7.0
# fig,ax=plt.subplots(figsize=(12,6))
# NewCoalFire_sorted=NewCoalFire.sort_values(by='GDP-PerCapita',ascending=True)
# ax2=ax.twinx()
# ax.bar(NewCoalFire_sorted.index,NewCoalFire_sorted[label],color='#8B4000')
# ax2.scatter(NewCoalFire_sorted.index,NewCoalFire_sorted['GDP-PerCapita']/Exchange_Rate,marker='x',s=100,color='blue')
# ax.tick_params(axis='x', labelrotation=70)
# ax.set_ylabel('Coal-Fired Power (TWh)',color='#8B4000')
# ax2.set_ylabel('GDP Per Capita (US$)',color='blue')
# ax.yaxis.set_label_coords(-0.08,0.4)
# ax2.yaxis.set_label_coords(1.15,0.45)
# xlabels=list(NewCoalFire_sorted.index)
# for i in range(len(xlabels)):
#     province=xlabels[i]
#     if province=='InnerMongolia':
#         xlabels[i]='Inner\nMongolia'
#     if not NewCoalFire_sorted.loc[province,label]>0.0:
#         xlabels[i]=''
# ax.set_xticks(NewCoalFire_sorted.index,xlabels,fontsize=22)
# ax.set_xlabel('Province')
# ax.xaxis.set_label_coords(0.9,-0.2)
# ax.spines['right'].set_color('#8B4000')
# ax.tick_params(axis='y', colors='#8B4000')
# ax2.spines['right'].set_color('blue')
# ax2.tick_params(axis='y',colors='blue')
# fig.tight_layout()
# fig.savefig('图片/废弃2.pdf',format='pdf',bbox_inches='tight')

#%%1.3 环境问题的集中部分
# 各个省份的发电量
label='NewCoalFire(SelfGen)'
curr_year=2045
fig,ax=plt.subplots(figsize=(12,6))
original_coalfire=df_transfer_byyear_list[curr_year-year_list[0]].loc[province_list,'CoalFire']*1.14*1e9*0.4536/1e3/1e6
proposed_coalfire=(df_transfer_byyear_list[curr_year-year_list[0]].loc[province_list,'NewCoalFire(SelfGen)']+df_transfer_byyear_list[curr_year-year_list[0]].loc[province_list,'NewCoalPurchase'])*1.14*1e9*0.4536/1e3/1e6
province_list_sorted=proposed_coalfire.sort_values(ascending=False).index
ax.bar(np.arange(len(province_list_sorted))-0.2,original_coalfire.loc[province_list_sorted],color='#A4514F',width=0.3,label='Current Framework')
# fig,ax=plt.subplots(figsize=(12,6))
ax.bar(np.arange(len(province_list_sorted))+0.2,proposed_coalfire.loc[province_list_sorted],color='#2e7ebb',width=0.3,label='Proposed Framework')

ax.set_xlabel('Province Index')
ax.set_ylabel('Combusted Coals\n(Million Tonnes)')
from collections import defaultdict
dict_change_yaxis=defaultdict(float)
dict_change_xaxis=defaultdict(float)
dict_change_yaxis['Shandong']+=5
dict_change_yaxis['Hebei']+=5.0
dict_change_yaxis['Jiangsu']-=0.0
dict_change_xaxis['Jiangsu']-=0.2
dict_change_yaxis['Guangdong']-=20.0
dict_change_yaxis['Xinjiang']-=10.0
dict_change_xaxis['Xinjiang']+=0.5
dict_change_xaxis['Hubei']-=1.5
dict_change_yaxis['Henan']+=0.0
dict_change_xaxis['Henan']-=0.5
dict_change_xaxis['Shanghai']-=0.5
dict_change_xaxis['InnerMongolia']+=0.5
dict_change_yaxis['InnerMongolia']-=10
provinces_coalrich=['InnerMongolia','Xinjiang','Shanxi']
provinces_idx_coalrich={province:list(province_list_sorted).index(province) for province in provinces_coalrich}
ax.scatter([provinces_idx_coalrich[province]+dict_change_xaxis[province]-0.5 for province in provinces_coalrich],[proposed_coalfire.loc[province]+dict_change_yaxis[province]+5 for province in provinces_coalrich],marker='*',color='red',s=300,label='Coal-Rich Provinces')

for province_sorted_idx in range(6):
    province=province_list_sorted[province_sorted_idx]
    ax.annotate(province+' ('+str(int(province_basicinfo.loc[province,'GDP-PerCapita']/1000))+'k$)',xy=(province_sorted_idx+dict_change_xaxis[province],proposed_coalfire.loc[province]+dict_change_yaxis[province]),color='#2e7ebb')
for province_sorted_idx in range(len(province_list)):
    province=province_list_sorted[province_sorted_idx]
    if province in []:
        ax.annotate(province+' ('+str(int(province_basicinfo.loc[province,'GDP-PerCapita']/1000))+'k$)',xy=(province_sorted_idx+dict_change_xaxis[province],original_coalfire.loc[province]+dict_change_yaxis[province]),color='#A4514F')
    elif province in ['Hubei','Guangdong','Jiangsu','Zhejiang','Shanghai']:
        ax.annotate(province+'\n('+str(int(province_basicinfo.loc[province,'GDP-PerCapita']/1000))+'k$)',xy=(province_sorted_idx+dict_change_xaxis[province],original_coalfire.loc[province]+dict_change_yaxis[province]),color='#A4514F')
handles,labels=plt.gca().get_legend_handles_labels()
order=[1,2,0]
ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],bbox_to_anchor=(1.015,1.01))
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)
fig.savefig('图片/Fig5-3-环境与GDP.pdf',format='pdf',bbox_inches='tight')

#%% 省掉的煤电随着年份的增长
# 碳捕捉度电成本：可认为是0.35元/度，参考https://mp.weixin.qq.com/s/XFRkfCJ4jTrvrl1rjjDLcg
Saved_Coal=Renewable_Discard_National_byyear_list*1.14*1e9*0.4536/1e3/1e6
CCS_unit_price=0.35/Exchange_rate

fig,ax=plt.subplots(figsize=(10,4))
ax.bar(year_list,Saved_Coal,color='#A4514F',alpha=0.7,width=0.7)

ax.set_ylabel('Saved Coal\n(Million Tonnes)')
ax.set_xlabel('Year')
ax.yaxis.set_label_coords(-0.1,0.4)
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)
fig.savefig('图片/Fig5-2-省去的煤炭.pdf',format='pdf',bbox_inches='tight')

#%% 就业机会和GDP变化
#中国2020年煤电量为4917.7TWh（https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://new.qq.com/rain/a/20220305A01ZVO00&ved=2ahUKEwiiqIHl3pSHAxVLr1YBHZK7BPIQFnoECCkQAQ&usg=AOvVaw2JhKWa8eV01tJ6No_EAy8a），
#总就业机会为300万个（https://mp.weixin.qq.com/s/lYKxyWp6JkWNy88Vr9yYwA）。
#那么可以以新疆为例，看出创造的新就业机会
curr_province_list=['Xinjiang','InnerMongolia','Shanxi']
df_coal2power=pd.DataFrame(index=curr_province_list,columns=year_list)
for year in year_list:
    df_coal2power[year]=-df_transfer_byyear_list[year-year_list[0]]['SellCoal→SellPower']

# 将TWh转变为就业机会（千人）
df_employment=df_coal2power/4917.7*300*10000
fig,ax=plt.subplots(figsize=(10,4))
ax.plot(year_list[:21],df_employment.loc['InnerMongolia'][:21],'.-',label='Inner Mongolia',markersize=15)
ax.plot(year_list[:21],df_employment.loc['Xinjiang'][:21],'.-',label='Xinjiang',markersize=10,marker='>')
ax.plot(year_list[:21],df_employment.loc['Shanxi'][:21],'.-',label='Shanxi',markersize=10,marker='<',color='purple')
ax.set_xlabel('Year')
ax.set_ylabel('Employment')
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)
ax.ticklabel_format(axis='y',style='sci',scilimits=(3,5))
ax.legend(bbox_to_anchor=(1.05,-0.2),ncol=3)
# 每度电带来的利润变化是多大呢
# 一吨煤的发电成本约135元/吨，转化成元/度
Coal_Profit_perkWh=135/(1000/1.14*0.4536)/Exchange_rate
Electricity_Profit_perkWh=0.1/Exchange_rate
#fig.savefig('图片/Fig5-4-增加的就业机会.pdf',format='pdf',bbox_inches='tight')

#%% 看一下煤电的发电熵是多少
# 公式为∑pilog(pi)
curr_year=2045
coal_fire_original=df_transfer_byyear_list[curr_year-year_list[0]].loc[province_list,'CoalFire']
coal_fire_original_normalized=np.array(coal_fire_original)/coal_fire_original.sum()
coal_fire_proposed=df_transfer_byyear_list[curr_year-year_list[0]].loc[province_list,'NewCoalFire(SelfGen)']\
    +df_transfer_byyear_list[curr_year-year_list[0]].loc[province_list,'SellCoal→SellPower']
coal_fire_proposed_normalized=np.array(coal_fire_proposed)/coal_fire_proposed.sum()
from scipy.stats import entropy
print(entropy(coal_fire_original_normalized)) #3.1243460367710445
print(entropy(coal_fire_proposed_normalized)) #2.7793464396401792
# 熵越小表示有序程度越大（coal_fire_proposed_normalized)，即样本之间差异越大

#%% 最后再给出方法出radar plot，评估安全性等指标
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

def radar_factory(num_vars, frame='circle'):
    # calculate evenly-spaced axis angles
    theta = np.linspace(0,2*np.pi,num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = 'radar'
        PolarTransform = RadarTransform
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)
    register_projection(RadarAxes)
    return theta


labels=['              Sustainability','Flexibility','Security','Spontaneity   ',
        '         Economy','     Eco-\n     Friendly','Social\nWelfare']
titles=['Current','Proposed']
df_radarplot=pd.DataFrame(index=labels,columns=titles)
df_radarplot[titles[0]]=[0.853,0.311,0.1658,0.3,0.3772,0.963,0.332]
df_radarplot[titles[1]]=[1.00,0.926,0.821,0.875,0.807,1.0,0.652]
num_labels=len(labels)
num_plot=len(titles)
theta=radar_factory(num_labels,frame='polygon')

fig,axs=plt.subplots(figsize=(9,9),nrows=1,ncols=2,
                        subplot_kw=dict(projection='radar'))
fig.subplots_adjust(wspace=0.25,hspace=0.20,top=0.85,bottom=0.05)

colors=['blue','green']
# Plot the four cases from the example data on separate Axes
for idx_plot in range(num_plot):
    axs[idx_plot].set_rgrids([0.0,0.2,0.4,0.6,0.8,1.0])
    axs[idx_plot].set_rticks([0.2,0.4,0.6,0.8,1.0],[1,'',3,'',5])
    axs[idx_plot].set_rlim([0.0,1.0])
    axs[idx_plot].set_title(titles[idx_plot],weight='bold',size='medium',position=(0.5,2),
                 horizontalalignment='center',verticalalignment='center')
    axs[idx_plot].plot(theta,df_radarplot.iloc[:,idx_plot],color=colors[idx_plot])
    axs[idx_plot].fill(theta,df_radarplot.iloc[:,idx_plot],facecolor=colors[idx_plot],alpha=0.25)
    axs[idx_plot].set_varlabels(labels)
fig.tight_layout()
#fig.savefig('图片/Fig5-6-雷达图.pdf',bbox_inches='tight')