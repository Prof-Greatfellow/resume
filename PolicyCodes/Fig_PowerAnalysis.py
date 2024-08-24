import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import geopandas as gpd
import gurobipy as gp
from gurobipy import GRB
import matplotlib as mpl
from matplotlib.patches import Circle,Rectangle
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib.patheffects as patheffects

mpl.rcParams['font.family'] = ['Arial'] #或者是Heiti TC
plt.rcParams['font.size']=18
plt.rcParams['axes.unicode_minus']=False
num_province=31
excel_name='../20240219整理文件/4.(测试2可相信）重新整合各省平衡表20240322.xlsx'
province_basicinfo=pd.read_excel(excel_name,sheet_name='2.ProvinceInfo',index_col=0,nrows=num_province,skiprows=[1,2])
province_property=province_basicinfo['Region'].to_dict() #东西南北中
province_list=list(province_basicinfo.index)
year_list=np.array([year for year in range(2023,2061)])
df_datasource=pd.read_excel(excel_name,sheet_name='1.SourceData',usecols='A:D',index_col=0,nrows=7410)
df_datasource['Power']/=10 #将单位从亿千瓦时改为TWh
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

df_datasource_pivoted=df_datasource[df_datasource['Year']>=2023].reset_index().pivot_table(values='Power',index=['Province','Year'],columns='Type').fillna(0.0)

# 这里有一个大前提：在传统方式下，理论上外发可以全部由传统煤运来完成，而这不现实。
# 所以煤运外发部分保持和一样不变，而是由。

#%% 图1:原本风电光伏和传统火电的变化
df_datasource_pivoted_aggbyyear=df_datasource_pivoted.groupby(by='Year').agg(sum)
fig,ax=plt.subplots(figsize=(10,5))
ax.bar(year_list[2:],df_datasource_pivoted_aggbyyear['CoalFire'].iloc[2:],label='Coal-Fired',color='red')
ax.bar(year_list[2:],df_datasource_pivoted_aggbyyear['Hydro'].iloc[2:],bottom=df_datasource_pivoted_aggbyyear['CoalFire'].iloc[2:],label='Hydro',color='blue')
ax.bar(year_list[2:],df_datasource_pivoted_aggbyyear['Nuclear'].iloc[2:]+df_datasource_pivoted_aggbyyear['Others'].iloc[2:],bottom=df_datasource_pivoted_aggbyyear['Hydro'].iloc[2:]+df_datasource_pivoted_aggbyyear['CoalFire'].iloc[2:],label='Others',color='gray')
ax.bar(year_list[2:],df_datasource_pivoted_aggbyyear['Wind'].iloc[2:]+df_datasource_pivoted_aggbyyear['Solar'].iloc[2:],bottom=df_datasource_pivoted_aggbyyear['CoalFire'].iloc[2:]+df_datasource_pivoted_aggbyyear['Hydro'].iloc[2:]+df_datasource_pivoted_aggbyyear['Nuclear'].iloc[2:]+df_datasource_pivoted_aggbyyear['Others'].iloc[2:],label='Renewable',color='green')

ax.set_xlabel('Year')
ax.set_ylabel('Power (TWh)')
ax.legend(loc='upper left',ncol=2,fontsize=15)
fig.tight_layout()
fig.savefig('图片/SuppFig-PowerTrend.pdf',format='pdf')

temp=df_datasource_pivoted_aggbyyear['Wind']+df_datasource_pivoted_aggbyyear['Solar']
print(temp.iloc[-1]/temp.iloc[0])

#%% 图2:对特高压的需求增长
UHV_demand=pd.read_excel(excel_name,sheet_name='4.UHVCapacity',nrows=num_province,usecols='A:AN',index_col=0)/10 #注意要转换成TWh
UHV_demand_current=np.zeros_like(year_list)
UHV_demand_proposed=np.zeros_like(year_list)
# 如何去计算Strong UHV Demand（现阶段）呢？
UHV_2022=UHV_demand[2022].sum()/2
for year_idx,year in enumerate(year_list):
    df_transfer_byyear=df_transfer_byyear_list[year_idx]
    UHV_demand_current[year_idx]=np.sum(np.abs(df_transfer_byyear['Gen-Load']))/2 # 因为负荷-发电部分是不包括运火电外发的（火电均以运煤方式转出）
    # UHV_demand_proposed为必需部分，应该是电网不能承担的部分+运煤改运火电
    UHV_demand_proposed[year_idx]=(df_transfer_byyear['CapacityUHV']-\
      df_transfer_byyear['RemainUHVForCoals']).sum()/2
UHV_installation_current=UHV_demand[2022].sum()/2*np.power(1.03,year_list-2022)
UHV_installation_proposed=list(UHV_demand[list(year_list)].sum()/2)
fig,ax=plt.subplots(figsize=(10,5))
ax.plot(year_list,UHV_demand_current,marker='^',color='blue',label='Inelastic Demand (Current)')
ax.plot(year_list,UHV_installation_current,linestyle='--',marker='x',color='blue',label='Installation (Current)')
ax.plot(year_list,UHV_demand_proposed,marker='^',color='green',label='Inelastic Demand (Proposed)')
ax.plot(year_list,UHV_installation_proposed,linestyle='--',marker='x',color='green',label='Installation (Proposed)')

ax.set_xlabel('Year')
ax.set_ylabel('Capacity (TWh·year$\mathdefault{^{-1}}$)')
ax.legend(fontsize=16)
fig.tight_layout()
# fig.savefig('图片/UHVCapacity.pdf',format='pdf')

#%% 图3:给定方法下，特高压中新能源的占比问题
renewable_in_UHV_current_list=np.zeros_like(year_list)
renewable_in_UHV_proposed_list=np.zeros_like(year_list)
for year_idx,year in enumerate(year_list):
    #总的UHV外发就是UHV_demand_current，这里就不重复了
    df_transfer_byyear=df_transfer_byyear_list[year_idx]
    renewable_in_UHV_current_list[year_idx]=df_transfer_byyear['RenewableIn'].sum()
    #renewable_in_UHV_proposed_list[year_idx]=(df_transfer_byyear['RemainUHVForCoals']+df_transfer_byyear['SellCoal→SellPower']-df_transfer_byyear['BuyCoal→BuyPower']).sum()/2
    renewable_in_UHV_proposed_list[year_idx]=df_transfer_byyear['RenewableIn(UHV)'].sum()

fig,ax=plt.subplots(figsize=(10,5))
ax.plot(year_list,renewable_in_UHV_current_list/UHV_demand_current,marker='^',label='Current',color='blue')
ax.plot(year_list,renewable_in_UHV_proposed_list/UHV_installation_proposed,marker='o',label='Proposed',color='green')

ax.set_xlabel('Year')
ax.set_ylabel('Penetration Rate')
ax.legend(fontsize=16)
ax.set_ylim([0,1])
fig.tight_layout()
# fig.savefig('图片/UHV-RenewablePenetraion.pdf',format='pdf')

#%% 煤炭运量的减少（和电柜运量的上升）？
buycoal_to_buypower=np.zeros_like(year_list)
rawcoal_purchase=np.zeros_like(year_list)
coal_fire_out=np.ones_like(year_list)
for year_idx,year in enumerate(year_list):
    df_transfer_byyear=df_transfer_byyear_list[year_idx]
    buycoal_to_buypower[year_idx]=df_transfer_byyear['BuyCoal→BuyPower'].sum()
    rawcoal_purchase[year_idx]=df_transfer_byyear['RawCoalPurchase'].sum()
    coal_fire_out[year_idx]=-df_transfer_byyear['CoalFireOut'].sum()
fig,ax=plt.subplots(figsize=(10,5))
ax2=ax.twinx()
#ax.plot(year_list,np.array(rawcoal_purchase)-np.array(buycoal_to_buypower),label='Proposed',marker=9)
ax.bar(year_list,rawcoal_purchase-buycoal_to_buypower,color='#32CD32',alpha=0.8,label='Transported Coal (Proposed)')
ax.bar(year_list,buycoal_to_buypower,bottom=rawcoal_purchase-buycoal_to_buypower,alpha=0.9,label='Transported Coal (Current)',color='#4169E1')
ax2.plot(year_list,buycoal_to_buypower/(buycoal_to_buypower+coal_fire_out),color='brown',marker=9,label='Proportion of Converted Power')

lines,labels=ax.get_legend_handles_labels()
lines2,labels2=ax2.get_legend_handles_labels()
ax2.legend(lines+lines2,labels+labels2,loc='best',fontsize=15)
ax2.spines['right'].set_color('brown')
ax2.tick_params(axis='y', colors='brown')
ax.set_xlabel('Year')
ax.set_ylabel('Coal To Power/TWh')
fig.tight_layout()
# fig.savefig('图片/CoalToPower.pdf',format='pdf')

#%% 在传统和新型情况下，对铁路负载的变化
# 参考源头是《20xx铁道统计公报》（注意铁路）
# https://www.nra.gov.cn/xwzx/xwxx/xwlb/202302/t20230222_339815.shtml
railway_data=pd.read_excel('../20240219整理文件/9.铁路数据20240319.xlsx',nrows=9,index_col=0)#[[i for i in range(2012,2023)]]
year_list_railway=np.array(railway_data.columns)
year_list_future=np.arange(2023,2061)
# fig,ax=plt.subplots(figsize=(10,5))
# ax.plot(year_list_railway,railway_data.loc['铁路运营里程']/railway_data.loc['铁路运营里程',2012],color='black',label='铁路运营里程')
# ax.plot(year_list_railway,railway_data.loc['高铁运营里程']/railway_data.loc['高铁运营里程',2012],color='red',label='高铁运营里程')
# ax.plot(year_list_railway,railway_data.loc['铁路旅客周转量']/railway_data.loc['铁路旅客周转量',2012],color='blue',label='铁路旅客周转量')
# ax.plot(year_list_railway,railway_data.loc['铁路运营里程']*railway_data.loc['电气化率']/railway_data.loc['电气化率',2012]/railway_data.loc['铁路运营里程',2012],color='green',label='电气化铁路')
# ax.legend()

# 那么会让出多少铁路容量来呢？
# 以2012年为基准，计算出高铁和铁路的周转量
# fig,ax=plt.subplots(figsize=(10,5))
#total_turnover=np.array(railway_data.loc['铁路旅客周转量'])
#highspeed_railway_turnover=np.array(railway_data.loc['高铁旅客周转量'])
railway_headcount=np.array(railway_data.loc['铁路客流量'])
highspeed_railway_headcount=np.array(railway_data.loc['高铁客流量'])
# fig,ax=plt.subplots(figsize=(10,5))
# ax.plot(year_list_railway,railway_headcount)
# ax.plot(year_list_railway,highspeed_railway_headcount)
# ax.plot(year_list_railway,total_turnover,color='blue')
# ax.plot(year_list_railway,highspeed_railway_turnover,color='black')
# ax.plot(year_list_railway,total_turnover-highspeed_railway_turnover,lw=5)
# ax.plot(year_list_railway,-(total_turnover-highspeed_railway_turnover)+(total_turnover-highspeed_railway_turnover)[0],lw=5,color='green')

# 那么从2023到2060年，又会是按怎样的趋势发展呢？
# 首先趋势的话，用2014-2018年的增平均速来作为增速
start_year=2014-railway_data.columns[0]
end_year=2019-railway_data.columns[0]
# railway_turnover_rate_increase=np.mean(np.diff(total_turnover)[start_year:end_year])
# highspeed_railway_turnover_rate_increase=np.mean(np.diff(highspeed_railway_turnover)[start_year:end_year])
# railway_future_turnover=total_turnover[-1]+railway_turnover_rate_increase*(year_list_future-year_list_future[0])
# highspeed_railway_future_turnover=highspeed_railway_turnover[-1]+highspeed_railway_turnover_rate_increase*(year_list_future-year_list_future[0])
railway_headcount_rate_increase=np.mean(np.diff(railway_headcount[start_year:end_year]))
railway_future_headcount=railway_headcount[-1]+railway_headcount_rate_increase*(year_list_future-year_list_future[0])
highspeed_railway_headcount_rate_increase=np.mean(np.diff(highspeed_railway_headcount)[start_year:end_year])
highspeed_railway_future_headcount=highspeed_railway_headcount[-1]+highspeed_railway_headcount_rate_increase*(year_list_future-year_list_future[0])
highspeed_railway_future_headcount=np.minimum(highspeed_railway_future_headcount,railway_future_headcount-(railway_future_headcount-highspeed_railway_future_headcount)[2035-2023])

# 可以看出，到2035年之后高铁已经追平铁路，所以后期无需移动铁路数据
# fig,ax=plt.subplots(figsize=(10,5))
# ax.plot(year_list_future,railway_future_headcount)
# ax.plot(year_list_future,highspeed_railway_future_headcount)
# 先看一下增长比例的问题
# 空余出来的容量就是能够额外供给新能源供应的
# 假设一节车厢承载75人，并且可以等效的运送2个40尺集装箱（2*10MW），那么能够输送的容量便是（考虑到来回的话）
created_CBS_capacity=(max(railway_headcount-highspeed_railway_headcount)-(railway_future_headcount-highspeed_railway_future_headcount))*1e4/75/2*2*10/1e6
# ax.plot(year_list_future,created_CBS_capacity)
# ax.plot(year_list_future,np.array([df_transfer_byyear['RenewableIn'].sum() for df_transfer_byyear in df_transfer_byyear_list])-renewable_in_UHV_proposed_list)
coal_capacity=np.array([df_transfer_byyear['RawCoalPurchase'].sum() for df_transfer_byyear in df_transfer_byyear_list])
coal_capacity_max=coal_capacity.copy()
for i in range(1,len(coal_capacity_max)):
    coal_capacity_max[i]=max(coal_capacity_max[i-1],coal_capacity[i])

#接下来就是比较认真地去画这样一张图
#现在的假设是一节车厢15*5=75人，可以换成运2个40尺（2个10MWh）集装箱；40尺煤炭限重26吨，可以换成2个40尺集装箱
#df_CBScapacity=pd.read_excel("../20240219整理文件/6.成本分析-光伏&储能公司.xlsx",sheet_name='1.财务成本参数设置',usecols='J:T',skiprows=[1],nrows=38)
#capacity_increase_rate=np.array(df_CBScapacity['储能初始容量（MWh）'])/np.array(df_CBScapacity['储能初始容量（MWh）'])[0]
capacity_increase_rate=np.ones_like(year_list_future)*10.0
capacity_increase_rate[0]+=np.exp(-1/8)
for i in range(1,len(year_list_future)):
    capacity_increase_rate[i]=capacity_increase_rate[i-1]+np.exp(-(i+1)/8)
capacity_increase_rate/=10.0
import matplotlib.patches as patches
fig,ax=plt.subplots(figsize=(10,5))
coal2power_capacity=(coal_capacity_max-coal_capacity+buycoal_to_buypower)/3.5*capacity_increase_rate
passenger_diversion_capacity=created_CBS_capacity*capacity_increase_rate
doubled_coal2power_capacity=coal2power_capacity*2
doubled_passenger_diversion_capacity=passenger_diversion_capacity*2
ax.bar(year_list_future,coal2power_capacity,color='#B2DF8A',label='Coal-to-Power Capacity')
ax.bar(year_list_future,passenger_diversion_capacity,bottom=coal2power_capacity,color='#FB9A99',label='Passenger-Diversion Capacity')
new_installation=renewable_in_UHV_current_list-renewable_in_UHV_proposed_list-passenger_diversion_capacity-coal2power_capacity
for i in range(1,len(new_installation)):
    new_installation[i]=max(new_installation[i-1],new_installation[i])
ax.bar(year_list_future,new_installation,
        bottom=passenger_diversion_capacity+coal2power_capacity,color='#F1EE8E',label='Accumulated Installation')
ax.plot(year_list_future,renewable_in_UHV_current_list-renewable_in_UHV_proposed_list,label='CBS Demand',marker='.',markersize=10)
ax.legend(fontsize=15)
ax.set_ylabel('Power (TWh)')
ax.set_xlabel('Year')
# ctwoheaded_arrow=patches.FancyArrowPatch((2032,1281-572),(2032,1281),arrowstyle='<|-|>',mutation_scale=10,color='brown')
# ax.annotate('Largest\nGap',xy=(2032.2,950),color='brown')
# ax.add_patch(twoheaded_arrow)
fig.tight_layout()
# fig.savefig('图片/RailwayInstallation.pdf',format='pdf')

#%% 最后我们还要跑一张图 全国电力平衡这一个概念 （感觉要以2023和2060年为基准来说明这件事情）
plt.rcParams['font.size']=18
plt.rcParams['axes.unicode_minus']=False
num_province=31
province_cn2en_dict=dict(zip(province_basicinfo['ProvinceCN'],province_basicinfo.index))

file=r"../数据整理/中国省级地图GS（2019）1719号.geojson"
china_main=gpd.read_file(file).rename(columns={'CNAME':'省份'}).set_index('省份').drop(index=['香港','澳门','台湾'])
china_main['Province']=china_main.index.map(province_cn2en_dict)
china_main=china_main.reset_index().set_index('Province').loc[province_list]
china_main_crs=china_main.geometry.to_crs(epsg=2343).to_frame()
china_main_crs=china_main_crs.merge(province_basicinfo[['Lat','Lon','Region']],left_index=True,right_index=True)
province_coordinates=gpd.GeoDataFrame(china_main[['lng','lat']], geometry=gpd.points_from_xy(china_main['lng'],china_main['lat'],crs='EPSG:4326')) #EPSG就是经典的（经度，纬度）表示法
province_coordinates_converted=province_coordinates.geometry.to_crs(epsg=2343) # 替换成epsg中经典的表示法

provincial_distance=pd.read_excel(excel_name,sheet_name='3.ProvincialDistance',usecols='A:AF',index_col=0)
# 那到底怎么传输呢？找到2023年的最优解
base_year=2022
curr_year=2040
# 找到那个找最优解的程序
model=gp.Model("NetworkFlow")
model.Params.LogToConsole=0
# 考察以CBS方式运送煤炭的数量及最优化方法
df_transfer_curryear=df_transfer_byyear_list[curr_year-base_year-1]
province_out=list(df_transfer_curryear[df_transfer_curryear['RenewableOut']-df_transfer_curryear['RenewableOut(UHV)']<0].index)
province_in=list(df_transfer_curryear[df_transfer_curryear['RenewableIn']-df_transfer_curryear['RenewableIn(UHV)']>0].index)
province_combination=[(i,j) for i in province_out for j in province_in]

province_flow=model.addVars(province_combination,lb=0.0,vtype=GRB.CONTINUOUS)
model.addConstrs((province_flow.sum(i,"*")==-(df_transfer_curryear.loc[i,'RenewableOut']-df_transfer_curryear.loc[i,'RenewableOut(UHV)']) for i in province_out),"FlowOut")
model.addConstrs((province_flow.sum("*",j)==(df_transfer_curryear.loc[j,'RenewableIn']-df_transfer_curryear.loc[j,'RenewableIn(UHV)']) for j in province_in),"FlowIn")
obj=sum(province_flow[(i,j)]*provincial_distance.loc[i,j] for (i,j) in province_combination)
model.setObjective(obj)
model.optimize()
df_result=pd.DataFrame(columns=['From','To','CBSAmount'])
for idx,(i,j) in enumerate(province_combination):
    df_result.loc[idx]=[i,j,province_flow[(i,j)].X]
df_result_positive=df_result[df_result['CBSAmount']>0].reset_index(drop=True)

province_out_UHV=list(df_transfer_curryear[df_transfer_curryear['RenewableOut(UHV)']<0].index)
province_in_UHV=list(df_transfer_curryear[df_transfer_curryear['RenewableIn(UHV)']>0].index)
province_combination_UHV=[(i,j) for i in province_out_UHV for j in province_in_UHV]
model_UHV=gp.Model("NetworkFlow")
model_UHV.Params.LogToConsole=0
province_flow_UHV=model_UHV.addVars(province_combination_UHV,lb=0.0,vtype=GRB.CONTINUOUS)
model_UHV.addConstrs((province_flow_UHV.sum(i,"*")==-df_transfer_curryear.loc[i,'RenewableOut(UHV)'] for i in province_out_UHV),"FlowOut")
model_UHV.addConstrs((province_flow_UHV.sum("*",j)==df_transfer_curryear.loc[j,'RenewableIn(UHV)'] for j in province_in_UHV),"FlowIn")
obj_UHV=sum(province_flow_UHV[(i,j)]*provincial_distance.loc[i,j] for (i,j) in province_combination_UHV)
model_UHV.setObjective(obj_UHV)
model_UHV.optimize()
df_result_UHV=pd.DataFrame(columns=['From','To','UHVAmount'])
for idx,(i,j) in enumerate(province_combination_UHV):
    df_result_UHV.loc[idx]=[i,j,province_flow_UHV[(i,j)].X]
df_result_UHV_positive=df_result_UHV[df_result_UHV['UHVAmount']>0].reset_index(drop=True)

# 接下来就可以开始绘图了
fig,ax=plt.subplots(figsize=(10,8),dpi=100)
china_main_crs.plot('Region',ax=ax,legend_kwds={'label':'Region'},alpha=0.3,legend=False,edgecolor='black')
ax.set_ylim([1.8e6,6.1e6])
excel_handle='../数据整理/生成数据/4.未来发电用电-模版-20231102.xlsx'
ax_pos=ax.get_position()
province_basicinfo['x_rel']=0.0
province_basicinfo['y_rel']=0.0
for province in province_list:
    x,y=province_coordinates_converted[province].x,province_coordinates_converted[province].y
    x_rel=(x-ax.get_xlim()[0])/(ax.get_xlim()[1]-ax.get_xlim()[0])*(ax_pos.x1-ax_pos.x0)+ax_pos.x0 #省会相对于fig的位置（注意有别于相对于ax的位置）
    y_rel=(y-ax.get_ylim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0])*(ax_pos.y1-ax_pos.y0)+ax_pos.y0
    province_basicinfo.loc[province,'x_abs']=x
    province_basicinfo.loc[province,'y_abs']=y
    province_basicinfo.loc[province,'x_rel']=x_rel
    province_basicinfo.loc[province,'y_rel']=y_rel

# 添加一下哪些地区的CBS运输，对应的弧线是下弯的
curvedown_set={('Xinjiang','Guangdong'),('Xinjiang','Qinghai'),('Xinjiang','Chongqing'),('Gansu','Hunan'),('InnerMongolia','Shandong'),('Guizhou','Guangxi'),('Guizhou','Guangdong'),('Shaanxi','Zhejiang'),('Ningxia','Jiangsu'),('Ningxia','Henan')}
for idx_flow in df_result_positive.index:
    province_out=df_result_positive.loc[idx_flow,'From']
    province_out_x=province_basicinfo.loc[province_out,'x_abs']
    province_out_y=province_basicinfo.loc[province_out,'y_abs'] 
    province_in=df_result_positive.loc[idx_flow,'To']
    province_in_x=province_basicinfo.loc[province_in,'x_abs']
    province_in_y=province_basicinfo.loc[province_in,'y_abs']
    if (province_out,province_in) in curvedown_set:
        test_arrow=mpatches.FancyArrowPatch((province_out_x,province_out_y),(province_in_x,province_in_y),arrowstyle='-|>,head_length=3,head_width=2',lw=df_result_positive.loc[idx_flow,'CBSAmount']/30,connectionstyle="arc3,rad=.3",color='green')
    else:
        test_arrow=mpatches.FancyArrowPatch((province_out_x,province_out_y),(province_in_x,province_in_y),arrowstyle='-|>,head_length=3,head_width=2',lw=df_result_positive.loc[idx_flow,'CBSAmount']/30,connectionstyle="arc3,rad=-.2",color='green')
    ax.add_patch(test_arrow)

for idx_flow in df_result_UHV_positive.index:
    province_out=df_result_UHV_positive.loc[idx_flow,'From']
    province_out_x=province_basicinfo.loc[province_out,'x_abs']
    province_out_y=province_basicinfo.loc[province_out,'y_abs'] 
    province_in=df_result_UHV_positive.loc[idx_flow,'To']
    province_in_x=province_basicinfo.loc[province_in,'x_abs']
    province_in_y=province_basicinfo.loc[province_in,'y_abs']
    if (province_out,province_in) in curvedown_set:
        test_arrow=mpatches.FancyArrowPatch((province_out_x,province_out_y),(province_in_x,province_in_y),arrowstyle='-|>,head_length=3,head_width=1',lw=df_result_UHV_positive.loc[idx_flow,'UHVAmount']/30,connectionstyle="arc3,rad=.3",color='black')
    else:
        test_arrow=mpatches.FancyArrowPatch((province_out_x,province_out_y),(province_in_x,province_in_y),arrowstyle='-|>,head_length=3,head_width=1',lw=df_result_UHV_positive.loc[idx_flow,'UHVAmount']/30,connectionstyle="arc3,rad=-.2",color='black')
    ax.add_patch(test_arrow)
fig.tight_layout()
ax.axis('off')
# fig.savefig('图片/CBSNationalTransfer.pdf',format='pdf')
# 是否也可以绘制一个uhv对应的曲线？（但理论上来说不值一提，因为）
#绘制一下各个地区的CBS输出量随年份增长的曲线

color_dict=dict({'Northern':'#dcccc8','Western':'#b9ebf0','Eastern':'#bfe2bf','Southern':'#d8d8d8','Central':'#bbd5e8'})

def darken_color(color, amount=2.5):
    # 如果amount大于1，那么就是darken_color,否则在0到1之间就是lighten_color
    import matplotlib.colors as mc
    import colorsys
    try:
        c=mc.cnames[color]
    except:
        c=color
    c=colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0],1-amount*(1-c[1]),c[2])

plt.rcParams['font.size']=15
region_list=np.array(['Northern','Western','Eastern','Central','Southern'])
region_CBS_dict=color_dict.copy()
for idx_region,region in enumerate(region_list):
    province_related=list(province_basicinfo[province_basicinfo['Region']==region].index)
    region_CBS_dict[idx_region]=[(df_transfer_byyear_list[year_idx].loc[province_related,'RenewableIn'].sum()-df_transfer_byyear_list[year_idx].loc[province_related,'RenewableIn(UHV)'].sum())+(df_transfer_byyear_list[year_idx].loc[province_related,'RenewableOut'].sum()-df_transfer_byyear_list[year_idx].loc[province_related,'RenewableOut(UHV)'].sum()) for year_idx in range(len(year_list_future))]
fig,ax=plt.subplots(5,1,figsize=(3,10))
for idx_region,region in enumerate(region_list):
    ax[idx_region].bar(year_list_future,region_CBS_dict[idx_region],
                label=region,color=color_dict[region])
    if region in ['Northern','Western']:
        ax[idx_region].set_ylim([-2350,0])
    elif region in ['Eastern','Central']:
        ax[idx_region].set_ylim([0,2350])
    else:
        ax[idx_region].set_ylim([-1175,1175])
    [t.set_color(darken_color(color_dict[region])) for t in ax[idx_region].yaxis.get_ticklabels()]
    ax[idx_region].set_title(region,color=darken_color(color_dict[region]))
    if idx_region!=4:
        ax[idx_region].spines[['left','right','top','bottom']].set_visible(False)
        ax[idx_region].tick_params(axis='x',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False)
        #ax[idx_reigon].tick_params(axis='x',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False)
    else:
        ax[idx_region].spines[['left','right','top']].set_visible(False)
        ax[idx_region].tick_params(axis='x',which='both',top=False)
    ax[idx_region].tick_params(color=darken_color(color_dict[region]))
    ax[idx_region].axis('off')
    ax[idx_region].tick_params(axis='y',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False) 
[t.set_color(darken_color(color_dict[region])) for t in ax[len(region_list)-1].xaxis.get_ticklabels()]
fig.tight_layout()
# fig.savefig('图片/CBSByyear.pdf',format='pdf')
    #if region=='Central':ax[idx_region].set_ylim()
    # 上次有信息 具体给哪类省份一个怎样的颜色

#%% UHV调度的过去与现在
# 算一下UHV价格会涨多少倍吧(上面已经实现了proposed，而下面要做)
# 假如是由传统方式运送新能源
# 如何去调度的一个问题
UHV_flow_results_current=[]
UHV_flow_obj_current=[]
UHV_flow_results_proposed=[]
UHV_flow_obj_proposed=[]

#这里我们需要一个base值，也就是2024年的值，来取一个delta
base_year=2024
# df_transfer_byyear=df_transfer_byyear_list[idx_year]
# province_out_UHV_current=list(df_transfer_byyear[df_transfer_byyear['Gen-Load']>0].index)
# province_in_UHV_current=list(df_transfer_byyear[df_transfer_byyear['Gen-Load']<0].index)
# model_UHV_current=gp.Model()
# model_UHV_current.Params.LogToConsole=0
# province_combination_UHV_current=[(i,j) for i in province_out_UHV_current for j in province_in_UHV_current]
# province_flow_UHV_current=model_UHV_current.addVars(province_combination_UHV_current,lb=0.0,vtype=GRB.CONTINUOUS)
# model_UHV_current.addConstrs((province_flow_UHV_current.sum(i,"*")==df_transfer_byyear.loc[i,'Gen-Load'] for i in province_out_UHV_current),"FlowOut")
# model_UHV_current.addConstrs((province_flow_UHV_current.sum("*",j)==-df_transfer_byyear.loc[j,'Gen-Load'] for j in province_in_UHV_current),"FlowIn")
# obj=sum(province_flow_UHV_current[(i,j)]*provincial_distance.loc[i,j] for (i,j) in province_combination_UHV_current)
# model_UHV_current.setObjective(obj)
# model_UHV_current.optimize()
# df_result_UHV_current=pd.DataFrame(columns=['From','To','UHVTransfer'])
# for idx,(i,j) in enumerate(province_combination_UHV_current):
#     df_result_UHV_current.loc[idx]=[i,j,province_flow_UHV_current[(i,j)].X]
# df_result_positive_UHV_current=df_result_UHV_current[df_result_UHV_current['UHVTransfer']>0].reset_index(drop=True)
# UHV_flow_results_current.append(df_result_positive_UHV_current)
# UHV_flow_obj_current.append(obj.getValue())

# province_out_UHV_proposed=list(df_transfer_byyear[df_transfer_byyear['RenewableOut(UHV)']<0].index)
# province_in_UHV_proposed=list(df_transfer_byyear[df_transfer_byyear['RenewableIn(UHV)']>0].index)
# province_combination_UHV_proposed=[(i,j) for i in province_out_UHV_proposed for j in province_in_UHV_proposed]
# model_UHV_proposed=gp.Model("NetworkFlow")
# model_UHV_proposed.Params.LogToConsole=0
# province_flow_UHV_proposed=model_UHV_proposed.addVars(province_combination_UHV_proposed,lb=0.0,vtype=GRB.CONTINUOUS)
# model_UHV_proposed.addConstrs((province_flow_UHV_proposed.sum(i,"*")==-df_transfer_byyear.loc[i,'RenewableOut(UHV)'] for i in province_out_UHV_proposed),"FlowOut")
# model_UHV_proposed.addConstrs((province_flow_UHV_proposed.sum("*",j)==df_transfer_byyear.loc[j,'RenewableIn(UHV)'] for j in province_in_UHV_proposed),"FlowIn")
# obj=sum(province_flow_UHV_proposed[(i,j)]*provincial_distance.loc[i,j] for (i,j) in province_combination_UHV_proposed)
# model_UHV_proposed.setObjective(obj)
# model_UHV_proposed.optimize()
# df_result_UHV_proposed=pd.DataFrame(columns=['From','To','UHVTransfer'])
# for idx,(i,j) in enumerate(province_combination_UHV_proposed):
#     df_result_UHV_proposed.loc[idx]=[i,j,province_flow_UHV_proposed[(i,j)].X]
# df_result_UHV_positive_proposed=df_result_UHV_proposed[df_result_UHV_proposed['UHVTransfer']>0].reset_index(drop=True)
# UHV_flow_results_proposed.append(df_result_UHV_positive_proposed)
# UHV_flow_obj_proposed.append(obj.getValue())

for idx_year,year in enumerate(range(2025,2061)):
    df_transfer_byyear=df_transfer_byyear_list[idx_year]
    province_out_UHV_current=list(df_transfer_byyear[df_transfer_byyear['Gen-Load']>0].index)
    province_in_UHV_current=list(df_transfer_byyear[df_transfer_byyear['Gen-Load']<0].index)
    model_UHV_current=gp.Model()
    model_UHV_current.Params.LogToConsole=0
    province_combination_UHV_current=[(i,j) for i in province_out_UHV_current for j in province_in_UHV_current]
    province_flow_UHV_current=model_UHV_current.addVars(province_combination_UHV_current,lb=0.0,vtype=GRB.CONTINUOUS)
    model_UHV_current.addConstrs((province_flow_UHV_current.sum(i,"*")==df_transfer_byyear.loc[i,'Gen-Load'] for i in province_out_UHV_current),"FlowOut")
    model_UHV_current.addConstrs((province_flow_UHV_current.sum("*",j)==-df_transfer_byyear.loc[j,'Gen-Load'] for j in province_in_UHV_current),"FlowIn")
    obj=sum(province_flow_UHV_current[(i,j)]*provincial_distance.loc[i,j] for (i,j) in province_combination_UHV_current)
    model_UHV_current.setObjective(obj)
    model_UHV_current.optimize()
    df_result_UHV_current=pd.DataFrame(columns=['From','To','UHVTransfer'])
    for idx,(i,j) in enumerate(province_combination_UHV_current):
        df_result_UHV_current.loc[idx]=[i,j,province_flow_UHV_current[(i,j)].X]
    df_result_positive_UHV_current=df_result_UHV_current[df_result_UHV_current['UHVTransfer']>0].reset_index(drop=True)
    UHV_flow_results_current.append(df_result_positive_UHV_current)
    UHV_flow_obj_current.append(obj.getValue())
    
    province_out_UHV_proposed=list(df_transfer_byyear[df_transfer_byyear['RenewableOut(UHV)']<0].index)
    province_in_UHV_proposed=list(df_transfer_byyear[df_transfer_byyear['RenewableIn(UHV)']>0].index)
    province_combination_UHV_proposed=[(i,j) for i in province_out_UHV_proposed for j in province_in_UHV_proposed]
    model_UHV_proposed=gp.Model("NetworkFlow")
    model_UHV_proposed.Params.LogToConsole=0
    province_flow_UHV_proposed=model_UHV_proposed.addVars(province_combination_UHV_proposed,lb=0.0,vtype=GRB.CONTINUOUS)
    model_UHV_proposed.addConstrs((province_flow_UHV_proposed.sum(i,"*")==-df_transfer_byyear.loc[i,'RenewableOut(UHV)'] for i in province_out_UHV_proposed),"FlowOut")
    model_UHV_proposed.addConstrs((province_flow_UHV_proposed.sum("*",j)==df_transfer_byyear.loc[j,'RenewableIn(UHV)'] for j in province_in_UHV_proposed),"FlowIn")
    obj=sum(province_flow_UHV_proposed[(i,j)]*provincial_distance.loc[i,j] for (i,j) in province_combination_UHV_proposed)
    model_UHV_proposed.setObjective(obj)
    model_UHV_proposed.optimize()
    df_result_UHV_proposed=pd.DataFrame(columns=['From','To','UHVTransfer'])
    for idx,(i,j) in enumerate(province_combination_UHV_proposed):
        df_result_UHV_proposed.loc[idx]=[i,j,province_flow_UHV_proposed[(i,j)].X]
    df_result_UHV_positive_proposed=df_result_UHV_proposed[df_result_UHV_proposed['UHVTransfer']>0].reset_index(drop=True)
    UHV_flow_results_proposed.append(df_result_UHV_positive_proposed)
    UHV_flow_obj_proposed.append(obj.getValue())
UHV_flow_obj_current=np.array(UHV_flow_obj_current)
UHV_flow_obj_proposed=np.array(UHV_flow_obj_proposed)
# 根据这个计算出总的计算长度×公里的值（2025年开始）
diffresult=np.array([0.00489418, 0.00449272, 0.00418013,
           0.00389946, 0.00421793, 0.00395127, 0.00269181, 0.00206275,
           0.00151414, 0.00107384, 0.00056918, 0.00014412, 0.00637659,
           0.02179007, 0.02899888, 0.03847811, 0.04402891, 0.04213505,
           0.04373741, 0.04719755, 0.0537482 , 0.05802973, 0.06597334,
           0.07364636, 0.08274553, 0.09251274, 0.10361886, 0.11468862,
           0.12618461, 0.13259439, 0.13844516, 0.13565962, 0.13291611,
           0.13020816, 0.1275366 , 0.12490012])
#max=0.13844,min=0.00014412

(6938.6622259228425, 7.223076631931372)




