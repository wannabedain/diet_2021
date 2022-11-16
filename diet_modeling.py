# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import libpysal as ps
import missingno as msno
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
import geopandas as gp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.font_manager as fm
from matplotlib import font_manager, rc
import seaborn as sns
import warnings
import lightgbm as lgbm

# %%
acc_data = pd.read_csv('./집계구_안전지수_최종.csv',encoding="EUC-KR")
seoul_shp = gp.read_file('./data/서울시 행정구역 및 집계구 정보(SHP)/서울시_집계구_4326.shp',encoding="EUC-KR")
# pop_data = pd.read_csv('./data/pop/19년 12월 집계구코드별 평균생활인구수.csv')

# %%
seoul_shp.to_crs(epsg=4326)
seoul_shp.crs

# %% [markdown]
# ### Column명 변경
# - 지하철(1km -> SUB_NUM
# - 지하철 거리점수 -> SUB_DIS_POINT
# - 버스정류장 -> BUS_NUM
# - 버스승하차 -> BUS_AVG
# - 승하차평균 -> SUB_AVG
# - 구별총생활 -> TOT_POP_GU
# - 구별총생_1 -> TOT_POP_GU_AVG
# - 사고수 -> ACC_NUM
# - 차선수 -> LOAD_NUM
# - 평균속도 -> SPEED_AVG
# - 표준편차 -> SPEED_STD
# - 혼잡수치 -> CONGESTION
# 
# TARGET VARIABLE : ACC_NUM

# %%
acc_data.rename(columns={'지하철(1km':'SUB_NUM','지하철 거리점수':'SUB_DIS_POINT'
                         ,'버스정류장':'BUS_NUM','버스승하차':'BUS_AVG','승하차평균':'SUB_AVG'
                         ,'구별총생활':'TOT_POP_GU','구별총생_1':'TOT_POP_GU_AVG','사고수n':'ACC_NUM'
                        , '차선수':'LOAD_NUM','평균속도':'SPEED_AVG','표준편차':'SPEED_STD'
                        , '혼잡수치':'CONGESTION'},inplace=True)      

# %% [markdown]
# ### 데이터 분포 확인 및 결측치 제거
# #### 1. 지하철 거리 점수

# %%
# standard scaler 적용
sns.histplot(data=acc_data,x='SUB_DIS_POINT',color='blue',kde=True, element='poly')
plt.show()

# %% [markdown]
# #### 2. 버스 승하차 평균
# - 왼쪽으로 치우친 형태의 분포를 띄고 있다. 
# - log함수를 이용하여 skew된 분포를 정규분포형태로 만들어주자.

# %%
sns.histplot(data=acc_data,x='BUS_AVG',element="poly",color='green',kde=True)
plt.show()

# %% [markdown]
# #### 3. 지하철 승하차 평균
# - 왼쪽으로 치우친 분포를 보이고 있다.
# - log함수를 이용하여 정규분포 형태를 만들어주자.

# %%
sns.histplot(data=acc_data,x='SUB_AVG',color='violet',element="poly",kde=True)
plt.show()

# %% [markdown]
# #### 4. 1km 내 지하철 역 수
# %% [markdown]
# - 일단 log 씌워주자

# %%
sns.histplot(data=acc_data,x='SUB_AVG',color='darkred',element="poly",kde=True)
plt.show()

# %% [markdown]
# #### 5. 구별총생활인구수 평균

# %%
sns.histplot(data=acc_data,x='TOT_POP_GU_AVG',color='magenta',element="poly",kde=True)
plt.show()

# %% [markdown]
# - 사고수n, 차선수, 평균속도, 표준편차, 혼잡수치에 결측치가 존재한다.
# - 사고수n : 0으로 대체한다.
# - 차선수 : 1로 대체한다.
# - 평균속도 : 평균속도의 분포를 살펴보고 평균으로 대체할 수 있도록 한다.
# - 표준편차 : 혼잡수치의 산출에 필요한 데이터이므로 column을 제거하도록한다.
# - 혼잡수치 : 혼잡수치의 평균으로 대체할 수 있도록 한다.

# %%
msno.matrix(df=acc_data,color=(0.6,0.1,0.7))
plt.show()

# %%
acc_data.isnull().sum()

# %% [markdown]
# ### 사고 수와 혼잡지수의 교집합 계산

# %%
# 사고 수와 혼잡지수 둘 다 null이 아는 row들만 추출하여 모델을 생성
intersect_data = acc_data[~(acc_data['ACC_NUM'].isna() | acc_data['CONGESTION'].isna())]
prediction_data = acc_data[acc_data['ACC_NUM'].isna() & ~acc_data['CONGESTION'].isna()]
empty_data = acc_data[acc_data['ACC_NUM'].isna() & acc_data['CONGESTION'].isna()]
len(empty_data)

# %%
msno.matrix(df=intersect_data,color=(0.8,0.1,0.3))
plt.show()

# %%
intersect_data.isnull().sum()

# %% [markdown]
# #### 1. 사고수

# %%
avg_acc_data = acc_data['ACC_NUM'].agg(['min','max','mean','std','median'])
# %%
sns.histplot(data=acc_data,x='ACC_NUM',color='indigo',element='poly',kde=True)
plt.show()

# %%
Q1 = acc_data['ACC_NUM'].quantile(.25)
Q2 = acc_data['ACC_NUM'].quantile(.5)
Q3 = acc_data['ACC_NUM'].quantile(.75)
display(Q1)
display(Q2)
display(Q3)

# %% [markdown]
# 일단 2분위수로 대체한다.

# %%
# 2분위수로 대체
acc_data['ACC_NUM'].fillna(Q2,inplace=True)
acc_data['ACC_NUM'].isnull().sum()

# %% [markdown]
# #### 2. 차선수

# %%
load_agg = acc_data['LOAD_NUM'].agg(['min','max','mean','std'])

# %%
sns.histplot(data=acc_data,x='LOAD_NUM',color='orange',element='poly',kde=True)
plt.show()

# %%
# 평균값으로 대체
acc_data['LOAD_NUM'].fillna(load_agg['mean'],inplace=True)
acc_data['LOAD_NUM'].isnull().sum()

# %% [markdown]
# #### 3. 차량 속도

# %%
speed_avg_agg = acc_data['SPEED_AVG'].agg(['min','max','mean','std'])

# %%
# standard scaler
sns.histplot(data=acc_data,x='SPEED_AVG',color='red',element='poly',kde=True)
plt.show()

# %%
acc_data['SPEED_AVG'].fillna(speed_avg_agg['mean'],inplace=True)
acc_data['SPEED_AVG'].isnull().sum()

# %% [markdown]
# #### 4. 혼잡수치

# %%
congestion_agg = acc_data['CONGESTION'].agg(['min','max','mean','std'])

# %%
# standard scaler 적용.
sns.histplot(data=acc_data,x='CONGESTION',color='cadetblue',element='poly',kde=True)
plt.show()

# %% [markdown]
# - 0보다 작은 수들은 0으로 대체한다.
# - 음수값들을 0으로 대체하고 분포를 다시 살펴본 후 null값을 대체하자.

# %%
acc_data.loc[acc_data['CONGESTION'] < 0,'CONGESTION'] = 0
speed_avg_agg = acc_data['CONGESTION'].agg(['min','max','mean','std'])

# %%
intersect_data.loc[intersect_data['CONGESTION']<0,'CONGESTION'] = 0
intersect_data['CONGESTION'].agg(['min','max'])

# %%
acc_data['CONGESTION'].fillna(speed_avg_agg['mean'],inplace=True)
acc_data['CONGESTION'].isnull().sum()

# %%
acc_data.isnull().sum()

# %% [markdown]
# ### Merge shp data and accident data

# %%
# 현재 TOT_REG_CD가 int type이므로 str type으로 변경해주도록 한다.
# 차후 merge를 하기 위해서이다.
intersect_data['TOT_REG_CD']= intersect_data['TOT_REG_CD'].astype(str)

# %%
# prepare dataset
data = intersect_data.merge(seoul_shp,left_on='TOT_REG_CD',right_on='TOT_REG_CD')
data.head()

# %%
essential_col=['TOT_REG_CD','BUS_NUM','BUS_AVG','SUB_NUM','SUB_DIS_POINT','SUB_AVG,'TOT_POP_GU_AVG','LOAD_NUM','SPEED_AVG','CONGESTION','geometry','ACC_NUM']
required_data = data.loc[:,essential_col]
required_data.head()

# %% [markdown]
# ### Data 분포 변환
# %% [markdown]
# #### 1. 버스 이용객 평균

# %%
required_data.loc[:,'BUS_AVG'] = np.log1p(data['BUS_AVG'])
sns.histplot(data=required_data,x='BUS_AVG',color='skyblue',element="poly",kde=True)
plt.show()

# %%
required_data.loc[:,'BUS_AVG'] = np.log1p(data['BUS_AVG'])
sns.histplot(data=required_data,x='BUS_AVG',element="poly",color='green',kde=True)
plt.show()

# %% [markdown]
# #### 2. 지하철 이용객 평균

# %%
required_data.loc[:,'SUB_AVG'] = np.log1p(data['SUB_AVG'])
sns.histplot(data=required_data,x='SUB_AVG',color='violet',element="poly",kde=True)
plt.show()

# %% [markdown]
# #### 3. 구별 총 생활인구 평균

# %%
required_data.loc[:,'TOT_POP_GU_AVG'] = np.log1p(data['TOT_POP_GU_AVG'])
sns.histplot(data=required_data,x='TOT_POP_GU_AVG',color='magenta',element="poly",kde=True)
plt.show()

# %% [markdown]
# #### 4. 지하철 거리 점수

# %%
sub_dis_point = required_data.loc[:,'SUB_DIS_POINT'].values.reshape((-1,1))
required_data.loc[:,'SUB_DIS_POINT'] = pd.Series(sd_scaler.fit_transform(sub_dis_point).flatten())
sns.histplot(data=required_data,x='SUB_DIS_POINT',color='magenta',element="poly",kde=True)
plt.show()

# %% [markdown]
# #### 5. 속도 평균

# %%
sub_dis_point = required_data.loc[:,'SPEED_AVG'].values.reshape((-1,1))
required_data.loc[:,'SPEED_AVG'] = pd.Series(sd_scaler.fit_transform(sub_dis_point).flatten())
sns.histplot(data=required_data,x='SPEED_AVG',color='magenta',element="poly",kde=True)
plt.show()

# %% [markdown]
# #### 6. 혼잡지수

# %%
sub_dis_point = required_data.loc[:,'CONGESTION'].values.reshape((-1,1))
required_data.loc[:,'CONGESTION'] = pd.Series(sd_scaler.fit_transform(sub_dis_point).flatten())
sns.histplot(data=required_data,x='CONGESTION',color='orange',element="poly",kde=True)
plt.show()

# %%
center = seoul_shp.centroid
X = pd.Series(center.x)
Y = pd.Series(center.y)
required_data.loc[:,'lon'] = X
required_data.loc[:,'lat'] = Y

# %% [markdown]
# ### 독립변수, 종속변수 설정

# %%
#Prepare Georgia dataset inputs
s_y = required_data['ACC_NUM'].values.reshape((-1,1))
s_X = required_data[essential_col[1:-2]].values
u = required_data['lon']
v = required_data['lat']
s_coords = list(zip(u,v))

# %% [markdown]
# ### Modeling

# %%
#Calibrate GWR model
gwr_selector = Sel_BW(s_coords, s_y, s_X,kernel='gaussian')
gwr_bw = gwr_selector.search(bw_min=2)
print(gwr_bw)
gwr_model = GWR(s_coords, s_y, s_X, gwr_bw,kernel='gaussian')
gwr_results = gwr_model.fit()

# %%
gwr_results.params[0:5]

# %%
gwr_results.localR2[0:10]

# %% [markdown]
# ### 모델링 결과 확인

# %%
gwr_results.summary()

# %% [markdown]
# ### 예측 모델생성

# %%
#Prepare Georgia dataset inputs
col = essential_col[1:-2]
col.append('lon')
col.append('lat')

s_y = required_data['ACC_NUM']
s_X = required_data[col]

x_train, x_valid, y_train, y_valid = train_test_split(s_X, s_y, test_size=0.2, shuffle=True,random_state=34)

cal_u = x_train['lon'].values
cal_v = x_train['lat'].values
cal_coords = list(zip(cal_u,cal_v))

pred_u = x_valid['lon'].values
pred_v = x_valid['lat'].values
pred_coords = list(zip(pred_u,pred_v))

# 위도, 경도 column 제거
# display(x_train.columns)
x_train.drop(['lon','lat'],axis=1,inplace=True)
x_valid.drop(['lon','lat'],axis=1,inplace=True)

# array로 변환
X_train = x_train.values
Y_train = y_train.values.reshape((-1,1))
X_valid = x_valid.values
Y_valid = y_valid.values.reshape((-1,1))
# %%
#Calibrate GWR model
gwr_selector = Sel_BW(cal_coords, Y_train, X_train,kernel='gaussian')
gwr_bw = gwr_selector.search(bw_min=2)
print(gwr_bw)
gwr_model = GWR(cal_coords, Y_train, X_train, gwr_bw,kernel='gaussian')
gwr_results = gwr_model.fit()

# %%
scale = gwr_results.scale
residuals = gwr_results.resid_response
# 
display(type(pred_coords))
# test data로 예측을 해보고 결과 저장
pred_results = gwr_model.predict(np.array(pred_coords), X_valid, scale, residuals)

# %%
np.corrcoef(pred_results.predictions.flatten(), Y_valid.flatten())[0][1]

# %% [markdown]
# ### 사고수 예측

# %%
prediction_data.loc[prediction_data['CONGESTION']<0,'CONGESTION'] = 0
prediction_data['CONGESTION'].agg(['min','max'])

# %%
# 현재 TOT_REG_CD가 int type이므로 str type으로 변경해주도록 한다.
# 차후 merge를 하기 위해서이다.
prediction_data['TOT_REG_CD']= prediction_data['TOT_REG_CD'].astype(str)

# %%
# prepare dataset
p_data = prediction_data.merge(seoul_shp,left_on='TOT_REG_CD',right_on='TOT_REG_CD')

# %%
essential_col=['TOT_REG_CD','BUS_NUM','BUS_AVG','SUB_NUM','SUB_DIS_POINT','SUB_AVG'
               ,'TOT_POP_GU_AVG','LOAD_NUM','SPEED_AVG','CONGESTION','geometry','ACC_NUM']
p_required_data = p_data.loc[:,essential_col]

# %%
center = seoul_shp.centroid
X = pd.Series(center.x)
Y = pd.Series(center.y)
p_required_data.loc[:,'lon'] = X
p_required_data.loc[:,'lat'] = Y
p_required_data.head()

# %%
s_y = p_required_data['ACC_NUM'].values.reshape((-1,1))
s_X = p_required_data[essential_col[1:-2]].values
u = p_required_data['lon']
v = p_required_data['lat']
s_coords = list(zip(u,v))

# %%
scale = gwr_results.scale
residuals = gwr_results.resid_response

display(scale)
display(residuals)
pred_results = gwr_model.predict(np.array(s_coords), s_X, scale, residuals)

# %%
pred_results.predictions

보행수요지수
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import libpysal as ps
import missingno as msno
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
import geopandas as gp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.font_manager as fm
from matplotlib import font_manager, rc
import seaborn as sns
import warnings
from mgwr.utils import shift_colormap, truncate_colormap

from sklearn.model_selection import KFold, StratifiedKFold

# %%
ped_data = pd.read_csv('./서울시_집계구_보행수요_최종.csv',encoding="EUC-KR")
road_data = pd.read_csv('./서울시_도로_보행수요_최종.csv',encoding="EUC-KR")
seoul_shp = gp.read_file('./data/서울시 행정구역 및 집계구 정보(SHP)/서울시_집계구_4326.shp',encoding="EUC-KR")

# %%
ped_data.info()

# %%
road_data.info()

# %%
seoul_shp.info()

# %%
seoul_shp.to_crs(epsg=4326)
seoul_shp.crs

# %% [markdown]
# ### Column명 변경
# - 사회복지시 -> SOC_NUM
# - 지하철(1km이내) -> SUB_NUM
# - 지하철거리점수 -> SUB_DIS_POINT
# - 버스정류장수 -> BUS_NUM
# - 버스승하차 -> BUS_AVG
# - 지하철승하차 -> SUB_AVG
# - 구별총생활 -> TOT_POP_GU
# - 구별총생활인구 -> TOT_POP_GU_AVG
# - 상가수 -> SHOP_NUM
# - 어린이집개수 -> CHILD_NUM
# - 면적 -> AREA
# 
# TARGET VARIABLE : TOT_POP_GU_AVG

# %%
ped_data.rename(columns={'사회복지시':'SOC_NUM','지하철(1km이내)':'SUB_NUM','지하철거리점수':'SUB_DIS_POINT'
                         ,'버스정류장수':'BUS_NUM','버스승하차':'BUS_AVG','지하철승하차':'SUB_AVG'
                         ,'구별총생활':'TOT_POP_GU','구별총생활인구':'TOT_POP_GU_AVG','상가수':'SHOP_NUM'
                        , '어린이집개수':'CHILD_NUM','면적':'AREA'}
                        ,inplace=True)      
road_data.rename(columns={'사회복지시설':'SOC_NUM','지하철수':'SUB_NUM','지하철거리점수':'SUB_DIS_POINT'
                         ,'버스정류장수':'BUS_NUM','버스승하차':'BUS_AVG','지하철승하차':'SUB_AVG'
                         ,'구별총생활':'TOT_POP_GU','구별총생활인구':'TOT_POP_GU_AVG','상권정보':'SHOP_NUM'
                        , '어린이집':'CHILD_NUM','시설수':'FACILITY_NUM'}
                        ,inplace=True) 
display(ped_data.columns)
display(road_data.columns)

# %% [markdown]
# ### 결측치 확인 및 제거
# - 상점의 수에 nan이 존재하여 0으로 대체

# %%
msno.matrix(df=ped_data,color=(0.6,0.1,0.7))
plt.show()

# %%
ped_data.isnull().sum()

# %%
msno.matrix(df=road_data,color=(0.6,0.8,0.7))
plt.show()

# %%
road_data.isnull().sum()

# %% [markdown]
# #### 1. 지하철 거리 점수
# - 0으로 대체

# %%
sub_dis_point_agg = ped_data['SUB_DIS_POINT'].agg(['mean'])
sub_dis_point_agg

# %% [markdown]
# sns.histplot(data=ped_data,x='SUB_DIS_POINT',color='blue',kde=True,element='poly')
# plt.show()

# %%
ped_data['SUB_DIS_POINT'].fillna(0,inplace=True)
ped_data['SUB_DIS_POINT'].isnull().sum()

# %% [markdown]
# #### 2. 상점수

# %%
ped_data['SHOP_NUM'].fillna(0,inplace=True)
ped_data['SHOP_NUM'].isnull().sum()

# %% [markdown]
# #### 3. 어린이집 개수

# %%
ped_data['CHILD_NUM'].fillna(0,inplace=True)
ped_data['SHOP_NUM'].isnull().sum()

# %%
ped_data.isnull().sum()

# %% [markdown]
# %%
ped_data['FACILITY_NUM'] = (ped_data['공공기관']+ped_data['보육시설']+ped_data['CHILD_NUM']+ped_data['SOC_NUM'])

# %%
ped_data.head()

# %%
# 현재 TOT_REG_CD가 int type이므로 str type으로 변경해주도록 한다.
# 차후 merge를 하기 위해서이다.
ped_data['TOT_REG_CD']= ped_data['TOT_REG_CD'].astype(str)

# %%
# prepare dataset
data = ped_data.merge(seoul_shp,left_on='TOT_REG_CD',right_on='TOT_REG_CD')
data.head()

# %%
essential_col=['TOT_REG_CD','BUS_NUM','BUS_AVG','SUB_NUM','SUB_DIS_POINT','SUB_AVG','SHOP_NUM','FACILITY_NUM','geometry','TOT_POP_GU_AVG']
required_data = data.loc[:,essential_col]
required_data.head()

# %% [markdown]
# ### 데이터 분포 확인
# #### 1. 지하철 거리
# %% [markdown]
# #### 2. 버스 승하차
# - 왼쪽으로 치우친 형태의 분포를 띄고 있다. 
# - log함수를 이용하여 skew된 분포를 정규분포형태로 만들어주자.

# %%
sns.histplot(data=ped_data,x='BUS_AVG',element="poly",color='green',kde=True)
plt.show()

# %% [markdown]
# #### 3. 지하철 평균

# %%
sns.histplot(data=ped_data,x='SUB_AVG',color='violet',element="poly",kde=True)
plt.show()

# %%
# 현재 TOT_REG_CD가 int type이므로 str type으로 변경해주도록 한다.
# 차후 merge를 하기 위해서이다.
ped_data['TOT_REG_CD']= ped_data['TOT_REG_CD'].astype(str)

# %%
# prepare dataset
data = ped_data.merge(seoul_shp,left_on='TOT_REG_CD',right_on='TOT_REG_CD')
data.head()

# %%
data.columns

# %%
essential_col=['TOT_REG_CD','BUS_NUM','BUS_AVG','SUB_NUM','SUB_DIS_POINT','SUB_AVG'
               ,'SHOP_NUM','FACILITY_NUM','geometry','TOT_POP_GU_AVG']
required_data = data.loc[:,essential_col]
required_data.head()

# %%
center = seoul_shp.centroid
X = pd.Series(center.x)
Y = pd.Series(center.y)
required_data.loc[:,'lon'] = X
required_data.loc[:,'lat'] = Y
required_data.head()

# %% [markdown]
# ### 독립변수, 종속변수 설정
# %%
#Prepare Georgia dataset inputs
s_y = required_data['TOT_POP_GU_AVG'].values.reshape((-1,1))
s_X = required_data[essential_col[1:-2]].values
u = required_data['lon']
v = required_data['lat']
s_coords = list(zip(u,v))

# p_y = road_data['TOT_POP_GU_AVG'].values.reshape((-1,1))
p_X = road_data[essential_col[1:-2]]
p_u = road_data['X']
p_y = road_data['Y']
p_coords = list(zip(p_u,p_y))

# %% [markdown]
# ### Modeling

# %%
#Calibrate GWR model
gwr_selector = Sel_BW(s_coords, s_y, s_X,kernel='gaussian')
gwr_bw = gwr_selector.search(bw_min=2)
print(gwr_bw)
gwr_model = GWR(s_coords, s_y, s_X, gwr_bw,kernel='gaussian')
gwr_results = gwr_model.fit()

# %%
gwr_results.params[0:5]

# %%
gwr_results.localR2[0:10]

# %%
gwr_results.summary()

# %%
result = np.array([])
div_num = int(len(road_data)/len(s_X))+1
batch_size = int(len(road_data) / div_num)+1

for idx in range(0,div_num):
    scale = gwr_results.scale
    residuals = gwr_results.resid_response
    # test data로 예측을 해보고 결과 저장
    start_idx = batch_size*idx
    if batch_size*(idx+1) > len(road_data):
        end_idx = len(road_data)
    else:
        end_idx = batch_size*(idx+1)
        
    local_p_coords = p_coords[start_idx:end_idx]
    local_p_X = p_X.iloc[start_idx:end_idx]
    pred_results = gwr_model.predict(np.array(local_p_coords), local_p_X, scale, residuals)
    display(start_idx,end_idx)
    result = np.append(result,pred_results.predictions.flatten())
#     display(result)
#     display(len(result))

result = result.reshape((-1,1))

# %%
len(result[result<0])
print('0보다 작은 예측값의 비율 : {:.3f}%'.format(len(result[result<0])/len(road_data)*100))

# %%
road_data['EXPECTED_POP_AVG']= pd.Series(result.flatten())
road_data.head()
# %%
road_data.loc[road_data['EXPECTED_POP_AVG'] < 0,'EXPECTED_POP_AVG'] = 0
road_data['EXPECTED_POP_AVG'].agg(['min'])

# %%
road_data.to_csv('예상 도로 생활 인구.csv')

# %% [markdown]
# ### 예측모델 생성

# %%
col = essential_col[1:-2]
col.append('lon')
col.append('lat')
display(col)
s_y = required_data['TOT_POP_GU_AVG']
s_X = required_data[col]

# test set은 20%, 
x_train, x_valid, y_train, y_valid = train_test_split(s_X, s_y, test_size=0.2, shuffle=True,random_state=34)

cal_u = x_train['lon'].values
cal_v = x_train['lat'].values
cal_coords = list(zip(cal_u,cal_v))

pred_u = x_valid['lon'].values
pred_v = x_valid['lat'].values
pred_coords = list(zip(pred_u,pred_v))

# 위도, 경도 column 제거
# display(x_train.columns)
x_train.drop(['lon','lat'],axis=1,inplace=True)
x_valid.drop(['lon','lat'],axis=1,inplace=True)

# array로 변환
X_train = x_train.values
Y_train = y_train.values.reshape((-1,1))
X_valid = x_valid.values
Y_valid = y_valid.values.reshape((-1,1))

# %%
#Calibrate GWR model
p_gwr_selector = Sel_BW(cal_coords, Y_train, X_train,kernel='gaussian')
p_gwr_bw = p_gwr_selector.search(bw_min=2)
print(p_gwr_bw)
p_gwr_model = GWR(cal_coords, Y_train, X_train, p_gwr_bw,kernel='gaussian')
p_gwr_results = p_gwr_model.fit(
# %%
p_gwr_results.summary()

# %%
p_scale = p_gwr_results.scale
p_residuals = p_gwr_results.resid_response
# 
display(type(pred_coords))
# test data로 예측을 해보고 결과 저장
pred_results = p_gwr_model.predict(np.array(pred_coords), X_valid, p_scale, p_residuals)

# %%
np.corrcoef(pred_results.predictions.flatten(), Y_valid.flatten())[0][1]

# %%
import math

# %%
MAE = abs(Y_valid.flatten() - pred_results.predictions.flatten()).mean()
MSE = math.sqrt(((Y_valid.flatten() - pred_results.predictions.flatten())**2).mean())
MAPE = (abs(Y_valid.flatten() - pred_results.predictions.flatten())/Y_valid.flatten()*100).mean()
pd.DataFrame([MAE, MSE, MAPE], index=['MAE', 'MSE', 'MAPE'], columns=['Score']).T

# %%
resi = Y_valid.flatten() - pred_results.predictions.flatten()
resi = pd.DataFrame(resi,columns={'result'})

# %
sns.histplot(data=resi,x='result',color='lightblue',element="poly",kde=True)
plt.show()

# %%
import scipy.stats as stats

# %%
F_statistic, pVal = stats.f_oneway(pred_results.predictions.flatten(),Y_valid.flatten())

print('Altman 910 데이터의 일원분산분석 결과 : F={0:.1f}, p={1:.5f}'.format(F_statistic, pVal))
if pVal < 0.05:
    print('P-value 값이 충분히 작음으로 인해 그룹의 평균값이 통계적으로 유의미하게 차이납니다.')

# %% [markdown]
# ### Parameter 시각화

# %%
# 모델 만들때 쓰인 data를 이용할 것. required data
display(len(required_data))
display(len(gwr_results.params))
# sample_data에 서울의 polygon이 들어있으므로 이를 활용하여 시각화하자.
visualization_data = required_data.copy()
visualization_data.head()

# %%
#Prepare GWR results for mapping

#Add GWR parameters to GeoDataframe
'''
1. 버스정류장수
2. 버스이용객 평균
3. 지하철 수
4. 지하철 거리점수
5. 지하철이용객 평균
6. 상가 수
7. 시설 수
8. 위도
9. 경도
'''
visualization_data['gwr_intercept'] = gwr_results.params[:,0]
visualization_data['gwr_bus_num'] = gwr_results.params[:,1]
visualization_data['gwr_bus_avg'] = gwr_results.params[:,2]
visualization_data['gwr_sub_num'] = gwr_results.params[:,3]
visualization_data['gwr_sub_dis_point'] = gwr_results.params[:,4]
visualization_data['gwr_sub_avg'] = gwr_results.params[:,5]
visualization_data['gwr_shop_num'] = gwr_results.params[:,6]
visualization_data['gwr_facility_num'] = gwr_results.params[:,7]

#Obtain t-vals filtered based on multiple testing correction
gwr_filtered_t = gwr_results.filter_tvals()

# %%
columns = ['geometry','gwr_intercept','gwr_bus_num','gwr_bus_avg','gwr_sub_num','gwr_sub_dis_point','gwr_sub_avg'
           ,'gwr_shop_num','gwr_facility_num']
s_geo_data = visualization_data.loc[:,columns]
s_geo_data = gp.GeoDataFrame(s_geo_data,geometry='geometry')
# %%
#Comparison maps of GWR vs. MGWR parameter surfaces where the grey units pertain to statistically insignificant parameters

#Prep plot and add axes
fig, ax = plt.subplots(1, figsize=(30,20))
ax.set_title('GWR Intercept Surface (BW: ' + str(gwr_bw) +')', fontsize=40)

#Set color map
cmap = plt.cm.seismic

#Find min and max values of the two combined datasets
vmin = s_geo_data['gwr_intercept'].min()
vmax = s_geo_data['gwr_intercept'].max()

#If all values are negative use the negative half of the colormap
if (vmin < 0) & (vmax < 0):
    cmap = truncate_colormap(cmap, 0.0, 0.5)
#If all values are positive use the positive half of the colormap
elif (vmin > 0) & (vmax > 0):
    cmap = truncate_colormap(cmap, 0.5, 1.0)
#Otherwise, there are positive and negative values so the colormap so zero is the midpoint
else:
    cmap = shift_colormap(cmap, start=0.0, midpoint=1 - vmax/(vmax + abs(vmin)), stop=1.)

#Create scalar mappable for colorbar and stretch colormap across range of data values
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

#Plot GWR parameters
s_geo_data.plot('gwr_intercept', cmap=sm.cmap, ax=ax, vmin=vmin, vmax=vmax, **{'edgecolor':'black', 'alpha':.65})

#Set figure options and plot 
fig.tight_layout()    
fig.subplots_adjust(right=0.9)
cax = fig.add_axes([0.92, 0.14, 0.03, 0.75])
sm._A = []
cbar = fig.colorbar(sm, cax=cax)
cbar.ax.tick_params(labelsize=50) 
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()

# %%
정류장별 승객수 및 집계구 결합
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import geopandas as gpd
import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
import math
import missingno as msno
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as qsns
from matplotlib import font_manager, rc

# %% [markdown]
# ## 버스 정류소 승하차 데이터 및 집계구 데이터 결합

# %%
bus_stop_500m = pd.read_csv('./data/전처리 필요/버스정류소_500m_집계구.csv',encoding="EUC-KR")
bus_stop_ctm = pd.read_csv('./bus_stop_ctm_with_pos.csv')

# %%
bus_stop_ctm = bus_stop_ctm.drop(columns={'Unnamed: 0','X좌표','Y좌표'},axis=1)
# %%
bus_stop_data = bus_stop_500m.merge(bus_stop_ctm,left_on='표준ID',right_on='표준ID')
bus_stop_data.head()

bus_stop_data[['TOT_REG_CD','ADM_CD']] = bus_stop_data[['TOT_REG_CD','ADM_CD']].astype(str)

# %%
bus_stop_data_left = bus_stop_500m.merge(bus_stop_ctm,how='left',left_on='표준ID',right_on='표준ID')

# %%
# 집계구 데이터에는 데이터가 있는데, 승하차 데이터는 없는 경우
msno.matrix(df=bus_stop_data_left,color=(0.6,0.1,0.7))
plt.show()

# %%
bus_stop_data_right = bus_stop_500m.merge(bus_stop_ctm,how='right',left_on='표준ID',right_on='표준ID')

msno.matrix(df=bus_stop_data_right,color=(0.6,0.7,0.7))
plt.show()

# %%
bus_stop_data.drop('역명',axis=1,inplace=True)

# %%
bus_stop_count = bus_stop_data.groupby('TOT_REG_CD').size()
count = list()

for i, row in bus_stop_data.iterrows():
    tot_reg_cd = row.loc['TOT_REG_CD'] # 집계구 코드를 가져와서 저장.
    count.append(bus_stop_count.loc[tot_reg_cd])

bus_stop_data['정류장수'] = pd.Series(count)

# %%
bus_stop_ride = bus_stop_data.groupby('TOT_REG_CD')['승차'].sum()
bus_stop_get_off = bus_stop_data.groupby('TOT_REG_CD')['하차'].sum()
count_ride = list()
count_get_off = list()

for i, row in bus_stop_data.iterrows():
    tot_reg_cd = row.loc['TOT_REG_CD'] # 집계구 코드를 가져와서 저장.
    count_ride.append(bus_stop_ride.loc[tot_reg_cd]/row['정류장수']/31)
    count_get_off.append(bus_stop_get_off.loc[tot_reg_cd]/row['정류장수']/31)

bus_stop_data['평균승차수'] = pd.Series(count_ride)
bus_stop_data['평균하차수'] = pd.Series(count_get_off)

# %%
bus_stop_data.to_csv('집계구별 정류장수 및 승하차 데이터.csv')


버스정류장별 이용객 수 집계(월평균)
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### 버스정류장 위치 확인

# %%
import pandas as pd
import missingno as msno
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import font_manager, rc

# %%
bus_stop= gpd.read_file('./서울시_버스정류소_좌표데이터/서울시_버스정류소_좌표데이터.shp',encoding="EUC-KR")
bus_stop.head()

# %%
msno.matrix(df=bus_stop,color=(0.6,0.1,0.7))
plt.show()

# %%
bus_stop.crs

# %%
import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
import math

# %%
# 버스 정류장 좌표 지도
# Create a map
m_3 = folium.Map(location=[37.5665,126.9780], tiles='cartodbpositron', zoom_start=13)

# Add points to the map
mc = MarkerCluster()
for idx, row in bus_stop.iterrows():
    if not math.isnan(row['Y좌표']) and not math.isnan(row['X좌표']):
        mc.add_child(Marker([row['Y좌표'], row['X좌표']],tooltip=row['정류소명']))
m_3.add_child(mc)

# Display the map
m_3

# %% [markdown]
# ### 버스 정류장 이용객 수

# %%
bus_stop_ctm = pd.read_csv('./data/bus/2019년 버스노선별 정류장별 시간대별 승하차 인원 정보1.csv',encoding='EUC-KR')
bus_stop_ctm.head()
# %%
bus_stop_ctm['사용년월'].unique()

# %%
len(bus_stop['표준ID'].unique())

# %%
bus_stop_ctm = bus_stop_ctm[bus_stop_ctm['사용년월'] == 201912]
len(bus_stop_ctm)

# %%
display(len(bus_stop_ctm['표준버스정류장ID'].unique()))
not_n_ars = bus_stop_ctm[bus_stop_ctm['버스정류장ARS번호'] != '~']['버스정류장ARS번호'].unique()
display(len(not_n_ars))
not_b_bus= bus_stop_ctm[bus_stop_ctm['버스정류장ARS번호'] != '~']['표준버스정류장ID'].unique()
display(len(not_b_bus))

# %% [markdown]
# 각 정류장의 이용객 수를 모두 더한 결과를 출력하도록 한다.(사용년월 기준 202101)

# %%
sample = bus_stop_ctm.groupby(['표준버스정류장ID','역명']).agg('sum').reset_index()
sample.head()

# %% [markdown]
# 승차, 하차 기준으로 승객수를 모두 합하자.<br>
# 이후 합한 데이터를 heatmap으로 표시할 수 있도록 하자.

# %%
get_off = ['시하차총승객수' in col for col in list(sample)]
ride = ['시승차총승객수' in col2 for col2 in list(sample)]
sample['하차'] = sample.iloc[:,get_off].sum(axis=1)
sample['승차'] = sample.iloc[:,ride].sum(axis=1)

# %%
sample = sample.rename(columns={'표준버스정류장ID':'표준ID'})

# %%
bs = set(bus_stop['표준ID'])
sam = set(sample['표준ID'])
only_bs = bs-sam
only_sam = sam-bs

# %% [markdown]
# ## 정류소 좌표데이터와 승하차 데이터 결합

# %%
# 정류장 좌표데이터는 있으나 승하차 데이터가 없음
# 차후에 채워야할 필요가 있음.

m_bus_stop1 = bus_stop.merge(sample,how='left',left_on='표준ID',right_on='표준ID').loc[:,['표준ID','역명','X좌표','Y좌표','승차','하차']]

# %%
msno.matrix(df=m_bus_stop1,color=(0.6,0.1,0.9))
plt.show()

# %%
m_bus_stop2 = bus_stop.merge(sample,how='right',left_on='표준ID',right_on='표준ID').loc[:,['표준ID','역명','X좌표','Y좌표','승차','하차']]

# %%
# 승객수 데이터는 있는데 정류장 좌표가 없음.
# 차후에 채워야할 필요가 있음.
msno.matrix(df=m_bus_stop2,color=(0.6,0.1,0.1))
plt.show()

# %%
m_bus_stop_not_exist_pos = m_bus_stop2[m_bus_stop2['표준ID'].isin(only_sam)]
display(m_bus_stop_not_exist_pos)
m_bus_stop_not_exist_pos.to_excel('have_to_get_pos.xlsx')

# %%
not_virtual = list()
m_bus_stop_not_exist_pos = m_bus_stop2[m_bus_stop2['표준ID'].isin(only_sam)]
for i,row in m_bus_stop_not_exist_pos.iterrows():
    if '가상' not in row['역명']:
        not_virtual.append(row['역명'])

# %%
# inner join

m_bus_stop = bus_stop.merge(sample,left_on='표준ID',right_on='표준ID').loc[:,['표준ID','역명','X좌표','Y좌표','승차','하차']]
m_bus_stop.head()

# %%
m_bus_stop['월평균승차수'] = m_bus_stop['승차']/31
m_bus_stop['월평균하차수'] = m_bus_stop['하차']/31
m_bus_stop.head()

# %%
m_bus_stop[['승차','하차']].agg(['min','max','mean','std'])

# %%
m_bus_stop.to_csv('bus_stop_ctm_with_pos.csv')

# Create a base map
m_4 = folium.Map(location=[37.5665,126.9780], tiles='cartodbpositron', zoom_start=13)

def color_producer(val):
    if val <= 97700:
        return 'red'
    else:
        return 'blue'

# Add a bubble map to the base map
for i in range(0,len(m_bus_stop)):
    Circle(
        location=[m_bus_stop.iloc[i]['Y좌표'], m_bus_stop.iloc[i]['X좌표']],
        radius=5,
        color=color_producer(m_bus_stop.iloc[i]['승차']),
        tooltip=str(m_bus_stop.iloc[i]['승차'])+'명',
        popup=m_bus_stop.iloc[i]['역명']).add_to(m_4)

# Display the map
m_4

# %%
# Create a base map
# 승차

m_5 = folium.Map(location=[37.5665,126.9780], tiles='cartodbpositron', zoom_start=13)

mc = MarkerCluster()
for idx, row in m_bus_stop.iterrows():
    if not math.isnan(row['Y좌표']) and not math.isnan(row['X좌표']):
        mc.add_child(Marker([row['Y좌표'], row['X좌표']],popup=row['역명'],tooltip=row['승차']))
m_5.add_child(mc)

# Add a heatmap to the base map
HeatMap(data=m_bus_stop[['Y좌표', 'X좌표']], radius=10).add_to(m_5)

# Display the map
m_5

보행자 사고 수 집계 및 시각화
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import geopandas as gpd
import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
import math
import missingno as msno
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager, rc

# %%
pd_acc_data = pd.read_csv('./data/보행자사고/accident_17_19.csv')
pd_acc_data.head()

# %%
pd_acc_data['피해운전자 연령'].unique()

# %% [markdown]
# 아동 : 13세 이하<br>
# 노인 : 65세 이상<br>
# 필터링해서 객체에 저장해보자.<br>
# 일단 '세'라는 단어를 없애자. 이후 int type으로 변경

# %%
for idx, age in pd_acc_data.iterrows():
    if '세 이상' in age['피해운전자 연령']:
        pd_acc_data.loc[idx]['피해운전자 연령'] = age.replace('세 이상','')
    elif '세' in age:
        pd_acc_data.loc[idx]['피해운전자 연령'] = age.replace('세','')

pd_acc_data['피해운전자 연령'].unique()

# %%
pd_acc_data = pd_acc_data[pd_acc_data['피해운전자 연령']!='미분류']

# %%
pd_acc_data[['사망자수','중상자수','경상자수','부상신고자수']]

# %%
pd_acc_data['총사고자수'] = pd_acc_data[['사망자수','중상자수','경상자수','부상신고자수']].sum(axis=1)

# %%
pd_acc_data['총사고자수'].unique()

# %%
pd_acc_data.drop(columns={'총사고수','총사망자수'},inplace=True)

# %%
pd_acc_data[['사망자수','중상자수','경상자수','부상신고자수']].agg(['min','max'])

# %%
pd_acc_data['피해운전자 연령'] = pd_acc_data['피해운전자 연령'].astype(int)
pd_acc_data.info()

# %%
pd_acc_data_child = pd_acc_data[pd_acc_data['피해운전자 연령'] <= 13]
pd_acc_data_older = pd_acc_data[pd_acc_data['피해운전자 연령'] >= 65]

# %%
pd_acc_data['피해운전자 연령'].agg(['min','max','mean','std'])

# %%
sns.histplot(data=pd_acc_data,x='피해운전자 연령')
plt.show()

# %%
# 어린이 버블맵

sns.set_palette("pastel")

# Create a base map
m_4 = folium.Map(location=[37.5665,126.9780], tiles='cartodbpositron', zoom_start=13)

# Add a bubble map to the base map
for i in range(0,len(pd_acc_data_child)):
    Circle(
        location=[pd_acc_data_child.iloc[i]['Y'], pd_acc_data_child.iloc[i]['X']],
        radius=5, color='red',
        tooltip=str(pd_acc_data_child.iloc[i]['피해운전자 연령'])+'세').add_to(m_4)

# Display the map
m_4

# %%
# 노인 버블맵

sns.set_palette("pastel")

# Create a base map
m_4 = folium.Map(location=[37.5665,126.9780], tiles='cartodbpositron', zoom_start=13)

# Add a bubble map to the base map
for i in range(0,len(pd_acc_data_older)):
    Circle(
        location=[pd_acc_data_older.iloc[i]['Y'], pd_acc_data_older.iloc[i]['X']],
        radius=5,
        color='blue',
        tooltip=str(pd_acc_data_older.iloc[i]['피해운전자 연령'])+'세').add_to(m_4)

# Display the map
m_4

# %%
# 어린이 히트맵

m_5 = folium.Map(location=[37.5665,126.9780], tiles='cartodbpositron', zoom_start=13)

mc = MarkerCluster()
for idx, row in pd_acc_data.iterrows():
    if not math.isnan(row['X']) and not math.isnan(row['Y']):
        mc.add_child(Marker([row['Y'], row['X']],tooltip=row['피해운전자 연령']))
m_5.add_child(mc)

# Add a heatmap to the base map
HeatMap(data=pd_acc_data_child[['Y', 'X']], radius=5).add_to(m_5)

# Display the map
m_5


지하철 이용객 수 집계(월평균)
# %%
import geopandas as gpd
import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
import math
import missingno as msno
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager, rc
import re

# %%
subway_station = pd.read_csv('./data/subway/CARD_SUBWAY_MONTH_201912.csv',encoding="EUC-KR")
subway_station
# %%
subway_pos = pd.read_csv('./data/subway/subway_crd_line_info-main/지하철역_좌표.csv')
subway_pos

# %%
subway_pos = subway_pos.rename(columns={'역이름':'역명'})

# %%
regex = "\(.*\)|\s-\s.*"

# %%
subway_station_name = subway_station['역명'].unique()

subway_station['역명']= pd.Series([re.sub(regex,'',row) for row in subway_station['역명']])

# %%
# inner join 진행 시 서로에게 없는 역들은 정보가 사라진다.
# join 진행 시 결측치들을 찾아보자.

ss = set(subway_station['역명'])
sp = set(subway_pos['역명'])
only_ss = ss-sp # subway_station에만 있는 역이름들이다.
only_sp = sp-ss # subway_pos에만 있는 역이름들이다.

# %% [markdown]
# 6월 한달 간의 지하철 역별 이용객 수이다.

# %%
subway_station_data1 = subway_station.merge(subway_pos,how='right',left_on='역명',right_on='역명')
# 좌표는 있으나 승객 정보가 없는 경우이다.
# 차후에 데이터를 채워넣던가 해야할듯 싶다.
# display(only_sp)

# %%
msno.matrix(df=subway_station_data1,color=(0.3,0.4,0.3))
plt.show()

# %%
subway_station_data2 = subway_station.merge(subway_pos,how='left',left_on='역명',right_on='역명')
display(subway_station_data2)
# 승객 데이터는 있지만 좌표데이터가 없는 경우이다.

# %%
# 데이터는 있으나 좌표가 없는 데이터들은 좌표를 채워줄 수 있도록 한다.
msno.matrix(df=subway_station_data2,color=(0.3,0.4,0.7))
plt.show()

# %%
# only_ss_data = subway_station[subway_station['역명'].isin(only_ss)]
# station = only_ss_data[['노선명','역명']]
# station.to_csv('have_to_get_station_pos.csv')

# %%
subway_station_data = subway_station.merge(subway_pos,left_on='역명',right_on='역명')

# %%
subway_station_sum = subway_station_data.groupby(['역명','x','y']).sum().reset_index()

# %%
subway_station_sum.drop(['사용일자','등록일자'],axis=1,inplace=True)

# %%
subway_station_sum['월평균승차총승객수'] = subway_station_sum['승차총승객수'] / 31
subway_station_sum['월평균하차총승객수'] = subway_station_sum['하차총승객수'] / 31

# %%
subway_station_sum.head()

# %%
subway_station_sum.to_csv('지하철 승하차 수(좌표포함).csv')
# %%
subway_station_sum[['승차총승객수','하차총승객수']].agg(['min','max','mean','std'])

# %%
# Create a base map
# 지하철 역의 분포를 보여줌....

m_5 = folium.Map(location=[37.5665,126.9780], tiles='cartodbpositron', zoom_start=13)

mc = MarkerCluster()
for idx, row in subway_station_sum.iterrows():
    if not math.isnan(row['x']) and not math.isnan(row['y']):
        mc.add_child(Marker([row['y'], row['x']],popup=row['역명'],tooltip=row['승차총승객수']))
m_5.add_child(mc)

# Add a heatmap to the base map
HeatMap(data=subway_station_sum[['y', 'x']], radius=10).add_to(m_5)

# Display the map
m_5


차량속도 집계 및 시각화
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('./LOCAL_PEOPLE_20210730.csv',encoding='cp949')

# %%
group_data = df.groupby('집계구코드')['총생활인구수'].sum().sort_values(ascending=False)
sum_data = group_data.to_frame('구별총생활인구수')
sum_data.to_csv('total_pop_by_gu.csv')

# %%
traffic_data = pd.read_excel("./data/traffic/2019년 12월 서울시 차량통행속도.xlsx")
traffic_data.head()

# %% [markdown]
# 링크아이디가 유일하게 결정된다.<br>
# 따라서 도로명 별로 말고 링크아이디 별로 산정된 내역도 보내주자.
# %%
display(len(traffic_data))
display(len(traffic_data['링크아이디'].unique()))

# %%
mean = list()
var = list()
std = list()
for i in range(0,len(traffic_data)):
    mean_data = traffic_data.loc[i,'01시':'24시'].mean()
    var_data = traffic_data.loc[i,'01시':'24시'].var()
    std_data = traffic_data.loc[i,'01시':'24시'].std()
    mean.append(mean_data)
    var.append(var_data)
    std.append(std_data)

# %%
traffic_data['MEAN'] = mean
traffic_data['VAR'] = var
traffic_data['STD'] = std

# %% [markdown]
# 주말과 평일의 데이터의 차이가 있겠는가? 검증해볼 필요가 있다.

# %%
aggregate_data_by_linkId = traffic_data.groupby('링크아이디')[['MEAN','VAR','STD']].mean()

# %%
aggregate_data_by_linkId.to_excel('링크아이디별 집계 데이터(주말,주중 구분X).xlsx')

# %%
weekDay = ['월','화','수','목','금']
weekEnd = ['토','일']

aggregate_weekDay_by_linkId = traffic_data[traffic_data['요일'].isin(weekDay)]
aggregate_weekEnd_by_linkId = traffic_data[traffic_data['요일'].isin(weekEnd)]

weekDay_agg_data = aggregate_weekDay_by_linkId.groupby(['링크아이디','시점명','종점명'])[['MEAN','VAR','STD']].mean()
weekEnd_agg_data = aggregate_weekEnd_by_linkId.groupby(['링크아이디','시점명','종점명'])[['MEAN','VAR','STD']].mean()

display(weekDay_agg_data.sort_values(by=['STD','VAR','MEAN']))
display(weekEnd_agg_data.sort_values(by=['STD','VAR','MEAN']))

# %%
weekDay_agg_data.to_excel('링크아이디별 주중 집계 데이터.xlsx')
weekEnd_agg_data.to_excel('링크아이디별 주말 집계 데이터.xlsx')

# %%
sns.histplot(data=weekDay_agg_data,x='MEAN')
plt.show()

# %%
sns.histplot(data=weekEnd_agg_data,x='STD')
plt.show()

# %%
import numpy as np

print('-'*15+' 주중 속도 표준편차 사분위수 '+'-'*15)
print('Q1 : {}'.format(np.quantile(weekDay_agg_data['STD'],0.25)))
print('Q2 : {}'.format(np.quantile(weekDay_agg_data['STD'],0.5)))
print('Q3 : {}\n'.format(np.quantile(weekDay_agg_data['STD'],0.75)))

print('-'*15+' 주말 속도 표준편차 사분위수 '+'-'*15)
print('Q1 : {}'.format(np.quantile(weekEnd_agg_data['STD'],0.25)))
print('Q2 : {}'.format(np.quantile(weekEnd_agg_data['STD'],0.5)))
print('Q3 : {}'.format(np.quantile(weekEnd_agg_data['STD'],0.75)))

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(20,5))
ax1 = sns.boxplot(ax=ax,x='STD',data=weekDay_agg_data, width=0.5)
plt.show()

# %%
fig, ax = plt.subplots(figsize=(20,5))
ax2 = sns.boxplot(ax=ax,x='STD',data=weekEnd_agg_data, width=0.5)

# %%
traffic_data.to_csv('traffic_aggregate_data.csv')
aggregate_result = traffic_data.groupby('도로명')[['MEAN','VAR','STD']].mean()
aggregate_result.to_csv('traffic_aggregate_data_by_road_name.csv')

# %%
traffic_data.to_excel('traffic_aggregate_data.xlsx')
aggregate_result = traffic_data.groupby('도로명')[['MEAN','VAR','STD']].mean()
aggregate_result.to_excel('traffic_aggregate_data_by_road_name.xlsx')

# %%
Q3 = traffic_data['STD'].quantile(.75)
print(Q3*1.5)
# %%
std_data = traffic_data['STD']

# %%
edit_std_data = traffic_data[traffic_data['기능유형구분'] != '도시고속도로']['STD']
a = edit_std_data[edit_std_data > 0].sort_values()
a

# %%
rank = std_data.sort_values()
std_not_nan = [row for row in std_data.iloc[:] if row != 0]
a= std_not_nan.sort()
print(a)

# %%
traffic_data.iloc[148797].to_frame()

# %%
print(len(std_data[std_data > Q3*1.5]))
len(std_data[std_data > Q3*1.5]) / len(traffic_data)*100

# %%
import seaborn as sns
ax = sns.boxplot(x='STD',data=traffic_data)

# %%
ax = sns.histplot(data=traffic_data,x='STD')

# %%
sns.heatmap(traffic_data.isnull())

집계구별 생활인구(월평균)
import pandas as pd
import os

pop_data_list = os.listdir('./LOCAL_PEOPLE_201912')
i = 0
# 일별로 집계구코드 기준 생활인구를 합산한다.
def agg_pop(date):
    # print('./LOCAL_PEOPLE_201912/'+date)
    df = pd.read_csv('./LOCAL_PEOPLE_201912/'+date)

    group_data = df.groupby('집계구코드')['총생활인구수'].sum().sort_values(ascending=False)
    print('집계구코드 개수 : {}'.format(len(df['집계구코드'].unique())))
    sum_data = group_data.to_frame('구별총생활인구수')
    
    return sum_data

pop_data = pd.DataFrame()

# 일별로 만든 생활인구 집계 data를 concat한다.
for date in pop_data_list:
    pop_data = pd.concat([pop_data,agg_pop(date)],axis=0)

# 마지막으로 합계된 dataframe을 다시 groupby한 후 평균 column을 생성해주자.
data = pop_data.groupby('집계구코드')['구별총생활인구수'].sum().to_frame()
data['구별총생활인구수(월평균)'] = data['구별총생활인구수'] / 31
data.to_csv('19년 12월 집계구코드별 평균생활인구수.csv')
data.head() 
