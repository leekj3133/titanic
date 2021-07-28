# 타이타닉 생존자 예측

![image](https://user-images.githubusercontent.com/75352728/127244675-2722ccb6-69cd-4e04-9029-630fa3fc230c.png)

*그림 출처 : Second Layer

***

## 1. INTRO

## 1-1. 목적
- kaggle이 제공한 타이타닉 호 침몰 사건 당시의 사망자와 생존자를 구분하는 요인 분석을 통해, 승객들의 생존 여부를 예측하는 회귀 분석 프로젝트 입니다.
- 타이타닉 생존자 예측 EDA

## 1-2. CONCLUSION
- Conclusion
  - 당시대는 신분사회로서 1class에 귀족들이 타고 있으며 위층에 위치했고, 신분이 낮은 계급은 밑에 층에 위치하고 있었다
  - lady first 가 있을 정도로 여성을 보호하려는 시대상이 있었다.
  - 타이타닉호는 1class 일수록, 여성일 수록 생존률이 높다.  

  <br/>

  
## 1-3. 수집 데이터
- 출처 
  -  Kaggle : Titanic - Machine Learning from Disaster
  -  링크 : https://www.kaggle.com/c/titanic/overview 
  -  Column
        - pclass : 객실등급
        - survived : 생존유무
        - sex : 성별
        - age : skdl
        - sibsp : 형제 혹은 부부의 수
        - parch : 부모 혹은 자녀의 수
        - fare : 지불한 요금
        - boat : 탈출을 했다면 탑승한 보트의 번호
     

*****

<br/>

## 2. PROCESS
 
<br/>

### 2-1. 데이터 읽기

```
import pandas as pd

titanic_url = 'https://raw.githubusercontent.com/PinkWink/ML_tutorial'+\
            '/master/dataset/titanic.xls'
titanic = pd.read_excel(titanic_url)
titanic.head()
```

<img width="1171" alt="스크린샷 2021-07-28 오전 9 47 59" src="https://user-images.githubusercontent.com/75352728/127245725-1e2ac6d2-c63a-4437-9456-3ae2b3908d66.png">

<br/>
<br/>

### 2-2. EDA

```
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt


from matplotlib import font_manager, rc
plt.rcParams['axes.unicode_minus'] = False


f_path = "C:\Windows\Fonts\malgun.ttf"
font_name = font_manager.FontProperties(fname=f_path).get_name()
rc('font',family=font_name)
```



<br/>


```
f,ax = plt.subplots(1,2,figsize=(18,8))

titanic['survived'].value_counts().plot.pie(explode=[0,0.1],
                                          autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Pie plot - Survivied')
ax[0].set_ylabel('')
sns.countplot('survived', data=titanic, ax=ax[1])
ax[1].set_title('Count plot - Survived')

plt.show()
```
<br/>

<img width="579" alt="스크린샷 2021-07-28 오전 9 50 07" src="https://user-images.githubusercontent.com/75352728/127245877-d9c838e4-c5e9-4c3d-ada3-213f78d1f73d.png">

<br/>

## 생존 상황 - 38.2% 생존률


<br/>

```
f,ax = plt.subplots(1,2,figsize=(18,8))

sns.countplot('sex', data=titanic, ax=ax[0])
ax[0].set_title('Count of Passengers of sex')
ax[0].set_ylabel('')

sns.countplot('sex', hue='survived',data=titanic, ax=ax[1])
ax[1].set_title('Sex:Survived and Unsurvivied')

plt.show()
```


<img width="579" alt="스크린샷 2021-07-28 오전 9 51 36" src="https://user-images.githubusercontent.com/75352728/127246002-c389c8b0-2bc7-4909-b849-bf6398f4868c.png">



<br/>

## 성별에 따른 생존 상황 - 남성의 생존 가능성 낮다 


`pd.crosstab(titanic['pclass'], titanic['survived'], margins=True)`


<br/>

<img width="204" alt="스크린샷 2021-07-28 오전 9 52 53" src="https://user-images.githubusercontent.com/75352728/127246102-3c3aa291-a751-4a24-bf32-c3c0a03baf20.png">


<br/>

## 경젱력 대비 생존률 - 1등실, 여성의 생존률이 높다.

<br/>
<br/>

```
grid = sns.FacetGrid(titanic, row='pclass', col='sex',height=4,aspect=2)
grid.map(plt.hist, 'age', alpha=.8, bins=20)
grid.add_legend();
```


<br/>

<img width="584" alt="스크린샷 2021-07-28 오전 9 54 07" src="https://user-images.githubusercontent.com/75352728/127246188-b326e7ab-53e5-4a34-b3ae-23284f7eb54e.png">


<br/>

## 선실등급별 성별 상황 
## 3등실에 남성이 많다 - 특히 20대 남성

<br/>

```
import plotly.express as px

fig = px.histogram(titanic, x='age')
fig.show()
```

<br/>

<img width="561" alt="스크린샷 2021-07-28 오전 9 55 16" src="https://user-images.githubusercontent.com/75352728/127246284-29157045-1489-4233-8fa6-14cb5cc61400.png">

<br/>

## 나이별 승객 현황- 아이들, 20~30대가 많음

<br/>

```
grid = sns.FacetGrid(titanic, col= 'survived', row='pclass', height=4, aspect=2)
grid.map(plt.hist, 'age',alpha=.5, bins=20)
grid.add_legend();
```

<br/>

<img width="582" alt="스크린샷 2021-07-28 오전 9 56 47" src="https://user-images.githubusercontent.com/75352728/127246424-fa79369b-afd4-4735-b3f2-dd4b6e1eecf0.png">

<br/>

## 등실별 생존률을 연령별로 관찰 - 선실 등급이 높으면 생존률 높음

<br/>

```
# 나이 5단계로 분리
titanic['age_cat'] = pd.cut(titanic['age'], bins = [0,7,15,30,60,100],
                           include_lowest=True,
                           labels=['baby','teen','young','adult','old'])
titanic.head()
```

<br/>

#### 나이 5개로 분리 

<br/>

<img width="1181" alt="스크린샷 2021-07-28 오전 9 57 59" src="https://user-images.githubusercontent.com/75352728/127246523-a918aa85-a1c5-49db-b076-04b4e8766dd3.png">

<br/>

```
plt.figure(figsize=(12,4))
plt.subplot(131)
sns.barplot('pclass','survived',data=titanic)

plt.subplot(132)
sns.barplot('age_cat','survived',data=titanic)

plt.subplot(133)
sns.barplot('sex','survived',data=titanic)


plt.subplots_adjust(top=1, bottom=0.1, left=0.1, right=1, hspace=0.5, wspace=0.5)
```

<br/>

<img width="724" alt="스크린샷 2021-07-28 오전 9 59 13" src="https://user-images.githubusercontent.com/75352728/127246625-6d17e2bb-e19d-475a-8e09-5880a606c0f5.png">

<br/>

## 나이, 성별, 등급별 생존자 수 한번에 파악 - 어리고 1등실이고 여성이 생존률이 높다


<br/>

#### 남/여 나이별 생존 상황

<br/>

```
fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(14,6))

women = titanic[titanic['sex']=='female']
men = titanic[titanic['sex']=='male']

ax = sns.distplot(women[women['survived']==1]['age'], bins=20,
                  label='survived', ax = axes[0],kde=False)
ax = sns.distplot(women[women['survived']==0]['age'], bins=40,
                  label='not_survived', ax = axes[0],kde=False)

ax.legend(); ax.set_title('Female')

ax = sns.distplot(men[men['survived']==1]['age'], bins=18,
                  label='survived', ax = axes[1],kde=False)
ax = sns.distplot(men[men['survived']==0]['age'], bins=40,
                  label='not_survived', ax = axes[1],kde=False)
ax.legend(); ax.set_title('Male')
```

<br/>

<img width="724" alt="스크린샷 2021-07-28 오전 10 00 30" src="https://user-images.githubusercontent.com/75352728/127246714-4b339be1-22f6-495d-a247-c434d3ef5ff8.png">


<br/>

#### 탑승객의 이름 확인 - 이름에 신분이 나타난다.

<br/>

```
for idx, dataset in titanic.iterrows():
    print(dataset['name'])
```

<br/>

<img width="724" alt="스크린샷 2021-07-28 오전 10 01 26" src="https://user-images.githubusercontent.com/75352728/127246786-eaf75711-b0fa-4f39-8fc7-76195b9ffb6e.png">

<br/>

#### 정규식을 이용한 신분 정보 파악

<br/>

```
import re

title = []
for idx, dataset in titanic.iterrows():
    title.append(re.search('\,\s\w+(\s\w+)?\.',dataset['name']).group()[2:-1])
    
titanic['title'] = title
titanic.head()
```

<br/>

<img width="1175" alt="스크린샷 2021-07-28 오전 10 05 19" src="https://user-images.githubusercontent.com/75352728/127247092-3323a278-77cb-4fbf-8201-946bfde099c9.png">

<br/>

#### 성별별로 본 귀족

<br/>

```
pd.crosstab(titanic['title'],titanic['sex'])
```

<br/>

<img width="194" alt="스크린샷 2021-07-28 오전 10 06 07" src="https://user-images.githubusercontent.com/75352728/127247154-bcbdd2ab-aa40-4ada-910a-978ab0ac0c93.png">

<br/>

#### 사회적 신분 정리

<br/>

```
titanic['title'] = titanic['title'].replace('Mlle','Miss')
titanic['title'] = titanic['title'].replace('Ms','Miss')
titanic['title'] = titanic['title'].replace('Mme','Miss')

Rare_f = ['Dona','Dr','Lady','the Countess']

Rare_m = ['Capt','Col','Don','Major','Rev','Sir','Jonkheer','Master']

for each in Rare_f:
    titanic['title'] = titanic['title'].replace(each, 'Rare_f')
    
    
for each in Rare_m:
    titanic['title'] = titanic['title'].replace(each, 'Rare_m')

titanic['title'].unique()
```

<br/>

<img width="505" alt="스크린샷 2021-07-28 오전 10 07 01" src="https://user-images.githubusercontent.com/75352728/127247241-2367d98b-dacf-492d-861f-5b89ba669ce6.png">


<br/>

```
titanic[['title','survived']].groupby(['title'],as_index=False).mean()
```

<br/>

<img width="162" alt="스크린샷 2021-07-28 오전 10 08 11" src="https://user-images.githubusercontent.com/75352728/127247327-6f0fe96d-8243-4052-b677-d7b3df5df23e.png">

