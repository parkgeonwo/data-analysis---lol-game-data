# lol-game-data-analysis


## 리그오브레전드 승리에 미치는 조건 분석

### 1. 주제 선정

대용량 데이터를 기반으로 하는 빅데이터 시대가 도래하면서 미디어 및 콘텐츠 산업 분야에서의 빅데이터 활용 사례가 증가함에 따라 평소에 즐겨하던 리그오브레전드 게임 데이터를 분석

### 2. 데이터 설명

게임 내에서의 여러가지 요소를 파악한 한국 유저들의 게임데이터를 kaggle에서 받아서 활용

(출처 : https://www.kaggle.com/park123/lol-data)	


  결측치, 이상치 없음.
  

- 데이터 전처리

데이터 분석에 용이하게 하기 위해서 한 단어로 구성된 값들은 정수로 변경하고 여러 단어로 표시된 값들은 한 데이터에 대해서 유무에 따라 새로운 컬럼을 추가함

문자로 구성된 “tier” 컬럼의 값들은 0~4 사이의 정수로 변경

True, False 로 표현된 6가지 컬럼의 값들은 0또는 1로 변경

여러가지 몬스터로 적혀있는 7가지 몬스터를 잡았는지 안잡았느지 유무에 따라 14개의 컬럼을 추가

```py
# 1. 'matchId' , 'makeTime' 컬럼 제거

import pandas as pd
match = pd.read_csv("c:\\data\\match_full_time.csv")

match2 = match.iloc[ : , 2:]
# print(match2)


# 2. 'blue_moster' 컬럼과 'red_monster' 컬럼의 몬스터 종류 하나에 따라 새로운 컬럼으로 0,1로 구분한다.

col_list1 = [ 'b_RIF', 'b_BAR', 'b_AIR' , 'b_EAR', 'b_FIRE' , 'b_WAT' , 'b_ELD' ]
col_list2 = [ 'r_RIF', 'r_BAR', 'r_AIR' , 'r_EAR', 'r_FIRE' , 'r_WAT' , 'r_ELD' ]
mon_list = [  'RIFTHERALD', 'BARON_NASHOR', 'AIR_DRAGON', 'EARTH_DRAGON', 'FIRE_DRAGON', 'WATER_DRAGON', 'ELDER_DRAGON'  ]

for i in range(len(mon_list)):
    match2[col_list1[i]] = match2.blue_monster.apply(lambda x : mon_list[i] in x).astype(int)
    match2[col_list2[i]] = match2.red_monster.apply(lambda x : mon_list[i] in x).astype(int)


# 삭제
match2.drop('blue_monster', axis = 1,inplace = True)         # 실제 데이터 프레임에서 지움
match2.drop('red_monster', axis = 1,inplace = True)         # 실제 데이터 프레임에서 지움


# 3. 'tier' ,  'blue_firstBlood' ,  'blue_firstTower' , 'blue_firstInhibitor' , 'blue_firstBaron' , 'blue_firstDragon' , 'blue_firstRiftHerald' 컬럼을
	# 숫자로 바꿔준다

tier_list = ['BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'DIAMOND']

for i in range( len(tier_list) ):
    match2.loc [ match2['tier'] == tier_list[i] , 'tier' ] = i

col_list3 = [ 'blue_firstBlood' ,  'blue_firstTower' , 'blue_firstInhibitor' , 'blue_firstBaron' , 'blue_firstDragon' , 'blue_firstRiftHerald' ]

for i in range( len(col_list3) ):
    match2[col_list3[i]] = match2[col_list3[i]].astype(int)


# 4. 'blue_win' 컬럼을 마지막으로 옮겨줌

match2['b_win'] = match['blue_win']

# 삭제
match2.drop('blue_win', axis = 1,inplace = True)         # 실제 데이터 프레임에서 지움


# 팽창계수가 높은 컬럼 삭제

match2.drop('blue_gold', axis = 1,inplace = True)
match2.drop('red_gold', axis = 1,inplace = True)
match2.drop('gameDuration', axis = 1,inplace = True)

```

### 3. 머신러닝 모델 적용

전처리한 데이터를 바탕으로 KNN / bernouilliNB / decision tree / Riper / decision tree + bagging /  randomForest 총 6가지 머신러닝 모델에 적용하여 결론을 도출

```py

# by 건우, 21/10/01, decision tree + bagging


from sklearn.preprocessing import MinMaxScaler

match3 = match2.iloc[     :    ,   :-1   ]        # 정답컬럼제외

scaler = MinMaxScaler()

scaler.fit(match3)                   # 최대 최소법으로 데이터를 계산합니다.

match_scaled = scaler.transform( match3 )            # 위에서 계산한 내용으로 데이터를 변환해서 match_scaled 담습니다.
# print(match_scaled)

# print ( match_scaled.shape )         # ( 5000, 28 )

y = match2['b_win'].to_numpy()        # 정답 데이터를 numpy array 로 변경합니다.
# print(y)


# 훈련데이터와 테스트데이터로 분리합니다. ( 훈련 90% , 테스트 10% )

from sklearn.model_selection import train_test_split

x_train , x_test, y_train, y_test = train_test_split( match_scaled, y, test_size = 0.1 , random_state = 1 )    

# print( x_train.shape )      # (4500, 28)
# print( x_test.shape )         # (500, 28)
# print( y_train.shape )         # (4500,)
# print( y_test.shape )          # (500,)


for i in range(1,101):              # random_state 값을 변화

    
    from sklearn.tree import DecisionTreeClassifier
    
    model = DecisionTreeClassifier( criterion = 'entropy', max_depth = 20, random_state =1 )
    
    from sklearn.ensemble import BaggingClassifier
    
    bagging = BaggingClassifier( model, max_samples = 0.9, max_features = 0.5, random_state = i )
    
    # 설명 : max_samples = 0.9 는 bag 에 데이터 담을때 훈련 데이터의 90% 를 샘플링하겠다.
    #             max_features = 0.5 하나의 예측기가 가져갈 수 있는 최대 컬럼의 갯수
    
    # 모델훈련
    bagging.fit( x_train, y_train )
    
    # 모델 예측
    result = bagging.predict( x_test )
    
    #모델평가
    
    accuracy = sum(result == y_test) / len(y_test)
    if accuracy >= 0.98:
        print('random_state : ', i , 'accuracy : ', accuracy )

```


### 4. 결과 

- 건물 지표 비교

블루팀의 타워가 많이 깨지면 질 확률이 가장 높았고, 그 다음 블루팀의 승리에 가장 큰 영향을 끼치는 요인은 레드팀의 깨진 타워수이다. 예상외로 킬수와 억제기가 깨진 수는 타워가 깨진 수보다 영향력이 덜했다.

- 몬스터 포획 지표 비교

승리에 가장 영향을 미치는 몬스터는 바론과 전령이였고, 드래곤 4종류 중에서는 물 > 대지 > 불 > 바람 순으로 큰 영향을 미친다.
장로드래곤이 가장 큰 영향을 미칠것이라는 예상과 달리 가장 낮았다. 이는 장로 드래곤을 잡은 데이터 표본 수 자체가 적기 때문에 나온 결과라고 예상된다.

- 첫번째 지표 비교

블루팀의 첫번째 억제기가 깨졌을 때 가장 승리에 영향을 미쳤고, 그 다음은 첫번째 타워가 큰 영향을 미쳤다.
첫번째 킬, 용, 전령은 상대적으로 게임 결과에 큰 영향이 없었다.










