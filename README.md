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


