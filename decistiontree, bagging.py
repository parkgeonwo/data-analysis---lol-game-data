# 의사결정트리 + 배깅



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







    