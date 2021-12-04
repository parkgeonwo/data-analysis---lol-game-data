박건우_R_코드

match <- read.csv("match_change.csv", stringsAsFactors = T )

normalize <- function(x) {
  return ( (x-min(x)) / ( max(x)-min(x) ) )
}

match_n <- as.data.frame( lapply( match, normalize ) )

# match_n

# library(caret)

train_num <- createDataPartition ( match_n$b_win, p = 0.9 , list = F )

train_data <- match_n[ train_num,  ]
test_data <- match_n[ -train_num,  ]

# nrow(train_data)       # 4500
# nrow(test_data)        # 500

# library(party)
model <- ctree( b_win~. , data = train_data )        # '.' 은 나머지 모든컬럼을 말한다. / 여기서는 라벨을 뺀 모든컬럼

plot(model)



library(psych)

pairs.panels( match_n[  , c('tier', 'blue_kill', 'red_kill', 'blue_tower', 'blue_inhibitor', 'red_tower', 'red_inhibitor', 'b_win') ] )

pairs.panels( match_n[  , c('blue_firstBlood', 'blue_firstTower', 'blue_firstInhibitor', 'blue_firstBaron', 'blue_firstDragon', 'blue_firstRiftHerald', 'b_win') ] )

pairs.panels( match_n[  , c( 'b_RIF', 'b_BAR', 'b_AIR', 'b_EAR', 'b_FIRE', 'b_WAT',  'b_ELD', 'b_win') ] )
