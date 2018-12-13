nodup = distinct(alldata, lat, lon, year, .keep_all=TRUE)
ndyear = nodup$year
nodup = nodup[-c(1,102,103,112)][-c(1:3,12:17,22:39,44:46,79:90)]
nodup$LC = as.factor(nodup$LC)

trainset = nodup[ndyear <= 2015,]
valset = nodup[ndyear == 2016,]
testset = nodup[ndyear == 2017,]

trainset2 = trainset[sample(1:499475,100000,replace = T),]

mylist = rep(0,66)
for (i in (1:66)[-58]) {
  print(i)
  mymod= glm(fire ~ LC +SM_0_10_1d_clim+SM_10_40_1d_clim+ET_1d+ trainset[,i], data= trainset, family = binomial)
  
  mylist[i] = mymod$deviance
  #}
}
mylist[mylist==0] = NA
names(trainset)[order(mylist)]
plot(mylist)
which.min(mylist)
names(trainset)[43]

acclist = c()
sel10 = strsplit('LC + ET_1d + WS_3m +precip_1w_clim +SM_40_100_3m_clim + SM_100_200_3m_clim+ LST_Day_1km_3m + LST_Day_1km_1m + SM_0_10_1d_clim + SM_10_40_3m_clim','+',fixed=T)[[1]]
sel10 = strsplit(' LC +SM_0_10_1d_clim+SM_10_40_1d_clim+ET_1d')
for (i in 1:4) {
  print(i)
  myformula = paste("fire ~ ",paste(sel10[1:i], collapse=" + "),sep = "")
  mymod= glm(as.formula(myformula), data= trainset, family = binomial)
  mypred = predict(mymod, valset)
  #qtab = table(valset$fire,mypred > 0)/(dim(valset)[1])
  acclist = c(acclist,mean((mypred > 0) == valset$fire))
}
plot(acclist - mean(valset$fire==0))
finalmod = glm(fire ~ LC +GCVI_1w + SWIR2_3m +NIR_1w+NBR1_3m ,data= trainset, family = binomial)

finalmod = glm(fire ~ LC +GCVI_1w + SWIR2_3m +NIR_1w+NBR1_3m ,data= trainset, family = binomial)



mymod= glm(fire ~ LC, data= trainset, family = binomial)
mean((predict(mymod,trainset)>0)==trainset$fire)
mypred = predict(mymod, valset)
mean((mypred > 0) == valset$fire)
mean(valset$fire==0)
#logistic only clim
#train acc: 0.7489964
#test acc:  0.7280816


#ROC curve for all-feature models
#table of accuracies for all 6 models

library(xgboost)

#trainset = 
#valset = onehotdata[nodup$year == 2016,]
#testset = onehotdata[nodup$year == 2017,]


mymod = xgboost(data = model.matrix(fire ~ ., trainset), label = trainset$fire, objective='binary:logistic',
                max.depth=2, eta=1, nrounds = 100)
#watchlist=xbb.DMatrix(model.matrix(fire ~ ., testset),label=testset$fire))
plot(mymod$evaluation_log)

myacc = rep(0,100)
for (i in 1:100) {
  print(i)
  xpred = predict(mymod, newdata=model.matrix(fire ~ ., valset), ntreelimit=i)
  myacc[i] = mean((xpred>0.5) == valset$fire)
}
plot(myacc - mean(valset$fire==0))
which.max(myacc)
xpred = predict(mymod, newdata=model.matrix(fire ~ ., testset),ntreelimit=57)
mean((xpred > 0.5) == testset$fire)
xpred = predict(mymod, newdata=model.matrix(fire ~ ., trainset),ntreelimit=56)
mean((xpred > 0.5) == trainset$fire)
#xgboost only clim
#train error: 0.7893268
#test error: 0.7689389


qtab = table(testset$fire,xpred > 0.5)
