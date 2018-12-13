library(dplyr)

alldata = read.csv('C:/Users/nholtzma/Downloads/fire_data_2001_2017.csv')

nodup = distinct(alldata, lat, lon, year, .keep_all=TRUE)
ndyear = nodup$year
nodup = nodup[-c(1,102,103,112)] #all vars
#nodup = nodup[-c(1,102,103,112)][c(c(1:3,12:17,22:39,44:46,79:90),18,100)] #for reflectance only
#nodup = nodup[-c(1,102,103,112)][-c(1:3,12:17,22:39,44:46,79:90)] #for climate only
nodup$LC = as.factor(nodup$LC)

trainset = nodup[ndyear <= 2015,]
valset = nodup[ndyear == 2016,]
testset = nodup[ndyear == 2017,]

mylist = rep(0,43)
for (i in (1:43)) {
  print(i)
  mymod= glm(fire ~ LC + GCVI_1w + SWIR2_3m+ NDVI_1w+ NDWI_1w + NDMI_1w +trainset[,i], data= trainset, family = binomial)
  
  mylist[i] = mymod$deviance
  #}
}
mylist[mylist==0] = NA
names(trainset)[order(mylist)]
plot(mylist)
which.min(mylist)
names(trainset)[43]

acclist = c()
sel10 = strsplit('LC +GCVI_1w + SWIR2_3m +NIR_1w+NBR1_3m+ SWIR1_1w','+',fixed=T)[[1]]
sel10 = strsplit('LC + GCVI_1w + SWIR2_3m+ NDVI_1w+ NDWI_1w + NDMI_1w','+',fixed=T)[[1]]

for (i in 1:6) {
  print(i)
  myformula = paste("fire ~ ",paste(sel10[1:i], collapse=" + "),sep = "")
  mymod= glm(as.formula(myformula), data= trainset, family = binomial)
  mypred = predict(mymod, valset)
  acclist = c(acclist,mean((mypred > 0) == valset$fire))
}
plot(acclist - mean(valset$fire==0))
finalmod = glm(fire ~ LC +GCVI_1w + SWIR2_3m +NIR_1w+NBR1_3m ,data= trainset, family = binomial)

finalmodsel = glm(fire ~ LC+GCVI_1w+ SWIR2_3m +ET_1w ,data= trainset, family = binomial)

finalmod = glm(fire ~ .,data= trainset, family = binomial)


mypred = predict(finalmod, testset)
myprob = 1/(1+exp(-mypred))

cutoffs = seq(0,1,0.01)
fpr = c()
tpr = c()
for (i in cutoffs) {
  fpr = c(fpr, sum(myprob > i & testset$fire==0)/sum(testset$fire==0))
  tpr = c(tpr, sum(myprob > i & testset$fire==1)/sum(testset$fire==1))
}
plot(fpr,tpr, t='l',col='orange', xlab='False positive rate', ylab='True positive rate',lwd=2,asp=1)
abline(0,1)

myfun = approxfun(fpr,tpr)
myauc = sum(myfun(seq(0,1,0.01)))*0.01
myauc #0.676025

library(xgboost)

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

qtab = table(testset$fire,xpred > 0.5)

fpr = c()
tpr = c()
for (i in cutoffs) {
  fpr = c(fpr, sum(xpred > i & testset$fire==0)/sum(testset$fire==0))
  tpr = c(tpr, sum(xpred > i & testset$fire==1)/sum(testset$fire==1))
}
lines(fpr,tpr,col='blue',lwd=2)
legend('bottomright', pch=15, legend=c('Log reg (no selection)','Log reg (feature sel)', 'XGBoost'), col=c('red','green','blue'), inset = c(0.05,0.05))

myfun = approxfun(fpr,tpr)
myauc2 = sum(myfun(seq(0,1,0.01)))*0.01
myauc2 #0.7540388
