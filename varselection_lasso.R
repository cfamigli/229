alldata = read.csv('C:/Users/nholtzma/Downloads/fire_data_2001_2017.csv')

#1. Use LASSO to find variables of interest
#2. Calculate AUC on validation set for each number of variables
#3. test on test set
#this doesn't make snse because too many colinear vars
#instead, do forward selection, but drop each group after select one from it
#after select 10 features, test each combination against validation set to see which gives best accuracy

#do for only clim and only reflectance

library(dplyr)
nodup = distinct(alldata, lat, lon, year, .keep_all=TRUE)
nodup$LC = as.factor(nodup$LC)


rawmatrix = model.matrix(fire ~ .,data=nodup[-c(1,102,103,112)])
rawmatrix = rawmatrix[,-1]
onehotdata = data.frame(rawmatrix)
onehotdata$fire = nodup$fire
N = dim(nodup)[1]
split1 = as.integer(0.1*N)

valset = onehotdata[1:split1,]
testset = onehotdata[(split1+1):(2*split1),]
trainset = onehotdata[(2*split1+1):N,]

trainset = onehotdata[nodup$year <= 2015, ]
valset = onehotdata[nodup$year == 2016,]
testset = onehotdata[nodup$year == 2017,]

trainset = nodup[nodup$year <= 2015,-c(1,102,103,112) ]
valset = nodup[nodup$year == 2016,-c(1,102,103,112)]
testset = nodup[nodup$year == 2017,-c(1,102,103,112)]




library(glmnet)
library(stats)
lasso1 = glmnet(rawmatrix[nodup$year <= 2015,], trainset$fire, family='binomial', alpha=1 )


ridge1 = glmnet(rawmatrix[nodup$year <= 2015,], trainset$fire, family='binomial', alpha=0 )


c20 = lasso1$beta[,20]; i20 = lasso1$a0[20]

valmat = as.matrix(valset[,1:121])

mydf = lasso1$df[1:40]
whichdiff = which(diff(mydf) > 0)+1
myAIC = c()
myacc = c()
for (i in 1:100) {
  #sel21 = colnames(rawmatrix)[lasso1$beta[,i] != 0]
  #mymodL16 = glm(as.formula(paste("fire ~ ",paste(sel21, collapse="+"),sep = "")), data=trainset, family=binomial)
  #mypred = predict(mymodL16, valset)
  c20 = lasso1$beta[,i]; i20 = lasso1$a0[i]
  mypred = (i20 + valmat %*% c20)[,1]
  
  qtab = table(valset$fire,mypred > 0)/(dim(valset)[1])
  if (dim(qtab)[2] ==1) {
  myacc = c(myacc, qtab[1,1])
  }
  if (dim(qtab)[2]== 2) {
    myacc = c(myacc, qtab[1,1] + qtab[2,2])
  }
  
  print(i)
}

mysd = apply(rawmatrix[nodup$year <= 2015,], 2, sd)
plot(mysd*ridge1$beta[,20])

plot(myacc - mean(valset$fire ==0))
#plot(mydf[whichdiff],myAIC-min(myAIC))
plot(mydf[whichdiff],myacc - mean(valset$fire ==0))
mydf[whichdiff][6]
goodcol = whichdiff[6]
sel21 = colnames(rawmatrix)[lasso1$beta[,goodcol] != 0]
myformula = paste("fire ~ ",paste(sel21, collapse=" + "),sep = "")
myformula2 = "fire ~ ET_1d  + GCVI_1w + LC2 + LC12 + LC14 + LC16 + LC17 + LST_Day_1km_3m + NBR1_3m + NBR2_1w  +
  NDVI_3m + NIR_1w + SM_0_10_1d_clim +
  TCG_1w  + WS_3m + WS_3m_clim + precip_1d_clim"
mymodL16 = glm(as.formula(myformula), data=trainset, family=binomial)

mymodS = glm(fire ~ LC12 + GCVI_1w +SWIR2_3m+ ET_1w, data= trainset, family = binomial)

mypred = predict(mymodL16, testset)
mypred = predict(mymod, testset)

myprob = 1/(1+exp(-mypred))
qtab = table(testset$fire,mypred > 0)/(dim(testset)[1])
qtab = table(testset$fire,myprob > 0.25)/(dim(testset)[1])

qtab
qtab[1,1] + qtab[2,2]

cutoffs = seq(0,1,0.01)
fpr = c()
tpr = c()
for (i in cutoffs) {
  fpr = c(fpr, sum(myprob > i & testset$fire==0)/sum(testset$fire==0))
  tpr = c(tpr, sum(myprob > i & testset$fire==1)/sum(testset$fire==1))
}
plot(fpr,tpr, t='l',col='red', xlab='False positive rate', ylab='True positive rate',lwd=2,asp=1)
abline(0,1)
myfun = approxfun(fpr,tpr)
myauc = sum(myfun(seq(0,1,0.01)))*0.01
myauc


library(xgboost)

trainset = onehotdata[nodup$year <= 2015, ]
valset = onehotdata[nodup$year == 2016,]
testset = onehotdata[nodup$year == 2017,]


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
plot(myacc)
which.max(myacc)
xpred = predict(mymod, newdata=model.matrix(fire ~ ., testset),ntreelimit=32)

#validate with xpred = predict(mymod, newdata=model.matrix(fire ~ ., valset), ntreelimit = i) for i in 1:50, find accuracy

cutoffs = seq(0,1,0.01)
fpr = c()
tpr = c()
for (i in cutoffs) {
  fpr = c(fpr, sum(xpred > i & testset$fire==0)/sum(testset$fire==0))
  tpr = c(tpr, sum(xpred > i & testset$fire==1)/sum(testset$fire==1))
}
lines(fpr,tpr,col='green',lwd=2)
legend('bottomright', pch=15, legend=c('Logistic regression', 'XGBoost'), col=c('red','blue'), inset = c(0.05,0.05))

myfun = approxfun(fpr,tpr)
myauc2 = sum(myfun(seq(0,1,0.01)))*0.01

myacc = mean((xpred>0.5) == testset$fire)

trainset2 = trainset[sample(1:499475,50000,replace = T),]


mylist = rep(0,108)
for (i in (1:108)[c(-18,-100)]) {
  print(i)
  #if (!'GCVI' %in% names(trainset)[i] & !'SWIR' %in% names(trainset)[i] & !'ET_' %in% names(trainset)[i] ) {
  #mymod= glm(fire ~ LC12 + GCVI_1w +SWIR2_3m + ET_1w + LC16 +  LC17 + NDMI_1w + precip_1w_clim+ LC14 + trainset[,i], data= trainset, family = binomial)
  mymod= glm(fire ~ LC+GCVI_1w +trainset2[,i], data= trainset2, family = binomial)
  
  mylist[i] = mymod$deviance
  #}
}
mylist[mylist==0] = NA
names(trainset)[order(mylist)]
plot(mylist)
which.min(mylist)
names(trainset)[43]


acclist = c()
sel10 = strsplit('LC12 + GCVI_1w +SWIR2_3m + ET_1w + LC16 +  LC17 + NDMI_1w + precip_1w_clim+ LC14 + WS_3m','+',fixed=T)[[1]]
sel10 = strsplit('LC+GCVI_1w+ SWIR2_3m +ET_1w + NDMI_1w + precip_1w_clim+ WS_3m + 
                 NDVI_3m+NDWI_3m +GCVI_1m + NIR_1w ','+',fixed=T)[[1]]

for (i in 1:11) {
  print(i)
myformula = paste("fire ~ ",paste(sel10[1:i], collapse=" + "),sep = "")
mymod= glm(as.formula(myformula), data= trainset, family = binomial)
mypred = predict(mymod, valset)
qtab = table(valset$fire,mypred > 0)/(dim(valset)[1])
acclist = c(acclist,qtab[1,1] + qtab[2,2])
}
plot(acclist - mean(valset$fire==0))


finalmod = glm(fire ~ LC+GCVI_1w+ SWIR2_3m +ET_1w ,data= trainset, family = binomial)
finalmod = glm(fire ~ LC+GCVI_1w+ SWIR2_3m +ET_1w + NDMI_1w + precip_1w_clim+ WS_3m + 
                 NDVI_3m, data= trainset, family = binomial)
#finalmod = glm(fire ~ LC+GCVI_1m+ NIR_1m +SWIR2_3m ,data= trainset, family = binomial)

mypred = predict(finalmod, testset)
qtab = table(testset$fire,mypred > 0)/(dim(testset)[1])

mymod= glm(fire ~ LC12 + GCVI_1w +SWIR2_3m + ET_1w + LC16 +  LC17 + NDMI_1w + precip_1w_clim+ LC14 + trainset[,i], data= trainset, family = binomial)


mymod= glm(fire ~ LC12 + GCVI_1w +SWIR2_3m + ET_1w, data= trainset, family = binomial)

rawnoLC = model.matrix(fire ~ .,data=nodup[-c(1,19,102,103,112)])
rawnoLC = rawnoLC[,-1]

#svd makes no sense here because there are multiple versions of some columns but not others

mysvd = svd(rawnoLC[nodup$year <= 2015,])
plot(mysvd$d)
myclip = data.frame(mysvd$u[,1:6])
myclip$fire = nodup$fire[nodup$year <= 2015]
#myclip[,1:6] = scale(myclip[,1:6])
svdmod = glm(fire ~ ., data=myclip, family=binomial)
mymat = nodup[,c('fire','LC')]
mymat = model.matrix(fire~., mymat)[,-1]
mymat = as.data.frame(mymat)
mymat$LC1 = nodup$LC==1
cliplc = cbind(myclip,mymat[nodup$year <= 2015,])
svdmod2 = glm(fire ~ ., data=cliplc, family=binomial)
which(names(cliplc) == 'LC15')
svdmod2 = glm(fire ~ ., data=cliplc[,-c(20)], family=binomial)



mylist = rep(0,23)
for (i in (1:23)[c(-2,-7,-17,-16)]) {
  print(i)
  mymod= glm(fire ~ X2 + LC12 + LC16 + cliplc[,i], data= cliplc, family = binomial)
  mylist[i] = mymod$deviance
}
mylist[mylist==0] = NA
#mylist[7] = NA
plot(mylist)
which.min(mylist)
names(cliplc)[2]

cor(cliplc$X6, cliplc$fire)

i = 3
plot(mysvd$v[,i]*mysvd$d[i])
colnames(rawnoLC)[abs(mysvd$v[,i]) > 0.01*sum(abs(mysvd$v[,i]))]
