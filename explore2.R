alldata = read.csv('C:/Users/nholtzma/Downloads/fire_data_2001_2017.csv')

#1. Use LASSO to find variables of interest
#2. Calculate AUC on validation set for each number of variables
#3. test on test set

library(dplyr)
nodup = distinct(alldata[dim(alldata)[1]:1,], lat, lon, year, .keep_all=TRUE)
nodup$LC = as.factor(nodup$LC)

newdata = subset(nodup, fire==0)
newdata$yd = 365*(newdata$year-2001) + newdata$doy
newdata$daywt = 0
for (i in unique(newdata$yd)) {
  print(i)
  newdata$daywt[newdata$yd==i] = 1/sum(newdata$yd==i)
}

newdataF = subset(nodup, fire==1)
newdataF$yd = 365*(newdataF$year-2001) + newdataF$doy
pfactor = 1/(sum(alldata$fire==1)/ (2942*948780))
newdataF$daywt = 1/pfactor * sum(newdata$daywt)/dim(newdataF)[1]
#plot(newdata$NDWI_1w[newdata$LC==lci], newdata$LST_Day_1km_1w[newdata$LC==lci],pch='.',xlim=c(-0.8,0.1),ylim=c(260,335))
#plot(nodup$NDWI_1w[nodup$LC==lci & nodup$fire==1], nodup$LST_Day_1km_1w[nodup$LC==lci & nodup$fire==1],pch='.',xlim=c(-0.8,0.1),ylim=c(260,335))

alldata2 = rbind(newdata, newdataF)


a1 = dim(nodup)[1]
a2 = as.integer(0.8*a1)
alldata2 = nodup[sample(1:a1,a1,replace = F),]

mymod= glm(fire ~ LC *(GCVI_1m + NIR_1m + SWIR2_3m), data= alldata2[1:a2,], family = binomial)

testset = alldata2[(a2+1):a1,]
#testprobs = testset$daywt / sum(testset$daywt)
#testset = testset[sample(1:(a1-a2), 100000,prob = testprobs, replace = T),]


a1 = dim(nodup)[1]
a2 = as.integer(0.8*a1)
trainset = nodup[1:a2,]
testset = nodup[(a2+1):a1,]

rawmatrix = model.matrix(fire ~ . ,data=trainset[-c(1,102,103,112)])
rawmatrix = rawmatrix[,-1]
onehotdata = data.frame(rawmatrix)
onehotdata$fire = trainset$fire
halfdata = onehotdata[sample(1:449329,225000,replace = T),]
mylist = rep(0,122)
for (i in (1:122)) {
  print(i)
  mymod= glm(fire ~ LC12 + GCVI_1w +SWIR2_3m+ halfdata[,i], data= halfdata, family = binomial)
  mylist[i] = mymod$deviance
}
mylist[mylist==0] = NA
plot(mylist[-122])
which.min(mylist[1:121])
names(halfdata)[8]


rawmatrix = model.matrix(fire ~ . ,data=nodup[-c(1,102,103,112)])
rawmatrix = rawmatrix[,-1]
onehotdata = data.frame(rawmatrix)
onehotdata$fire = nodup$fire
N = dim(nodup)[1]
split1 = as.integer(0.1*N)

valset = onehotdata[1:split1,]
testset = onehotdata[(split1+1):(2*split1),]
trainset = onehotdata[(2*split1+1):N,]

library(glmnet)
lasso1 = glmnet(rawmatrix[(2*split1+1):N,], trainset$fire, family='binomial', alpha=1 )
plot(lasso1$lambda[1:67], diff(lasso1$dev.ratio))
which(lasso1$dev.ratio)
plot(diff(lasso1$dev.ratio))
mycoef = lasso1$beta[,20]
colnames(rawmatrix)[mycoef != 0]
plot(lasso1$df[1:20])
sel4 = colnames(rawmatrix)[lasso1$beta[,16] != 0]
sel20 = colnames(rawmatrix)[lasso1$beta[,20] != 0]
sel23 = colnames(rawmatrix)[lasso1$beta[,23] != 0]


mymodL23 = glm(as.formula(paste("fire ~ ",paste(sel23, collapse="+"),sep = "")), data=onehotdata, family=binomial)

mydf = lasso1$df[1:40]
whichdiff = which(diff(mydf) > 0)+1
myAIC = c()
myacc = c()
for (i in whichdiff) {
sel21 = colnames(rawmatrix)[lasso1$beta[,i] != 0]
mymodL16 = glm(as.formula(paste("fire ~ ",paste(sel21, collapse="+"),sep = "")), data=trainset, family=binomial)

mypred = predict(mymodL16, valset)
qtab = table(testset$fire,mypred > 0)/(dim(testset)[1])
myacc = c(myacc, qtab[1,1] + qtab[2,2])
myAIC = c(myAIC, AIC(mymodL16))
print(i)
}

plot(mydf[whichdiff],myAIC-min(myAIC))
plot(mydf[whichdiff],myacc)



mymodS = glm(fire ~ LC12 + GCVI_1w +SWIR2_3m+ ET_1w, data= onehotdata, family = binomial)
mymodS3 = glm(fire ~ LC12 + GCVI_1w +SWIR2_3m, data= onehotdata, family = binomial)

rawmatrixT = model.matrix(fire ~ . ,data=testset[-c(1,102,103,112)])[,-1]
onehotdataT = data.frame(rawmatrixT)
onehotdataT$fire = testset$fire

mypred = predict(mymodL23, onehotdataT)
myprob = 1/(1+exp(-mypred))
qtab = table(testset$fire,mypred > 0)/(dim(testset)[1])
qtab
qtab[1,1] + qtab[2,2]

mean(testset$fire==0)

library(stats)
AIC(mymodL15)

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

totrain = alldata2[1:a2,-c(1,102,103,112:115)]
totest = alldata2[(a2+1):a1,-c(1,102,103,112:115)]
library(xgboost)
mymod = xgboost(data = model.matrix(fire ~ ., totrain), label = totrain$fire, objective='binary:logistic', max.depth=2, eta=1, nrounds = 50)
xpred = predict(mymod, newdata=model.matrix(fire ~ ., totest))

cutoffs = seq(0,1,0.01)
fpr = c()
tpr = c()
for (i in cutoffs) {
  fpr = c(fpr, sum(xpred > i & totest$fire==0)/sum(totest$fire==0))
  tpr = c(tpr, sum(xpred > i & totest$fire==1)/sum(totest$fire==1))
}
plot(fpr,tpr, t='l',col='red', xlab='False positive rate', ylab='True positive rate',lwd=2,asp=1)
abline(0,1)
myfun = approxfun(fpr,tpr)
myauc2 = sum(myfun(seq(0,1,0.01)))*0.01

xgb.importance(model=mymod)

lines(fpr,tpr,col='blue',lwd=2)
legend('bottomright', pch=15, legend=c('Logistic regression', 'XGBoost'), col=c('red','blue'), inset = c(0.05,0.05))
