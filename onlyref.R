library(dplyr)

alldata = subset(alldata, LC != 12 & LC != 13 & LC != 14)

nodup = distinct(alldata, lat, lon, year, .keep_all=TRUE)
ndyear = nodup$year
nodup = nodup[-c(1,102,103,112)]#[c(c(1:3,12:17,22:39,44:46,79:90),18,100)]
nodup$LC = as.factor(nodup$LC)

trainset = nodup[ndyear <= 2015,]
valset = nodup[ndyear == 2016,]
testset = nodup[ndyear == 2017,]

trainset2 = trainset[sample(1:499475,100000,replace = T),]

mylist = rep(0,43)
for (i in (1:43)) {
  print(i)
  mymod= glm(fire ~ LC + GCVI_1w + SWIR2_3m+ NDVI_1w+ NDWI_1w + NDMI_1w +trainset2[,i], data= trainset2, family = binomial)
  
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

mean((predict(finalmod,trainset)>0)==trainset$fire)
mean((predict(finalmod,testset)>0)==testset$fire)
#log reg only ref
#train acc: 0.7833665
#test acc: 0.762491


#xgboost only ref
#train acc: 0.796212
#test acc:  0.7738212


#ALL VARS
#log reg
#train acc: 0.785745
#test acc: 0.7612439

#log reg, no feature selection
#0.7938315
#0.7729456

#xgb
#train acc: 0.8042705
#test acc: 0.7762624


xpred = predict(mymod, newdata=model.matrix(fire ~ ., testset),ntreelimit=32)
mean((xpred > 0.5) == testset$fire)
xpred = predict(mymod, newdata=model.matrix(fire ~ ., trainset),ntreelimit=32)
mean((xpred > 0.5) == trainset$fire)


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

lines(fpr,tpr,col='green',lwd=2)

#full log: 0.7437272

myfun = approxfun(fpr,tpr)
myauc = sum(myfun(seq(0,1,0.01)))*0.01
myauc #0.676025

cutoffs = seq(0,1,0.01)
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

write.csv(summary(finalmodsel)$coefficients,file='logistic_4feature.csv')


mypred = predict(finalmod, testset)
qtab = table(testset$fire,mypred > 0)
