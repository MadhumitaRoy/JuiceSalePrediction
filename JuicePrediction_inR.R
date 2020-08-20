#Load Pacman and required libraries
if("pacman" %in% rownames(installed.packages())==FALSE){install.packages("pacman")}

pacman::p_load("xgboost","dplyr","caret",
               "ROCR","lift","glmnet","MASS","e1071"
               ,"mice","partykit","rpart","randomForest","dplyr"   
               ,"lubridate","ROSE","smotefamily","DMwR","beepr","MLmetrics",
               "caretEnsemble","mlbench","gbm","gdata")
library(tidyverse)
library(corrplot)
# Reading the csv file
Juice_data <- read.csv("OJ.csv")

head(Juice_data,6)
str(Juice_data)
summary(Juice_data)

## Data Exploration and Feature Selection ##

# Checking for missing data
map(Juice_data, ~sum(is.na(.)))
# No missing data

# Checking for correlated features
# Grab only numeric columns
num.cols <- sapply(Juice_data, is.numeric)
# Filter to numeric columns for correlation
cor.data <- cor(Juice_data[,num.cols])
cor.data
# Plotting the correlation
corrplot(cor.data, method = "color")


table(Juice_data$Purchase)
# Target variable is balanced
table(Juice_data$StoreID)

# Selecting the relevant features after removing the highly correlated features and duplicates
# Important to use dplyr::select instead of select as it will not work with MASS package
Juice_data <- dplyr::select(Juice_data,Purchase,StoreID,PriceCH,PriceMM,LoyalCH,SalePriceMM,SalePriceCH,SpecialCH,SpecialMM)

# Changing Purchase to binary
Juice_data$Purchase <- ifelse(Juice_data$Purchase == "CH",1,0)

# Changing categorical variables to factors
# Different from Python: Target variable in Python is not changed to categorical/factor
Juice_data$Purchase <- as.factor(Juice_data$Purchase)
Juice_data$StoreID <- as.factor(Juice_data$StoreID)
Juice_data$SpecialCH <- as.factor(Juice_data$SpecialCH)
Juice_data$SpecialMM <- as.factor(Juice_data$SpecialMM)

head(Juice_data)
str(Juice_data)

# Splitting into train and test with 80:20 ratio
set.seed(42) #set a random number generation seed to ensure that the split is the same everytime
inTrain <- createDataPartition(y = Juice_data$Purchase,
                               p = 0.8, list = FALSE)

training <- Juice_data[ inTrain,]
testing <- Juice_data[ -inTrain,]

#### Random Forest ####
# Without hyperparameter tuning
model_forest <- randomForest(Purchase~ ., data=training, 
                             type="classification",
                             importance=TRUE,
                             ntree = 500,           # hyperparameter: number of trees in the forest
                             mtry = 5,             # hyperparameter: number of random columns to grow each tree
                             nodesize = 10,         # hyperparameter: min number of datapoints on the leaf of each tree
                             maxnodes = 10,         # hyperparameter: maximum number of leafs of a tree
                             cutoff = c(0.5, 0.5)   # hyperparameter: how the voting works; (0.5, 0.5) means majority vote
) 
plot(model_forest)  
varImpPlot(model_forest) 
###Finding predicitons: probabilities and classification
forest_probabilities<-predict(model_forest,newdata=testing,type="prob") #Predict probabilities -- an array with 2 columns: for not retained (class 0) and for retained (class 1)
forest_classification<-rep("1",213) # 213 is the number of testing rows
# Threshold is taken as 0.5 as there is 50% chance a person will buy CH or MM
forest_classification[forest_probabilities[,2]<0.5]="0" 
forest_classification<-as.factor(forest_classification)

#There is also a "shortcut" forest_prediction<-predict(model_forest,newdata=testing, type="response") 
#But it by default uses threshold of 50%: actually works better (more accuracy) on this data

confusionMatrix(forest_classification,testing$Purchase, positive="1") 

####ROC Curve
forest_ROC_prediction <- prediction(forest_probabilities[,2], testing$Purchase) #Calculate errors
forest_ROC <- performance(forest_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(forest_ROC) 

####AUC (area under curve)
AUC.tmp <- performance(forest_ROC_prediction,"auc") #Create AUC data
forest_AUC <- as.numeric(AUC.tmp@y.values) #Calculate AUC
forest_AUC #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value
# Gives an AUC of 89%

#### Lift chart
plotLift(forest_probabilities[,2],  testing$Purchase, cumulative = TRUE, n.buckets = 10) # Plot Lift chart

### An alternative way is to plot a Lift curve not by buckets, but on all data points
Lift_forest <- performance(forest_ROC_prediction,"lift","rpp")
plot(Lift_forest)

## Hyperparameter Tuning
#Hyperparameter Tuning. Gives the optimized hyperparameters.

control<-trainControl(method="repeatedcv",number=10,repeats=10,search="random")

for(i in c(250,300,500,750)){
  rf_random<-train(Purchase~.,
                   data=training,ntree=i,method="rf",metric="Accuracy",tuneLength=15,trControl=control)
  print(i)
  print(rf_random)
  plot(rf_random)
}
# Best hyperparameters: ntree = 750, mtry = 2.
# Using the best hyperparameters in random forest
model_forest <- randomForest(Purchase~ ., data=training, 
                             type="classification",
                             importance=TRUE,
                             ntree = 750,           # hyperparameter: number of trees in the forest
                             mtry = 2,             # hyperparameter: number of random columns to grow each tree
                             nodesize = 10,         # hyperparameter: min number of datapoints on the leaf of each tree
                             maxnodes = 10,         # hyperparameter: maximum number of leafs of a tree
                             cutoff = c(0.5, 0.5)   # hyperparameter: how the voting works; (0.5, 0.5) means majority vote
) 
plot(model_forest)  
varImpPlot(model_forest) 
###Finding predicitons: probabilities and classification
forest_probabilities<-predict(model_forest,newdata=testing,type="prob") #Predict probabilities -- an array with 2 columns: for not retained (class 0) and for retained (class 1)
forest_classification<-rep("1",213) # 213 is the number of testing rows
# Threshold is taken as 0.5 as there is 50% chance a person will buy CH or MM
forest_classification[forest_probabilities[,2]<0.5]="0" 
forest_classification<-as.factor(forest_classification)

#There is also a "shortcut" forest_prediction<-predict(model_forest,newdata=testing, type="response") 
#But it by default uses threshold of 50%: actually works better (more accuracy) on this data

confusionMatrix(forest_classification,testing$Purchase, positive="1") 
# Accuracy increases from 0.8357 to 0.8545 after tuning

####ROC Curve
forest_ROC_prediction <- prediction(forest_probabilities[,2], testing$Purchase) #Calculate errors
forest_ROC <- performance(forest_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(forest_ROC) 

####AUC (area under curve)
AUC.tmp <- performance(forest_ROC_prediction,"auc") #Create AUC data
forest_AUC <- as.numeric(AUC.tmp@y.values) #Calculate AUC
forest_AUC #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value
# AUC is 0.88.

#### XgBoost ####
Juice_data_matrix <- model.matrix(Purchase~ ., data = Juice_data)[,-1]

x_train <- Juice_data_matrix[ inTrain,]
x_test <- Juice_data_matrix[ -inTrain,]

y_train <-training$Purchase
y_test <-testing$Purchase

# Without hyperparameter tuning
model_XGboost<-xgboost(data = data.matrix(x_train), 
                       label = as.numeric(as.character(y_train)), 
                       eta = 0.1,       # hyperparameter: learning rate 
                       max_depth = 20,  # hyperparameter: size of a tree in each boosting iteration
                       nround=50,       # hyperparameter: number of boosting iterations  
                       objective = "binary:logistic"
)

XGboost_prediction<-predict(model_XGboost,newdata=x_test, type="prob") #Predict classification (for confusion matrix)
confusionMatrix(as.factor(ifelse(XGboost_prediction>0.5,1,0)),y_test,positive="1") #Display confusion matrix

####ROC Curve
XGboost_ROC_prediction <- prediction(XGboost_prediction, y_test) #Calculate errors
XGboost_ROC_testing <- performance(XGboost_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(XGboost_ROC_testing) #Plot ROC curve

####AUC
auc.tmp <- performance(XGboost_ROC_prediction,"auc") #Create AUC data
XGboost_auc_testing <- as.numeric(auc.tmp@y.values) #Calculate AUC
XGboost_auc_testing #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value
# AUC value is 0.88

#### Lift chart
plotLift(XGboost_prediction, y_test, cumulative = TRUE, n.buckets = 10) # Plot Lift chart

### An alternative way is to plot a Lift curve not by buckets, but on all data points
Lift_XGboost <- performance(XGboost_ROC_prediction,"lift","rpp")
plot(Lift_XGboost)

# XgBoost Hyper-parameter tuning
control<-trainControl(method="repeatedcv",
                      number=5,repeats=1,search="random")

start_time_xg<-Sys.time()

xg_Boost<-train(y=y_train,
                x=x_train,method="xgbTree",metric="Accuracy",
                tuneLength=5,trControl=control)

end_time_xg<-Sys.time()

#Print summary of all Cross validations
print(xg_Boost)
#Print hyperparameters best tuned model
xg_Boost$bestTune

#Calculate Probability and build confusion matrix
XGboost_prediction<-predict(xg_Boost,newdata=x_test,type="prob") #Predict classification (for confusion matrix)
confusionMatrix(as.factor(ifelse(XGboost_prediction[,2]>0.5,1,0)),y_test,positive="1") #Display confusion matrix

####ROC Curve
XGboost_ROC_prediction <- prediction(XGboost_prediction[,2], y_test) #Calculate errors
XGboost_ROC_testing <- performance(XGboost_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(XGboost_ROC_testing) #Plot ROC curve

####AUC
auc.tmp <- performance(XGboost_ROC_prediction,"auc") #Create AUC data
XGboost_auc_testing <- as.numeric(auc.tmp@y.values) #Calculate AUC
XGboost_auc_testing #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value
beep()
# After hyperparameter tuning AUC is 0.898

#### Ensemble using weighted average ####

pred_avg<-(forest_probabilities*0.1+XGboost_prediction*0.9)

confusionMatrix(as.factor(ifelse(pred_avg[,2]>0.5,1,0)),y_test,positive="1") #Display confusion matrix

####ROC Curve
Ensemble_prediction <- prediction(pred_avg[,2], y_test) #Calculate errors
Ensemble_testing <- performance(Ensemble_prediction,"tpr","fpr") #Create ROC curve data
plot(Ensemble_testing) #Plot ROC curve

####AUC
auc.tmp <- performance(Ensemble_prediction,"auc") #Create AUC data
ensemble_auc_testing <- as.numeric(auc.tmp@y.values) #Calculate AUC
ensemble_auc_testing #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value

# Final AUC after ensemble = 0.8981 

#### Logistic Regression ####
model_logistic<-glm(Purchase~., data=training, family="binomial"(link="logit"))

summary(model_logistic) 

##The model clearly has too many variables, most of which are insignificant 

## Stepwise regressions. There are three aproaches to runinng stepwise regressions: backward, forward and "both"
## In either approach we need to specify criterion for inclusion/exclusion. Most common ones: based on information criterion (e.g., AIC) or based on significance  
model_logistic_stepwiseAIC<-stepAIC(model_logistic,direction = c("both"),trace = 1) #AIC stepwise
summary(model_logistic_stepwiseAIC) 

par(mfrow=c(1,4))
plot(model_logistic_stepwiseAIC) #Error plots: similar nature to lm plots
par(mfrow=c(1,1))

###Finding predicitons: probabilities and classification
logistic_probabilities<-predict(model_logistic_stepwiseAIC,newdata=testing,type="response") #Predict probabilities
logistic_classification<-rep("1",213)
logistic_classification[logistic_probabilities<0.5]="0" #Predict classification using 0.6073 threshold. Why 0.6073 - that's the average probability of being retained in the data. An alternative code: logistic_classification <- as.integer(logistic_probabilities > mean(testing$Retained.in.2012. == "1"))
logistic_classification<-as.factor(logistic_classification)

###Confusion matrix  
confusionMatrix(logistic_classification,testing$Purchase,positive = "1") #Display confusion matrix

####ROC Curve
logistic_ROC_prediction <- prediction(logistic_probabilities, testing$Purchase)
logistic_ROC <- performance(logistic_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(logistic_ROC) #Plot ROC curve

####AUC (area under curve)
auc.tmp <- performance(logistic_ROC_prediction,"auc") #Create AUC data
logistic_auc_testing <- as.numeric(auc.tmp@y.values) #Calculate AUC
logistic_auc_testing #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value

#### Lift chart
plotLift(logistic_probabilities, testing$Purchase, cumulative = TRUE, n.buckets = 10) # Plot Lift chart

# AUC = 0.896

#### Decision Tree ####
ctree_tree<-ctree(Purchase~.,data=training) #Run ctree on training data
plot(ctree_tree, gp = gpar(fontsize = 8)) #Plotting the tree (adjust fontsize if needed)

ctree_probabilities<-predict(ctree_tree,newdata=testing,type="prob") #Predict probabilities
ctree_classification<-rep("1",213)
ctree_classification[ctree_probabilities[,2]<0.5]="0" #Predict classification using 0.6073 threshold. Why 0.6073 - that's the average probability of being retained in the data. An alternative code: logistic_classification <- as.integer(logistic_probabilities > mean(testing$Retained.in.2012. == "1"))
ctree_classification<-as.factor(ctree_classification)

###Confusion matrix  
confusionMatrix(ctree_classification,testing$Purchase,positive = "1")

####ROC Curve
ctree_probabilities_testing <-predict(ctree_tree,newdata=testing,type = "prob") #Predict probabilities
ctree_pred_testing <- prediction(ctree_probabilities_testing[,2], testing$Purchase) #Calculate errors
ctree_ROC_testing <- performance(ctree_pred_testing,"tpr","fpr") #Create ROC curve data
plot(ctree_ROC_testing) #Plot ROC curve

####AUC (area under curve)
auc.tmp <- performance(ctree_pred_testing,"auc") #Create AUC data
ctree_auc_testing <- as.numeric(auc.tmp@y.values) #Calculate AUC
ctree_auc_testing #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value

#### Lift chart
plotLift(ctree_probabilities[,2],  testing$Purchase, cumulative = TRUE, n.buckets = 10) # Plot Lift chart

# AUC = 0.893

#### Support Vector Machine ####
model_svm <- svm(Purchase ~., data=training, probability=TRUE)
summary(model_svm)

svm_probabilities<-attr(predict(model_svm,newdata=testing, probability=TRUE), "prob")
svm_prediction<-svm_probabilities[,1]

svm_classification<-rep("1",213)
svm_classification[svm_prediction<0.5]="0" 
svm_classification<-as.factor(svm_classification)
confusionMatrix(svm_classification,testing$Purchase,positive = "1")

####ROC Curve
svm_ROC_prediction <- prediction(svm_prediction, testing$Purchase) #Calculate errors
svm_ROC_testing <- performance(svm_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(svm_ROC_testing) #Plot ROC curve

####AUC
auc.tmp <- performance(svm_ROC_prediction,"auc") #Create AUC data
svm_auc_testing <- as.numeric(auc.tmp@y.values) #Calculate AUC
svm_auc_testing #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value

#### Lift chart
plotLift(svm_prediction, testing$Purchase, cumulative = TRUE, n.buckets = 10) # Plot Lift chart

### An alternative way is to plot a Lift curve not by buckets, but on all data points
Lift_svm <- performance(svm_ROC_prediction,"lift","rpp")
plot(Lift_svm)

# AUC = 0.893
## End of Code ##