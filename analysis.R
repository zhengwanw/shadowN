
# libraries ---------------------------------------------------------------
#install.packages("KernelKnn")
library(solitude)
library(e1071)
library(caret)
library(KernelKnn)
library(rlang)
library(magrittr)
library(tidyverse)
library(umap)
library(ggrepel)
library(tsne)
library(rBayesianOptimization)
library(class)
library(keras)
library(ECoL)
library(dplyr)
library(FactoMineR)
library(rgl)
library(car)
library(RColorBrewer)
library(scatterplot3d)
library(factoextra)
library(kernlab)

# read in R code ----------------------------------------------------------
source(".../ShadowN/R/shadown.R")
source(".../ShadowN/R/kernelknn.R")
source(".../R/normalization.R")
source(".../R/RcppExports.R")
source(".../ShadowN/R/utils.R")
library(Rcpp)
library(RcppArmadillo)
sourceCpp("C:/Users/ZHENG/Documents/R/ShadowN/src/distance_metrics.cpp")


# data --------------------------------------------------------------------
Data = read.csv(".../dataset/indian_liver_patient.csv")
Data = read.csv(".../dataset/pima.csv")
Data = read.csv(".../dataset/heart_failure_clinical_records_dataset.csv")
Data = read.csv(".../dataset/parkinson.csv")
Data = read.csv(".../dataset/ionosphere.csv")
Data = read.csv("../dataset/wisconsin.csv")
Data = read.csv(".../dataset/vowel.csv")

Data = Data %>% filter(y %in% c("hAd","hId"))
head(Data)
dim(Data)
(tab = table(Data[,1]))
(tan1 = tab[1]/tab[2])
1/tan1
Data$y = as.numeric(as.factor(Data$y))
Data$y = as.factor(Data$y)

out = shadown(y~.,Data,knn=5, p=0.01,nshadow=(ncol(Data)-1))
# smoothing border --------------------------------------------------------
Data = pima

t.pred <- KernelKnn::KernelKnn(Data[,-1], NULL, Data$y, k=3,
                               threads = 8, weights_function ="gaussian", regression = TRUE)
t.pred=round(t.pred,0)
data_clean = Data[Data$y == t.pred,]
tune.res = kernlab::sigest(y~.,data_clean, scale=TRUE)
kpca.res <- kernlab::kpca(scale(data_clean[,-1]),kernel = "rbfdot", kpar=list(sigma=tune.res[2]))
ruiseki = cumsum(eig(kpca.res)/sum(eig(kpca.res)))
len = 1+length(ruiseki[ruiseki<=0.9])
var.kpca = rotated(kpca.res)[,1:len]    # returns the data projected in the (kernel) pca space
newName = paste0("kpca_",1:len, "_", names(ruiseki)[1:len])
colnames(var.kpca) = newName
head(var.kpca)

data = cbind.data.frame(var.kpca[,1:3],label=as.factor(data_clean[,1]))
colnames(data) = c("Meta.1","Meta.2","Meta.3","label")

ggplot(data, aes(x = Meta.1, y= Meta.2, shape = label, fill = label))+
  geom_point()+
  scale_fill_brewer(palette="Set2")+
  ggtitle("Pima with decision boundary smoothing")+
  #scale_fill_manual(values = c("#4e4d71","#e5cb78")) +
  #scale_shape_manual(values = c(21,22,24)) +
  scale_shape_manual(values = c(21,24)) +
  theme_classic()+
  #annotate("text", x=3, y=-1.5, label="F4:0.00\nN4:0.02\nN2:0.13\nmean:0.05") +
  theme(axis.text.x = element_text(size = 10, colour = "black", angle=0),
        axis.text.y = element_text(size = 10,colour = "black"),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12),
        strip.text.x = element_text(size = 10))

# class overlap complexity measures ---------------------------------------
measures = c("overlapping.F4.mean","neighborhood.N4.mean","neighborhood.N2.mean")
datas = c("liver", "pima", "heart", "parkinson","ionosphere", "wisconsin", "vowel")
complexity.res = matrix(0,7,3)
colnames(complexity.res) = measures
rownames(complexity.res) = datas
complixity_measure = as.matrix(complexity(as.factor(y) ~ ., Data))
complexity.res[7,] = complixity_measure[measures,]
complexity.res
#
# plot of the original data (PCA) ----------------------------------------

pca.res = Data[,-1] %>% FactoMineR::PCA(graph = FALSE)
data = cbind.data.frame(pca.res$ind$coord[,1:2],label=as.character(Data[,1]))
colnames(data) = c("Meta.1","Meta.2","label")

ggplot(data, aes(x = Meta.1, y= Meta.2, shape = label, fill = label))+
  geom_point()+
  scale_fill_brewer(palette="Set2")+
  ggtitle("Japanese vowels")+
  #scale_fill_manual(values = c("#4e4d71","#e5cb78")) +
  #scale_shape_manual(values = c(21,22,24)) +
  scale_shape_manual(values = c(21,24)) +
  theme_classic()+
  #annotate("text", x=3, y=-1.5, label="F4:0.00\nN4:0.02\nN2:0.13\nmean:0.05") +
  theme(axis.text.x = element_text(size = 10, colour = "black", angle=0),
        axis.text.y = element_text(size = 10,colour = "black"),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12),
        strip.text.x = element_text(size = 10))



# plot of the original data (KPCA) ----------------------------------------
#install.packages("kernlab", dependencies=TRUE)
#tunning for sigma
library(kernlab)
library(e1071)

Data = read.delim("clipboard")
Data$y[Data$y==0]="diabetic"
Data$y[Data$y==1]="non-diabetic"

tune.res = kernlab::sigest(y~.,Data, scale=TRUE)
kpca.res <- kernlab::kpca(scale(Data[,-1]),kernel = "rbfdot", kpar=list(sigma=tune.res[2]))
ruiseki = cumsum(eig(kpca.res)/sum(eig(kpca.res)))
len = 1+length(ruiseki[ruiseki<=0.9])
var.kpca = rotated(kpca.res)[,1:len]    # returns the data projected in the (kernel) pca space
newName = paste0("Meta.",1:len)
colnames(var.kpca) = newName
head(var.kpca)
X = var.kpca[,1:2]
y = as.factor(Data$y)

rbfsvm <- kernlab::ksvm(X, y,kernel="rbfdot",sigma=tune.res[2])
rbfsvm
kernlab::plot(rbfsvm, data = X)
#points(X[SVindex(rbfsvm),], pch = 5, cex = 1)


library(rgl)
library(car)
library(RColorBrewer)
library(scatterplot3d)
data = cbind.data.frame(var.kpca[,1:3],label=as.factor(Data[,1]))
colnames(data) = c("Meta.1","Meta.2","Meta.3","label")
Meta.1 = data$Meta.1
Meta.2 = data$Meta.2
Meta.3 = data$Meta.3

colors = brewer.pal(n=2, name="Set2")
scatter3d(Meta.1, Meta.2, Meta.3, sphere.size=1.4,
          groups = data$label,
          fit="smooth",
          surface = FALSE, grid = FALSE,
          ellipsoid = TRUE,level = 0.6, ellipsoid.alpha=0.2,
          surface.col = colors)

ggplot(data, aes(x = Meta.1, y= Meta.2, shape = label, fill = label))+
  geom_point()+
  scale_fill_brewer(palette="Set2")+
  #ggtitle("cleaned Heart by CL")+
  ggtitle("Pima")+
  #scale_fill_manual(values = c("#4e4d71","#e5cb78")) +
  #scale_shape_manual(values = c(21,22,24)) +
  scale_shape_manual(values = c(21,24)) +
  theme_classic()+
  #annotate("text", x=3, y=-1.5, label="F4:0.00\nN4:0.02\nN2:0.13\nmean:0.05") +
  theme(axis.text.x = element_text(size = 10, colour = "black", angle=0),
        axis.text.y = element_text(size = 10,colour = "black"),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12),
        strip.text.x = element_text(size = 10))


# plot of data with noise (PCA) -------------------------------------------
Data = read.delim("clipboard",row.names = 1)
randomSel = sample(1:nrow(Data),20,replace = F)
class_type = unique(Data$y)
ntype = length(class_type)
for (i in randomSel) {
  Data$y[i] = class_type[class_type!=Data$y[i]][round(runif(1,min=1,max=(ntype-1)))]
}
pca.res = Data[,-1] %>% FactoMineR::PCA(graph = FALSE)
data = cbind.data.frame(pca.res$ind$coord[,1:2],label=as.character(Data[,1]))
colnames(data) = c("Meta.1","Meta.2","label")

ggplot(data, aes(x = Meta.1, y= Meta.2, shape = label, fill = label))+
  geom_point()+
  scale_fill_brewer(palette="Set2")+
  ggtitle("Japanese vowels")+
  #scale_fill_manual(values = c("#4e4d71","#e5cb78")) +
  #scale_shape_manual(values = c(21,22,24)) +
  scale_shape_manual(values = c(21,24)) +
  theme_classic()+
  #annotate("text", x=3, y=-1.5, label="F4:0.00\nN4:0.02\nN2:0.13\nmean:0.05") +
  theme(axis.text.x = element_text(size = 10, colour = "black", angle=0),
        axis.text.y = element_text(size = 10,colour = "black"),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12),
        strip.text.x = element_text(size = 10))

original.label = Data$y
Data$y=as.numeric(as.factor(Data$y))-1
#find the best K
knn_holdout <- function(k){
  t.pred <- KernelKnn::KernelKnn(Data[,-1], NULL, Data$y, k=k,
                                 threads = 8, weights_function ="gaussian", regression = TRUE)
  Pred <- sum(diag(table(Data$y, round(t.pred,0))))/nrow(Data)
  list(Score=Pred, Pred=Pred)}

opt_knn <- BayesianOptimization(knn_holdout,
                                bounds=list(k=c(3L,20L)),
                                init_points=10, n_iter=10, acq='ei', 
                                eps=0.0, verbose=TRUE)
opt_knn$Best_Par=10
#noise detection
out = fmf(y~.,Data,knn=opt_knn$Best_Par, p=0.01,nshadow=(ncol(Data)-1))
Data$y = original.label
Data$y[randomSel] = "N"
noise_score = out$noise_score[1:nrow(Data)]
data = cbind.data.frame(pca.res$ind$coord[,1:2],label=as.character(Data[,1]),
                        score = noise_score)
colnames(data) = c("Meta.1","Meta.2","label","score")

#plot
states <- rownames(Data)
selected_states <- as.character(out$remIdx)
states[!states %in% selected_states] <- ""
states[out$remIdx]=Data$y[out$remIdx]

ggplot(data, aes(x = Meta.1, y= Meta.2))+
  geom_point(aes(shape = label,col = score))+
  scale_color_viridis_c(option = "C")+
  #ggtitle("Noise level:0.2")+
  xlab("Meta.1")+
  ylab("Meta.2")+
  geom_text_repel(aes(label = states), size = 3, colour = "black")+
  theme_classic()

beta=0.5
tp =  sum(out$remIdx %in% randomSel)
(Precision= tp/length(out$remIdx))
(Recall =  tp/length(randomSel))
((1+beta^2)*Precision*Recall)/((beta^2*Precision)+Recall)

# plot of the original data (tsne) ----------------------------------------
per = c(5,seq(10,50,by=10))
colors = rainbow(length(unique(Data[,1])))
names(colors) = unique(Data[,1])
ecb = function(x,y){ plot(x,t='n'); text(x,labels=Data[,1], col=colors[Data[,1]]) }

for(i in 1:length(per)){
  cat(per[i],"\n")
  tsne_inters = tsne(Data[,-1]%>% scale(),epoch_callback = ecb,perplexity=per[i])
}

tsne.res = Data[,-1] %>% scale() %>%tsne(perplexity=5)
head(tsne.res)

data = cbind.data.frame(tsne.res,label=as.character(Data[,1]))
colnames(data) = c("Dim.1","Dim.2","label")
ggplot(data, aes(x = Dim.1, y= Dim.2, shape = label, fill = label))+
  geom_point()+
  scale_fill_brewer(palette="Set2")+
  #scale_fill_manual(values = c("#4e4d71","#e5cb78")) +
  #scale_shape_manual(values = c(21,22,24)) +
  scale_shape_manual(values = c(21,22)) +
  theme_classic()

# plot of the original data (UMAP) ----------------------------------------
set.seed(123)
Data=liver
validationIndex <- createDataPartition(Data$y, p=0.70, list=FALSE)
train <- Data[validationIndex,] # 70% of data to training
test <- Data[-validationIndex,] # remaining 30% for test

knn_holdout <- function(k){
t.pred <- knn(train[,-1], test[,-1], cl=train$y, k=k)
Pred <- sum(diag(table(test$y, t.pred)))/nrow(test)
list(Score=Pred, Pred=Pred)}

opt_knn <- BayesianOptimization(knn_holdout,
                              bounds=list(k=c(2L,20L)),
                              init_points=20, n_iter=10, acq='ei', 
                              eps=0.0, verbose=TRUE)

custom.config = umap.defaults
custom.config$n_neighbors=opt_knn$Best_Par
custom.config$random_state = 123
umap.res = Data[,-1] %>% scale() %>%umap(config = custom.config)
head(umap.res$layout)

X = umap.res$layout
y = Data$y
rbfsvm <- kernlab::ksvm(X, y,kernel="rbfdot")
plot(rbfsvm, data = X)

data = cbind.data.frame(umap.res$layout,label=as.character(Data[,1]))

colnames(data) = c("Meta.1","Meta.2","label")
ggplot(data, aes(x = Meta.1, y= Meta.2, shape = label, fill = label))+
  geom_point()+
  scale_fill_brewer(palette="Set2")+
  ggtitle("Japanese Vowels")+
  #scale_fill_manual(values = c("#4e4d71","#e5cb78")) +
  #scale_shape_manual(values = c(21,22,24)) +
  scale_shape_manual(values = c(21,22)) +
  theme_classic()



# plot of the original data (AutoEncoder) ---------------------------------
dataset2 = cbind(y=as.factor(as.character(Data[,1])),Data[,-1])
str(dataset2)
dataset2 <- dataset2 %>% mutate_if(is.factor, as.numeric)

## Matrix型へ変換
dataset2 <-as.matrix(dataset2)
dimnames(dataset2) <- NULL

#目的変数yのダミー化
x_train = dataset2[,-1]
y_train = dataset2[,1] - 1
label_d <- to_categorical(y_train)
ncol(label_d)
# set model
units_num = ncol(x_train)
model <- keras_model_sequential()
model %>%
  #layer_dense(units = units_num, activation = "relu", ) %>%
  layer_dense(units = floor(units_num/2), activation = "relu", input_shape = c(units_num)) %>%
  layer_dense(units = 2, activation = "relu",name = "bottleneck") %>%
  layer_dense(units = 2,activation = 'softmax')

summary(model) #確認用

# compile model
model %>% compile(loss = 'categorical_crossentropy',
                  optimizer = 'adam',
                  metrics = 'accuracy'
)

m = apply(x_train, 2, mean)
s = apply(x_train, 2, sd)
x_train = scale(x_train, center=m, scale=s)

## 学習
history <- 
  model %>% fit(x_train, 
                label_d, 
                epochs = 100,
                batch_size = 10, 
                validation_split = 0.25)
history #確認用

# fit model
pred <- 
  model %>% 
  predict(x_train) %>% k_argmax()
table(y_train,as.numeric(pred))

## 正答率
model %>% 
  evaluate(x_train, label_d)


# extract the bottleneck layer
intermediate_layer_model <- keras_model(inputs = model$input, 
                                        outputs = get_layer(model, "bottleneck")$output)
var.auto <- predict(intermediate_layer_model, x_train)
head(var.auto)
newName = paste0("auto_",1:2, "_", c("v1","v2"))
colnames(var.auto) = newName
data = cbind.data.frame(var.auto,label=as.character(Data[,1]))

colnames(data) = c("Dim.1","Dim.2","label")
ggplot(data, aes(x = Dim.1, y= Dim.2, shape = label, fill = label))+
  geom_point()+
  scale_fill_brewer(palette="Set2")+
  #scale_fill_manual(values = c("#4e4d71","#e5cb78")) +
  #scale_shape_manual(values = c(21,22,24)) +
  scale_shape_manual(values = c(21,22)) +
  theme_classic()


# noise injection ---------------------------------------------------------
Data = read.csv("D:/kenkyu/noise_detection_with_shadow_instances/dataset/clean_parkinson.csv")
table(Data$y)

noise_rates = seq(0.01,0.40,0.01)
n_noise = floor(nrow(Data)*noise_rates)
Data$y = as.numeric(as.factor(Data$y))
label_origi = Data$y
class_type = unique(label_origi)
ntype = length(class_type)
beta = 0.5

det_res = matrix(0,length(noise_rates),3)
rownames(det_res) = noise_rates
colnames(det_res) = c("precision", "recall", "Fscore")

for (ss in 1:length(noise_rates)) {
  Data$y = label_origi
  randomSel = sample(1:nrow(Data),n_noise[ss],replace = FALSE)
  for (i in randomSel) {
    Data$y[i] = class_type[class_type!=Data$y[i]][round(runif(1,min=1,max=(ntype-1)))]
  }
  #knn_holdout <- function(k){
  #    t.pred <- KernelKnn::KernelKnn(Data[,-1], NULL, Data$y, k=k,
                          #threads = 8, weights_function ="gaussian", regression = TRUE)
   #  Pred <- sum(diag(table(Data$y, round(t.pred,0))))/nrow(Data)
  #    list(Score=Pred, Pred=Pred)}
    
  #opt_knn <- BayesianOptimization(knn_holdout,
  #                                  bounds=list(k=c(5L,20L)),
  #                                  init_points=10, n_iter=10, acq='ei', 
  #                                  eps=0.0, verbose=TRUE)
  #noise detection
  #out = fmf(y~.,Data,knn=opt_knn$Best_Par, p=0.01,nshadow=(ncol(Data)-1))
  out = fmf(y~.,Data,knn = 15, p=0.01,nshadow=(ncol(Data)-1))
  tp =  sum(out$remIdx %in% randomSel)
  det_res[ss,1] = Precision= tp/length(out$remIdx)
  det_res[ss,2] = Recall =  tp/length(randomSel)
  det_res[ss,3] = ((1+beta^2)*Precision*Recall)/((beta^2*Precision)+Recall)
}
det_res
write.csv(det_res,"D:/kenkyu/noise_detection_with_shadow_instances/result/sn_cleaned_liver2.csv")

# line plot for detection rates -------------------------------------------
library(reshape2)
det_res = read.delim("clipboard",row.names = 1)
#det_res2 = as.matrix(det_res[,c(1,4,7,10)])
#det_res3 = as.matrix(det_res[,c(2,5,8,11)])
#det_res4 = as.matrix(det_res[,c(3,6,9,12)])
colnames(det_res)
det_res2 = as.matrix(det_res[,c(1,10)])
det_res3 = as.matrix(det_res[,c(2,11)])
det_res4 = as.matrix(det_res[,c(3,12)])


colnames(det_res2)
colnames(det_res2)=colnames(det_res3)=colnames(det_res4)=c("ShadowN","CL_GTBoost")
#colnames(det_res2)=colnames(det_res3)=colnames(det_res4)=c("ShadowN","CL_SVM","CL_RF","CL_GTBoost")
noise_rates = seq(0.01,0.40,0.01)
rownames(det_res2) = rownames(det_res3) = rownames(det_res4) = noise_rates

det_melt2 = melt(det_res2)
det_melt3 = melt(det_res3)
det_melt4 = melt(det_res4)

det_melt2 = cbind.data.frame(det_melt2, metric = rep("Precision",nrow(det_melt2)))
#det_melt3 = cbind.data.frame(det_melt3, metric = rep("Recall",nrow(det_melt3)))
#det_melt4 = cbind.data.frame(det_melt4, metric = rep("F-score",nrow(det_melt4)))

colnames(det_melt2) = colnames(det_melt3) = colnames(det_melt4) =c("noise_level", "method", "value", "metric")

#det_melt = rbind.data.frame(det_melt2, det_melt3, det_melt4)
library(ggplot2)
ggplot(det_melt2, aes(x = noise_level, y = value,color=method))+ 
  scale_color_brewer(palette="Dark2")+
  xlab("Noise level")+
  #ylab("Precision")+
  #ylab("F-score")+
  ggtitle("cleaned Par by CL")+
  geom_point(aes(shape = method))+
  geom_line()+
 # stat_smooth(se = FALSE)+
  #facet_wrap( ~ metric)+
  theme_classic()

# noise score scatter plot --------------------------------------------------------

# n noise injection  ------------------------------------------------------
noise_rates
n_noise
ss=4
Data$y = label_origi
randomSel = sample(1:nrow(Data),n_noise[ss],replace = F)
randomSel = sample(1:nrow(Data),10,replace = F)
for (i in randomSel) {
  Data$y[i] = class_type[class_type!=Data$y[i]][round(runif(1,min=1,max=(ntype-1)))]
}



knn_holdout <- function(k){
  t.pred <- KernelKnn::KernelKnn(Data[,-1], NULL, Data$y, k=k,
                                 threads = 8, weights_function ="gaussian", regression = TRUE)
  Pred <- sum(diag(table(Data$y, round(t.pred,0))))/nrow(Data)
  list(Score=Pred, Pred=Pred)}

opt_knn <- BayesianOptimization(knn_holdout,
                                bounds=list(k=c(5L,20L)),
                                init_points=20, n_iter=10, acq='ei', 
                                eps=0.0, verbose=TRUE)
#noise detection
out = fmf(y~.,Data,knn=opt_knn$Best_Par, p=0.01,nshadow=(ncol(Data)-1))

#Data$y[randomSel] = "N"
#KPCA
tune.res = kernlab::sigest(y~.,Data, scale=TRUE)
kpca.res <- kernlab::kpca(scale(Data[,-1]),kernel = "rbfdot", kpar=list(sigma=tune.res[2]))
ruiseki = cumsum(eig(kpca.res)/sum(eig(kpca.res)))
len = 1+length(ruiseki[ruiseki<=0.9])
var.kpca = rotated(kpca.res)[,1:len]    # returns the data projected in the (kernel) pca space
newName = paste0("Meta.",1:len)
colnames(var.kpca) = newName
class_type = unique(Data$y)
ntype = length(class_type)
#
nrow(Data)
length(out$noise_score)
data = cbind.data.frame(var.kpca[,1:2],label=as.character(Data[,1]),
                        score = out$noise_score[1:nrow(Data)])
                        #score = out$noise_score[-c((length(out$noise_score)-ntype+1):length(out$noise_score))])
colnames(data) = c("Dim.1","Dim.2","label","score")

#plot
states <- rownames(Data)
selected_states <- as.character(out$remIdx)
states[!states %in% selected_states] <- ""
states[out$remIdx]=Data$y[out$remIdx]

ggplot(data, aes(x = Dim.1, y= Dim.2))+
  geom_point(aes(shape = label,col = score))+
  scale_color_viridis_c(option = "C")+
  ggtitle("Noise level:0.2")+
  xlab("")+
  ylab("")+
  #scale_shape_manual(values = c(21,22,24)) +
  geom_text_repel(aes(label = states), size = 3, colour = "black")+
  theme_classic()

# noise score scatter plot for original data ------------------------------
knn_holdout <- function(k){
  t.pred <- KernelKnn::KernelKnn(Data[,-1], NULL, Data$y, k=k,
                                 threads = 8, weights_function ="gaussian", regression = TRUE)
  Pred <- sum(diag(table(Data$y, round(t.pred,0))))/nrow(Data)
  list(Score=Pred, Pred=Pred)}

opt_knn <- BayesianOptimization(knn_holdout,
                                bounds=list(k=c(5L,20L)),
                                init_points=20, n_iter=10, acq='ei', 
                                eps=0.0, verbose=TRUE)
#noise detection
out = fmf(y~.,Data,knn=opt_knn$Best_Par, p=0.01,nshadow=(ncol(Data)-1))
length(out$remIdx)
out$remIdx
#KPCA
tune.res = kernlab::sigest(y~.,Data, scale=TRUE)
kpca.res <- kernlab::kpca(scale(Data[,-1]),kernel = "rbfdot", kpar=list(sigma=tune.res[2]))
var.kpca = rotated(kpca.res)  # returns the data projected in the (kernel) pca space
colnames(var.kpca) = paste0("Meta.",1:ncol(var.kpca))
class_type = unique(Data$y)
ntype = length(class_type)
#
data = cbind.data.frame(var.kpca[,1:2],label=as.character(Data[,1]),
                        score = out$noise_score[1:nrow(Data)])
colnames(data) = c("Meta.1","Meta.2","label","score")
data$label[data$label==0]="non-liver"
data$label[data$label==1]="liver"
#plot
ggplot()+
  geom_point(data = data, 
             mapping = aes(x = Meta.1, y= Meta.2,shape = label,col = score))+
  geom_point(data = data[out$remIdx,],
             mapping = aes(x = Meta.1, y= Meta.2),
             shape = 5,size=2)+
  scale_color_viridis_c(option = "C")+
  ggtitle("Liver")+
  xlab("")+
  ylab("")+
  #scale_shape_manual(values = c(21,22,24)) +
  theme_classic()

out$remIdx
#cl
score = read.delim("clipboard",row.names = 1)
data = cbind.data.frame(var.kpca[,1:2],label=as.character(Data[,1]),
                        score = score[,3])
colnames(data) = c("Meta.1","Meta.2","label","score")
data$label[data$label==0]="non-liver"
data$label[data$label==1]="liver"

id = read.delim("clipboard")
remIdx = id[,4][!is.na(id[,4])]
remIdx 
#plot
ggplot()+
  geom_point(data = data, 
             mapping = aes(x = Meta.1, y= Meta.2,shape = label,col = score))+
  geom_point(data = data[remIdx,],
             mapping = aes(x = Meta.1, y= Meta.2),
             shape = 5,size=2)+
  scale_color_viridis_c(option = "C")+
  ggtitle("Wisconsin -- CL_GTBoost")+
  xlab("")+
  ylab("")+
  #scale_shape_manual(values = c(21,22,24)) +
  theme_classic()

# histgram ----------------------------------------------------------------
library(reshape2)
library(ggplot2)
library(ggsci)
remIdx_svm = id[,2][!is.na(id[,2])]
remIdx_rf = id[,3][!is.na(id[,3])]
remIdx_xg =id[,4][!is.na(id[,4])]

Data$y[Data$y==0]="non-liver"
Data$y[Data$y==1]="liver"

#sn
remScore = cbind.data.frame(id = out$remIdx,score = out$noise_score[out$remIdx])
Score = cbind.data.frame(id = out$Idx, score = out$noise_score[out$Idx])

scores.origin = rbind.data.frame(remScore,Score)
scores.origin = cbind.data.frame(scores.origin,
                                 type=c(Data$y[out$remIdx],Data$y[out$Idx]),
                                 label=c(rep("removed",nrow(remScore)),rep("clean",nrow(Score))))

ggplot(scores.origin, aes(x = score, fill = label, color=label))+
  geom_histogram(position = "identity", alpha=0.5)+
  #geom_density(alpha=0.7)+
  xlab("Noise score")+
  ylab("Number of instances")+
  scale_color_brewer(palette="Dark2")+
  scale_fill_brewer(palette="Dark2")+
  facet_wrap( ~ type)+
  theme_classic()

#svm
clean.id = score[,1][!(1:nrow(score)) %in% remIdx_svm]
rank =c(remIdx_svm, (1:nrow(score)) [!(1:nrow(score)) %in% remIdx_svm])
scores.origin = c(score[remIdx_svm,1],clean.id)
length(scores.origin)
clean.num = length(clean.id)

scores.origin = cbind.data.frame(score  = scores.origin,
                                 type=Data$y[rank],
                                 label=c(rep("removed",length(remIdx_svm)),
                                         rep("clean",clean.num)))

ggplot(scores.origin, aes(x = score, fill = label, color=label))+
  geom_histogram(position = "identity", alpha=0.5)+
  xlab("Noise score")+
  ylab("Number of instances")+
  scale_color_brewer(palette="Dark2")+
  scale_fill_brewer(palette="Dark2")+
  facet_wrap( ~ type)+
  theme_classic()

#rf
clean.id = score[,1][!(1:nrow(score)) %in% remIdx_rf]
rank =c(remIdx_rf, (1:nrow(score)) [!(1:nrow(score)) %in% remIdx_rf])
scores.origin = c(score[remIdx_rf,1],clean.id)
length(scores.origin)
clean.num = length(clean.id)

scores.origin = cbind.data.frame(score  = scores.origin,
                                 type=Data$y[rank],
                                 label=c(rep("removed",length(remIdx_rf)),
                                         rep("clean",clean.num)))

ggplot(scores.origin, aes(x = score, fill = label, color=label))+
  geom_histogram(position = "identity", alpha=0.5)+
  xlab("Noise score")+
  ylab("Number of instances")+
  scale_color_brewer(palette="Dark2")+
  scale_fill_brewer(palette="Dark2")+
  facet_wrap( ~ type)+
  theme_classic()

#xg
clean.id = score[,1][!(1:nrow(score)) %in% remIdx_xg]
rank =c(remIdx_xg, (1:nrow(score)) [!(1:nrow(score)) %in% remIdx_xg])
scores.origin = c(score[remIdx_xg,1],clean.id)
length(scores.origin)
clean.num = length(clean.id)

scores.origin = cbind.data.frame(score  = scores.origin,
                                 type=Data$y[rank],
                                 label=c(rep("removed",length(remIdx_xg)),
                                         rep("clean",clean.num)))

ggplot(scores.origin, aes(x = score, fill = label, color=label))+
  geom_histogram(position = "identity", alpha=0.5)+
  xlab("Noise score")+
  ylab("Number of instances")+
  scale_color_brewer(palette="Dark2")+
  scale_fill_brewer(palette="Dark2")+
  facet_wrap( ~ type)+
  theme_classic()
# macro-F -----------------------------------------------------------------
F <-function(ct){
  ct<-ct+0.000001
  if(ncol(ct)==1){
    zero<-matrix(0.000001,2,1)
    names<-c("A","B")
    diff<-setdiff(names,colnames(ct))
    if(is.element("0",diff)){
      ct<-cbind(zero,ct)
    }else{
      ct<-cbind(ct,zero)
    }
  }
  pre1<-ct[1,1]/sum(ct[,1])
  rec1<-ct[1,1]/sum(ct[1,])
  pre2<-ct[2,2]/sum(ct[,2])
  rec2<-ct[2,2]/sum(ct[2,])
  pre<-mean(c(pre1,pre2))
  rec<-mean(c(rec1,rec2))
  2*pre*rec/(pre+rec)
}

normalize<-function(x){
  return((x-min(x))/(max(x)-min(x)))
}
# accuracy after removing noise -------------------------------------------
remIdx_sn = out$remIdx

sn=svm=rf=xg = matrix(0,5,2)
colnames(sn) = colnames(svm)= colnames(rf) = colnames(xg) = c("knn", "avNNet")

scale = as.data.frame(lapply(Data[,-1],normalize))
data.scale = cbind(y=Data[,1],scale)

#original data
folds = createFolds(data.scale$y,k=5)
original = matrix(0,5,2)
colnames(original) = c("knn", "avNNet")

for(k in 1:5){
  train.data = data.scale[-folds[[k]],]
  test.data = data.scale[folds[[k]],]
  
  tuneK= e1071::tune.knn(x = train.data[,-1], y = as.factor(train.data$y), k = 1:10) %>% 
    summary()
  pre = knn(train.data[,-1],test.data[,-1],as.factor(train.data$y),k=tuneK$best.parameters)
  tab = table(pre,test.data[,1])
  original[k,1] = F(tab)
  
  modelAvnnet <- caret::train(
    y~., 
    data = train.data,                      
    method = "avNNet",                       
    trControl = trainControl(method = "cv"),  
    tuneGrid = expand.grid(size = c(4, 6, 8),
                           decay = c(0.001, 0.01),
                           bag = c(TRUE, FALSE))
  )
  Umatrix<-cbind(test.data[,1],as.character(predict(modelAvnnet,test.data[,-1])))
  tab = table(Umatrix[,1],Umatrix[,2])
  original[k,2] = F(tab)
}

original
#sn
subdata = data.scale[-remIdx_sn,]
folds = createFolds(subdata$y,k=5)

for(k in 1:5){
  train.data = subdata[-folds[[k]],]
  test.data = subdata[folds[[k]],]
  
  tuneK= e1071::tune.knn(x = train.data[,-1], y = as.factor(train.data$y), k = 1:10) %>% 
    summary()
  pre = knn(train.data[,-1],test.data[,-1],as.factor(train.data$y),k=tuneK$best.parameters)
  tab = table(pre,test.data[,1])
  sn[k,1] = F(tab)
  
  modelAvnnet <- caret::train(
    y~., 
    data = train.data,                      
    method = "avNNet",                       
    trControl = trainControl(method = "cv"),  
    tuneGrid = expand.grid(size = c(4, 6, 8),
                           decay = c(0.001, 0.01),
                           bag = c(TRUE, FALSE))
  )
  Umatrix<-cbind(test.data[,1],as.character(predict(modelAvnnet,test.data[,-1])))
  tab = table(Umatrix[,1],Umatrix[,2])
  sn[k,2] = F(tab)
}

#svm
subdata = data.scale[-remIdx_svm,]
folds = createFolds(subdata$y,k=5)

for(k in 1:5){
  train.data = subdata[-folds[[k]],]
  test.data = subdata[folds[[k]],]
  
  tuneK= e1071::tune.knn(x = train.data[,-1], y = as.factor(train.data$y), k = 1:10) %>% 
    summary()
  pre = knn(train.data[,-1],test.data[,-1],as.factor(train.data$y),k=tuneK$best.parameters)
  tab = table(pre,test.data[,1])
  svm[k,1] = F(tab)
  
  modelAvnnet <- caret::train(
    y~., 
    data = train.data,                      
    method = "avNNet",                       
    trControl = trainControl(method = "cv"),  
    tuneGrid = expand.grid(size = c(4, 6, 8),
                           decay = c(0.001, 0.01),
                           bag = c(TRUE, FALSE))
  )
  Umatrix<-cbind(test.data[,1],as.character(predict(modelAvnnet,test.data[,-1])))
  tab = table(Umatrix[,1],Umatrix[,2])
  svm[k,2] = F(tab)
}
svm
#rf
subdata = data.scale[-remIdx_rf,]
folds = createFolds(subdata$y,k=5)

for(k in 1:5){
  train.data = subdata[-folds[[k]],]
  test.data = subdata[folds[[k]],]
  
  tuneK= e1071::tune.knn(x = train.data[,-1], y = as.factor(train.data$y), k = 1:10) %>% 
    summary()
  pre = knn(train.data[,-1],test.data[,-1],as.factor(train.data$y),k=tuneK$best.parameters)
  tab = table(pre,test.data[,1])
  rf[k,1] = F(tab)
  
  modelAvnnet <- caret::train(
    y~., 
    data = train.data,                      
    method = "avNNet",                       
    trControl = trainControl(method = "cv"),  
    tuneGrid = expand.grid(size = c(4, 6, 8),
                           decay = c(0.001, 0.01),
                           bag = c(TRUE, FALSE))
  )
  Umatrix<-cbind(test.data[,1],as.character(predict(modelAvnnet,test.data[,-1])))
  tab = table(Umatrix[,1],Umatrix[,2])
  rf[k,2] = F(tab)
}
rf
#xgboot
subdata = data.scale[-remIdx_xg,]
folds = createFolds(subdata$y,k=5)
for(k in 1:5){
  train.data = subdata[-folds[[k]],]
  test.data = subdata[folds[[k]],]
  
  tuneK= e1071::tune.knn(x = train.data[,-1], y = as.factor(train.data$y), k = 1:10) %>% 
    summary()
  pre = knn(train.data[,-1],test.data[,-1],as.factor(train.data$y),k=tuneK$best.parameters)
  tab = table(pre,test.data[,1])
  xg[k,1] = F(tab)
  
  modelAvnnet <- caret::train(
    y~., 
    data = train.data,                      
    method = "avNNet",                       
    trControl = trainControl(method = "cv"),  
    tuneGrid = expand.grid(size = c(4, 6, 8),
                           decay = c(0.001, 0.01),
                           bag = c(TRUE, FALSE))
  )
  Umatrix<-cbind(test.data[,1],as.character(predict(modelAvnnet,test.data[,-1])))
  tab = table(Umatrix[,1],Umatrix[,2])
  xg[k,2] = F(tab)
}
xg
com.res = cbind.data.frame(sn,svm,rf,xg)
com.res

t.test(com.res[,1],com.res[,3])
t.test(com.res[,1],com.res[,5])
t.test(com.res[,1],com.res[,7])

t.test(com.res[,2],com.res[,4])
t.test(com.res[,2],com.res[,6])
t.test(com.res[,2],com.res[,8])

# Detection rates with different sample sizes -----------------------------
n_noise = seq(10,100,5)
detec_rate = matrix(0,length(n_noise),100)
rownames(detec_rate) = n_noise

for(nn in 1:100){
  for(ss in 1:length(n_noise)){
    randomSel = sample(1:nrow(Data),n_noise[ss],replace = F)
    types = unique(Data$y)
    ntype = length(types)
    for (i in randomSel) {
      Data$y[i] = types[types!=Data$y[i]][round(runif(1,min=1,max=(ntype-1)))]
    }
    out = fmf(y~.,Data)
    detec_rate[ss,nn] = sum(randomSel %in% out$remIdx)/length(randomSel)
  }
}
detec_rate = detec_rate[,1:55]
detec_mat = cbind.data.frame(n_noise, detec_rate)
detec_mat = t(detec_mat)
detec_mat = detec_mat[-1,]
detec_melt = melt(detec_mat, variable.name = "n_noise")
head(detec_melt)
detec_melt$Var2=as.factor(detec_melt$Var2)

#box plot
ggplot(detec_melt,aes(x=Var2, y=value))+
  geom_boxplot(outlier.shape = NA)+
  xlab("")+
  ylab("Noise score") +
  theme_bw()+
  theme(axis.text.x = element_text(size = 10, colour = "black", angle=30),
        axis.text.y = element_text(size = 10,colour = "black"),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14))

#line graph
ggplot(detec_mat, aes(x = n_noise, y = detec_rate))+ 
  geom_line()



# histgram ----------------------------------------------------------------
ss=10
randomSel = sample(1:nrow(Data),ss,replace = F)
types = unique(Data$y)
ntype = length(types)
for (i in randomSel) {
  Data$y[i] = types[types!=Data$y[i]][round(runif(1,min=1,max=(ntype-1)))]
}
pca.res = Data[,-1] %>% FactoMineR::PCA(graph = FALSE)
data = cbind.data.frame(pca.res$ind$coord[,1:2],label=as.character(Data[,1]))
colnames(data) = c("Dim.1","Dim.2","label")
ggplot(data, aes(x = Dim.1, y= Dim.2, shape = label, fill = label))+
  geom_point()+
  scale_fill_brewer(palette="Set2")+
  scale_shape_manual(values = c(21,22)) +
  theme_classic()

knn_holdout <- function(k){
  t.pred <- KernelKnn::KernelKnn(Data[,-1], NULL, Data$y, k=k,
                                 threads = 8, weights_function ="gaussian", regression = TRUE)
  Pred <- sum(diag(table(Data$y, round(t.pred,0))))/nrow(Data)
  list(Score=Pred, Pred=Pred)}

opt_knn <- BayesianOptimization(knn_holdout,
                                bounds=list(k=c(2L,20L)),
                                init_points=20, n_iter=10, acq='ei', 
                                eps=0.0, verbose=TRUE)
#opt_knn$Best_Par=15
#visulize the origial data
out = fmf(y~.,Data,knn=opt_knn$Best_Par, p=0.05,nshadow=(ncol(Data)-1))
length(out$remIdx)

remScore = cbind.data.frame(id = out$remIdx,score = out$noise_score[out$remIdx])
Score = cbind.data.frame(id = out$Idx, score = out$noise_score[out$Idx])

scores.origin = rbind.data.frame(remScore,Score)
scores.origin = cbind.data.frame(scores.origin,
                                 type=c(Data$y[out$remIdx],Data$y[out$Idx]),
                                 label=c(rep("removed",nrow(remScore)),rep("clean",nrow(Score))))

library(reshape2)
library(ggplot2)
library(ggsci)
ggplot(scores.origin, aes(x = score, fill = label, color=label))+
  geom_histogram(position = "identity", alpha=0.5)+
  #geom_density(alpha=0.7)+
  xlab("Noise score")+
  ylab("Number of instances")+
  scale_color_brewer(palette="Dark2")+
  scale_fill_brewer(palette="Dark2")+
  facet_wrap( ~ type)+
  theme_classic()

#scatter plot
#out = fmf(y~.,Data)
target = Data$y
Data$y[randomSel] = "N"
noise_score = out$noise_score[1:nrow(Data)]
data = cbind.data.frame(pca.res$ind$coord[,1:2],label=as.character(Data[,1]),
                        score = noise_score)
colnames(data) = c("Dim.1","Dim.2","label","score")

#plot
states <- rownames(Data)
selected_states <- as.character(out$remIdx)
states[!states %in% selected_states] <- ""
states[out$remIdx]=Data$y[out$remIdx]

ggplot(data, aes(x = Dim.1, y= Dim.2))+
  geom_point(aes(shape = label,col = score))+
  scale_color_viridis_c(option = "C")+
  xlab("")+
  ylab("")+
  #scale_shape_manual(values = c(21,22,24)) +
  geom_text_repel(aes(label = states), size = 3, colour = "black")+
  theme_classic()

# boxplot of clean, removed, shadow variables -----------------------------
target = Data$y
remScore = cbind.data.frame(id = out$remIdx,score = out$noise_score[out$remIdx])
Score = cbind.data.frame(id = out$Idx, score = out$noise_score[out$Idx])

remScore = cbind.data.frame(remScore, label = target[remScore$id])
Score = cbind.data.frame(Score, label = target[Score$id])

table(target)
remScore.sub =  data.frame(dplyr::filter(remScore, label %in% "hId"))
Score.sub =  data.frame(dplyr::filter(Score, label %in% "hId"))

remScore.sort = remScore.sub[order(remScore.sub$score,decreasing = F),]
Score.sort = Score.sub[order(Score.sub$score,decreasing = F),]
out$shadow

score.f = data.frame(out$score)
nshadow=(ncol(Data)-1)

nums = c(Score.sort$id[1:20],191:200,remScore.sort$id)
score.sub  = t(score.f[nums,])

head(score.sub)
colnames(score.sub) = c(paste0("s",Score.sort$id[1:20]),
                        paste0("s",191:200),
                        paste0("s",remScore.sort$id))
data.melt = reshape2::melt(score.sub)

colnames(data.melt) = c("no.", "instance", "score")
label = c(rep("clean",20*100),rep("shadow",nshadow*100),rep("removed",length(remScore.sort$id)*100))
data.melt=cbind.data.frame(data.melt,label)

ggplot(data.melt,aes(x=instance, y=score,fill=label))+
  geom_boxplot(outlier.shape = NA)+
  xlab("")+
  ylab("Noise score") +
  ggtitle("hId")+
  scale_fill_brewer(palette="Dark2")+
  theme_classic()+
  theme(axis.text.x = element_text(size = 10, colour = "black", angle=30),
        axis.text.y = element_text(size = 10,colour = "black"),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14))

source("C:/Users/ZHENG/Documents/R/fmf3/R/plot.R")
plot(out$noise_score[-c((length(out$noise_score)-2):length(out$noise_score))],
     Data[,-1],as.numeric(as.factor(Data[,1])))


# spearman ----------------------------------------------------------------

data = read.delim("clipboard",row.names = 1)
cor.test(data$shadowN,data$F4,method = "spearman")
cor.test(data$shadowN,data$N4,method = "spearman")
cor.test(data$shadowN,data$N2,method = "spearman")
cor.test(data$shadowN,data$IR,method = "spearman")

cor.test(data$CL_SVM,data$F4,method = "spearman")
cor.test(data$CL_SVM,data$N4,method = "spearman")
cor.test(data$CL_SVM,data$N2,method = "spearman")
cor.test(data$CL_SVM,data$IR,method = "spearman")

cor.test(data$CL_RF,data$F4,method = "spearman")
cor.test(data$CL_RF,data$N4,method = "spearman")
cor.test(data$CL_RF,data$N2,method = "spearman")
cor.test(data$CL_RF,data$IR,method = "spearman")

cor.test(data$CL_GTBoost,data$F4,method = "spearman")
cor.test(data$CL_GTBoost,data$N4,method = "spearman")
cor.test(data$CL_GTBoost,data$N2,method = "spearman")
cor.test(data$CL_GTBoost,data$IR,method = "spearman")


# skewness & kurtosis -----------------------------------------------------

library(e1071)
data = read.delim("clipboard",row.names = 1)
skewness()
kurtosis()