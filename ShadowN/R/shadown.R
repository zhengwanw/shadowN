#'Fast Class Noise Detector with Multi-Factor-Based Learning
#'
#' This function computes the noise score for each observation
#' @param formula a formula describing the classification variable and the attributes to be used.
#' @param data,x data frame containing the tranining dataset to be filtered.
#' @param knn total number of nearest neighbors to be used.The default is 5.
#' @param classColumn positive integer indicating the column which contains the
#' (factor of) classes. By default, a dataframe built from 'data' using the variables indicated in 'formula' and The first column is the response variable, thus no need to define the classColumn.
#' @param iForest compute iForest score or not. The dafault is TRUE.
#' @param threads the number of cores to be used in parallel.
#' @param ... optional parameters to be passed to other methods.
#' @return an object of class \code{filter}, which is a list with four components:
#' \itemize{
#'    \item \code{cleanData} is a data frame containing the filtered dataset.
#'    \item \code{remIdx} is a vector of integers indicating the indexes for
#'    removed instances (i.e. their row number with respect to the original data frame).
#'    \item \code{noise_score} is a vector of values indicating the optential of being a noise.
#'    \item \code{call} contains the original call to the filter.
#' }
#' @author Wanwan Zheng
#' @import solitude
#' @import Rcpp
#' @import RcppArmadillo
#' @import e1071
#' @import caret
#' @importFrom("stats", "as.formula", "model.frame")
#'
#' @examples
#' 
#' data(iris)
#' out = shadown(Species~.,iris)
#' 
#'@name shadown
#'@export


shadown = function(x, ...){
  UseMethod("shadown")
}

#' @rdname shadown
#' @export
shadown.formula = function(formula,
                       data,
                       ...){
  if(!is.data.frame(data)){
    stop("data argument must be a data.frame")
  }
  modFrame = model.frame(formula,data) # modFrame is a data.frame built from 'data' using the variables indicated in 'formula'. The first column of 'modFrame' is the response variable, thus we will indicate 'classColumn=1' when calling the HARF.default method in next line.
  attr(modFrame,"terms") = NULL
  ret = shadown.default(x=modFrame,...,classColumn=1)
  ret$call = match.call(expand.dots = TRUE)
  ret$call[[1]] = as.name("shadown")
  cleanData = data
  ret$cleanData = cleanData[setdiff(1:nrow(cleanData),ret$remIdx),]
  return(ret)
}

#' @rdname shadown
#' @export
shadown.default = function(x,
                       knn=10,
                       classColumn=1,
                       iForest = TRUE,
                       threads = 5,
                       inters = 100,
                       p=0.05,
                       nshadow=1,
                       ...)
{
  if(!is.data.frame(x)){
    stop("data argument must be a data.frame")
  }
  if(!classColumn%in%(1:ncol(x))){
    stop("class column out of range")
  }
  #x=Data
  X.origin = scale(x[,-classColumn])
  Y.origin = as.numeric(as.factor(x[,classColumn]))
  shape = dim(X.origin)
  types = unique(Y.origin)
  ntype = length(types)
  score.frame = matrix(0,shape[1]+ntype*nshadow, inters)
  var.num = floor(2*shape[2]/3)
  rando = matrix(0,inters,var.num)
  
  for(inter in 1:inters){
    #rando[inter,] = floor(runif(var.num,min=1,max=shape[2]))
    rando[inter,] = sample(1:shape[2],var.num,replace = FALSE)
  }
  
  for(inter in 1:inters){
    X = X.origin[,rando[inter,]]
    Y = Y.origin
    
    X.shadow = matrix(rnorm(nshadow*ntype*ncol(X)),,ncol(X))
    colnames(X.shadow) = colnames(X)
    X = rbind.data.frame(X, X.shadow)
    Y = c(Y, rep(1:ntype,each=nshadow))
    
    out = kernelknn(X, TEST_data = NULL, Y, k = nrow(X)-1, method = 'euclidean',
                    threads = threads, weights_function ="gaussian", regression = FALSE, Levels=Y)
    #label of the nearest points
    Mat = out$idx
    #distance to the nearest points
    Dis = out$dis
    
    Tab_Y = as.matrix(table(Y))
    #len_min = min(Tab_Y)
    len_min = knn
    Mat_lab = cbind(Y, Mat)
    Rownames = as.integer(rownames(Tab_Y))
    
    for(i in 1:nrow(Tab_Y)){
      Log = (Mat_lab[,1]==Rownames[i])
      subdata_idx = Mat_lab[Log,1:len_min]
      subdata = as.data.frame(cbind(subdata_idx, Dis[Log,1:(len_min-1)]))
      real = rep(Rownames[i],len_min-1)
      
      score = apply(subdata,1, function(xx, true=real,length=len_min){
        pre = xx[2:length]
        llog = pre[pre!=true]
        Len= length(llog)/(length-1)
        if((Len == 0) ||(Len == 1)){
          Len
        }else{
          Len * (1-entropy(table(as.numeric(llog))))
        }
      })
      score = normalized(score)
      
      density = apply(subdata,1, function(xx, true=real,length=len_min){
        pre = xx[2:length]
        den = xx[-c(1:length)]
        Len= length(pre[pre!=true])
        
        if(Len == 0){
          0
        }else if (Len == length-1){
          1
        }else{
          density_hit = mean(0.000001+den[pre==true])
          density_diff = mean(0.000001+den[pre!=true])
          1-e1071::sigmoid(density_diff/density_hit)
        }
      })
      density = normalized(density)
      
      if(iForest == TRUE){
        isf = solitude::isolationForest$new(sample_size=floor(nrow(X[Log,])/3))
        isf$fit(X[Log,])
        iForest.score = normalized(isf$predict(X[Log,])$anomaly_score)
        score.frame[Log,inter]=(score+density+iForest.score)/3
      }else{
        score.frame[Log,inter]=(score+density)/2
      }
    }
  }
  
  #' @keywords internal
  #' @importFrom("graphics", "boxplot")
  score.frame.y = cbind.data.frame(Y,score.frame)
  isNoise  = c()
  shadow = list()
  
  for(s in 1:ntype){
    score.frame.sub = score.frame.y[score.frame.y$Y==s,]
    nn=nrow(score.frame.sub)-nshadow
    score.frame.original = score.frame.sub[1:nn,-1]
    score.frame.shadow = score.frame.sub[-(1:nn),-1]
    score.original.ave = apply(score.frame.original,1,mean)
    score.shadow.ave = apply(score.frame.shadow,1,mean)
    
    max.shadow.value = mean(score.shadow.ave)
    #max.shadow.loc = score.frame.shadow[which.max(score.shadow.ave),]
    max.shadow.loc = apply(score.frame.shadow,2,mean)
    shadow = c(shadow,list(max.shadow.loc))
    
    score.ave.noise= (score.original.ave >= max.shadow.value)
    p.values = apply(score.frame.original,1,function(x){
      t.test(max.shadow.loc,x,var=FALSE)$p.value})
    com = (p.values < p) + (score.ave.noise)
    isNoise = c(isNoise,names(com[com==2]))
  }
  
  
  ##### Building the 'filter' object ###########
  remIdx = as.numeric(isNoise)
  len = 1:length(Y.origin)
  Idx = len[-remIdx]
  
  #if(length(remIdx)>0){
  #cat("noise:",remIdx, "\n")
  #}
  cleanData = x[Idx,]
  call = match.call()
  call[[1]] = as.name("shadowN")
  ret = list(cleanData = cleanData,
             remIdx = remIdx,
             Idx = Idx,
             shadow = shadow,
             noise_score = apply(score.frame,1,mean),
             score = score.frame,
             call = call)
  class(ret) = "filter"
  return(ret)
}
