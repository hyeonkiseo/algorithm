
######################################################
####################### HW1 ##########################
##############  2021011169 Hyeonki Seo ###############
######################################################



####################################################
################## 1. Data preprocessing ###########
####################################################


#install.packages('/Users/hyeonki/Data/R/StatisticalLearningTheory/ElemStatLearn_2015.6.26.2.tar.gz', respos = NULL, type = 'source')
library(ElemStatLearn)

#read data
?prostate
prostate <- read.csv('/Users/hyeonki/Data/R/StatisticalLearningTheory/prostate.data.txt',sep='\t')
head(prostate)

# save objects
train_data <- prostate[prostate$train == 1,2:10]
train_X <- train_data[,1:8]
train_y <-train_data[,9]

# centering and standardizing X
train_X <- as.matrix(scale(train_X, scale = TRUE))
train_y <- as.matrix(scale(train_y, scale = FALSE))
head(train_X)
head(train_y)



####################################################
########## 2. Coordinate Descent Algorithm##########
####################################################


# soft_threshold
soft_threshold <- function(theta, lambda){
  if (theta > lambda) return(theta - lambda)
  else if (theta < -lambda) return(theta + lambda)
  else return(0)
}

# lambda sequance by warm start method
lambda_max <- max(abs(t(train_X) %*% train_y)) / dim(train_X)[1]
lambda_min <- lambda_max * 0.0001
lambdas <- exp(seq(log(lambda_min), log(lambda_max), length = 100))


#coordinate_descent_lasso fit by given lambda
coordinate_descent_lasso <- function(train_X,train_y, lambda,iteration){
  beta = runif(ncol(train_X))  # initial value of beta
    for (iter in 1:iteration){
      for ( j in 1:dim(train_X)[2]){
        X_wo_j <- train_X[,-j] # X without jth predictor
        partial_resid_j <- train_y - X_wo_j %*% beta[-j] #partial residuals
        beta[j] <- soft_threshold(t(partial_resid_j) %*% train_X[,j] / dim(train_X)[1], lambda)/
          mean(train_X[,j]^2)  # beta hat of lasso
      }
    }
  return(beta)
}


# beta coefficient trace matrix with each of lambda
betas <- matrix(NA,nrow = 100, ncol = dim(train_X)[2])
for (i in 1:100){
  betas[i,] <- coordinate_descent_lasso(train_X,train_y, lambdas[i], 100)
}
rownames(betas) <- lambdas





#######################################################
############# 3.10-fold cross validation ##############
#######################################################


# make validation data index for 10 fold cross validation
set.seed(1)
index <- sample(dim(train_X)[1])
fold <- rep(1:10,7)[1:dim(train_X)[1]]

mse <- matrix(NA, nrow = 100, ncol = 10)
for(f in 1:10){
  # split x and y by 10-fold
  x_tr <- train_X[-index[fold==f],] ;  x_val <- train_X[index[fold==f],]
  y_tr <- train_y[-index[fold==f],] ;  y_val <- train_y[index[fold==f],]
  
  # make coef with each of lambda value
  beta_fitted <- matrix(NA,nrow = 100, ncol = dim(x_tr)[2])
  for (i in 1:100){
    beta_fitted[i,] <- coordinate_descent_lasso(x_tr,y_tr,lambdas[i],100)
  }
  
  # calculate mse by fitted beta
  for (lambda in 1:100){
    y_hat <- x_val %*% beta_fitted[lambda,]
    resid <- y_val - y_hat
    mse[lambda,f]  <- (t(resid)%*%resid) / dim(resid)[1]
  }
}


msemean <- apply(mse,1,mean)
minmse <- min(msemean)
min_index <-which.min(msemean)
# min(mse) + 1 standard error bound index
boundindex <- which.max(msemean[msemean < minmse + sd(msemean)])
boundindex




########################################################
################### 4. Trace plot  #####################
########################################################

betas <- matrix(NA,nrow = 100, ncol = dim(train_X)[2])
for (i in 1:100){
  betas[i,] <- coordinate_descent_lasso(train_X,train_y, lambdas[i], 100)
}
lambdas <- exp(seq(log(lambda_min), log(lambda_max), length = 100))


#calculate shrinkage factor
lse <- lm(train_y ~ train_X)
t <-  sum(abs(coef(lse)[-1]))  # maximum value of sum of lasso coefs
s1 <- apply(abs(betas),1,sum) / t

# LASSO coefficient trace plot
matplot(s1, betas, type="l", lty=1, xlim=c(0, 1.05), 
        xlab="Shrinkage Factor s", ylab="Coefficients", 
        main = 'LASSO coefficients trace plot', col='blue')
text(1.05, betas[1,], colnames(train_X), cex=0.7, col='black')
abline(v = s1[boundindex], lty = 3, col = 'black')
text (s1[boundindex]-0.05, 0.35, labels = '1se\nbound', cex = 0.7, col = 'black')
abline(v = s1[boundindex -2], lty = 3, col = 'red') 
text(s1[boundindex-2]+0.05, 0.35, labels = 'sparse\npoint', cex = 0.7, col = 'black')
abline(h = 0, lty = 1)

