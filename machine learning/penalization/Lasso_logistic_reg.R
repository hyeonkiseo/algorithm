rm(list=ls())
####################################################
################## 1. Data preprocessing ###########
####################################################
setwd("C:/Rdirectory/Statictical learning theory")
data <-  read.csv('data.txt')
data <- data[,2:dim(data)[2]]
head(data)
X <-  data[,1:9]
X[,5] <- as.numeric(as.factor(data$famhist))-1
X <-  scale(X) # standardization 
intercept = 1
X <-  cbind(intercept,X)
X <-  data.matrix(X)
X <- X[,-c(5,7)]

y <-  data$chd
####################################################
############ 2. Logistic Lasso #####################
####################################################

# soft_threshold
soft_threshold <- function(theta, lambda){
  if (theta > lambda) return(theta - lambda)
  else if (theta < -lambda) return(theta + lambda)
  else return(0)
}
# weighted coordinate descent by lasso
weighted_cd_lasso <- function(X,y,beta,w_i, lambda,iteration){
  for (iter in 1:iteration){
    for (j in 1:dim(X)[2]){
      partial_resid <- y - X[,-j] %*% beta[-j] #partial residuals
      beta[j] <- soft_threshold( mean(w_i * partial_resid * X[,j]), lambda)/mean(w_i * X[,j]^2) 
    }
  }  
  return(beta)
}

  

# logistic regression with lasso penalty
logistic_lasso = function(X,y, lambda, maxiter){
  beta <-  rep(0,ncol(X)) # initial value of beta
  ll  <-  100000
  for (iter in 1:maxiter){
    eta_i <-  X %*% beta
    p_i <-  1/(1+exp(-eta_i))
    w_i <-  p_i * (1-p_i)
    w_i <-  ifelse(abs(w_i-0) < 0.0001, 0.0001, w_i)
    z_i <-  eta_i + (y-p_i)/w_i
    beta <- weighted_cd_lasso(X,z_i,beta,w_i,lambda,1000)
    ll_temp <-  1/2 * mean(w_i*(z_i-X%*%beta)^2) + lambda*sum(abs(beta))
    if (ll - ll_temp < 0.00000000000000000001){
      break
    }else{
      ll <- ll_temp
    }
  }
  return(beta)
}

logistic_lasso(X,y,0,10000)



# compare with glmnet package
library(glmnet)
fit <- glmnet(X,y,family = 'binomial', alpha = 1, lambda = 0.05)
coef(fit)
####################################################
########## 3. coefficient trace plot ###############
####################################################
betas <- matrix(NA,nrow = 100, ncol = dim(X)[2])
colnames(betas) <-  colnames(X)
lambdas <- seq(0.001,0.18,length = 100)

for (i in 1:100){
  betas[i,] <- logistic_lasso(X,y, lambdas[i], 10000)
}
# delete intercept
betas <- betas[,-1]
#calculate sum of absolute values of betas
x_axis <- apply(abs(betas),1,sum)
# LASSO coefficient trace plot
matplot(x_axis, betas, type="l", lty=1, 
        xlab= 'L1 norm of beta', ylab= 'Coefficient', main = 'logistic LASSO coefficients trace plot', col= c("black", "chocolate1", "turquoise", "palegreen3", "gray34", "firebrick", "royalblue3"))
text(2, betas[1,], colnames(betas), cex=0.7, col='black')
