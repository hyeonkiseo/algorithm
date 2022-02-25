##########################
##### Question 1 #########
##########################

set.seed(2021)
Ran = owin(xrange = c(0,50), yrange = c(0,50))
X = rStrauss(beta = 0.1, gamma = 0.5,R = 1.5, W = Ran )
plot(X, main = 'Strauss process X')
?plot



##########################
##### Question 3 #########
##########################

# process function without normalizing constant
set.seed(2021)
h <- function(X, beta, gamma) {
  n <-  npoints(X) 
  s <- (sum(pairdist(X) < 1.5) - n) / 2 
  return (beta^n * gamma ^s)
}

# Single variable exchange algorithm
sve <- function(X,beta_init,gamma_init,iter,beta_sd, gamma_sd){
  ## empty seq
  beta <- c()
  gamma <- c()
  accept_count <- 0
  
  beta <- append(beta,beta_init)
  gamma <- append(gamma,gamma_init)
  
  # run MCMC
  for (i in 1:(iter -1)) {
    
    if (i %% 1000 == 0) {
      print(i / iter)
    }
    
    # para candidate
    temp_beta <- rnorm(1,mean = beta[i], sd = beta_sd)
    temp_gamma <- rnorm(1,mean = gamma[i], sd = gamma_sd)
    while (temp_beta < 0 | temp_gamma < 0 | temp_gamma > 1 ) {
      temp_beta <-rnorm(1,mean = beta[i], sd = beta_sd)
      temp_gamma <- rnorm(1,mean = gamma[i], sd = gamma_sd)
    }
    
    # auxiliary varriable
    W <- rStrauss(beta = temp_beta, gamma = temp_gamma, R = 1.5, W = owin(xrange = c(0,50),yrange = c(0,50)))
    
    
    # density of q
    q_n <-  dnorm(x=beta[i], mean = temp_beta, sd = beta_sd, log = TRUE) +
      dnorm(x = gamma[i],mean = temp_gamma, sd = gamma_sd, log = TRUE)
    q_d <-  dnorm(x = temp_beta, mean = beta[i], sd = beta_sd, log = TRUE)+
      dnorm(x = temp_gamma, mean = gamma[i], sd = gamma_sd, log = TRUE)
    
    #prior is always same because they follows uniform dist
    
    #  obs h(x) 
    h_obs_n <- log(h(X,temp_beta,temp_gamma))
    h_obs_d <- log(h(X,beta[i], gamma[i]))
    
    # aux h(x)
    h_aux_n <-log(h(W,beta[i], gamma[i]))
    h_aux_d <-log(h(W,temp_beta,temp_gamma))
    
    # log ratio 
    log_ratio <- q_n + h_obs_n + h_aux_n - (q_d + h_obs_d + h_aux_d)
      
    if (is.na(log_ratio)){
      beta <- append(beta,beta[i])
      gamma <- append(gamma,gamma[i])
      next
    }
    
    # update
    r <- runif(1)
    log_r <- log(r)
    
    if(log_r < log_ratio) {
      beta <- append(beta,temp_beta)
      gamma <- append(gamma,temp_gamma)
      accept_count <- accept_count+1
    } else {
      beta <- append(beta,beta[i])
      gamma <- append(gamma,gamma[i])
    }
  }
  
  # acceptance_ratio
  acceptance_ratio <-  accept_count / iter
  
  result <- list(beta = beta, gamma = gamma, acceptance_ratio = acceptance_ratio)
  
  return(result)
}

# run MCMC
result <- sve(X,beta_init = 0.1, gamma_init = 0.5, iter = 10000, beta_sd = 0.05, gamma_sd = 0.05 )

# trace plot
## beta
pdf("traceplot.pdf", width=8, height=5)
par(mar=c(3,3,3,3),mfrow = c(1,2) )
plot.ts(result$beta, main = 'Trace plot of Beta')
abline(h = 0.1, col = 'red')
plot.ts(result$gamma, main = 'Trace plot of Gamma')
abline(h = 0.5, col = 'red')
dev.off()

# acceptance rate
result$acceptance_ratio

# posterior mean
mean(result$beta)
mean(result$gamma)

#HPD
quantile(result$beta, probs = c(0.025, 0.975))
quantile(result$gamma, probs = c(0.025, 0.975))




##########################
##### Question 4 #########
##########################

library(sp)

# birth death MCMC 
birth_death_MCMC <- function(X, beta, gamma, iter, domain_area){
  n <- npoints(X)
  s <- (sum(pairdist(X) < 1.5) - n)
  
  accept_count <- 0
  birth_count <- 0
  death_count <- 0
  
  # run MCMC
  for (i in 1:(iter -1)){
    
    if (i %% 1000 == 0){
      print(i / iter)
    }
    # decide birth or death
    birth <- rbinom(1,1,0.5)
    
    if (birth == 1) {
      # birth
      birth_count <- birth_count +1
      
      # generate new point
      new_pt <- rpoint(1, win = owin(xrange = c(0,50), yrange = c(0,50)))
      
      # insert new point
      tempX <- X
      tempX$n <- tempX$n +1 
      tempX$x[X$n+1] <- new_pt$x
      tempX$y[X$n+1] <- new_pt$y
      
      # accept prob
      numerator <-  log(h(tempX, beta = beta, gamma = gamma)) + log(domain_area)
      denom <- log(h(X, beta = beta, gamma = gamma)) + log(X$n +1)
      log_ratio <- numerator - denom
      
      r <- runif(1)
      log_r <- log(r)
      
      if(log_r < log_ratio) {
        X <-tempX
        accept_count <- accept_count+1
      }
      
    }else{
      # death
      death_count <- death_count + 1
      
      # remove one point
      tempX <- X[-sample(1:X$n,1)]
      
      # accept prob
      numerator <- log(h(tempX,beta = beta, gamma = gamma)) + log(X$n-1)
      denom <- log(h(X,beta = beta, gamma = gamma)) + log(domain_area)
      log_ratio <- numerator - denom
      ratio = exp(log_ratio)
      
      r <- runif(1)
      log_r <- log(r)
      
      if (log_r < log_ratio){
        X <-  tempX
        accept_count <- accept_count +1
      }
    }
    
    n <-  append(n, npoints(X))
    s <- append(s, (sum(pairdist(X) < 1.5) - npoints(X))/2)
  }
  
  accept_ratio <- accept_count / iter
  result <- list(n = n, s = s, accept_ratio = accept_ratio,
                 birth_count = birth_count, death_count = death_count,
                 X = X)
  return(result)
}

# trace plot of n(X)
par(mar = c(3,3,3,3), mfrow = c(1,1))
plot.ts(result2$n, main = 'Traceplot of n(X)')

##########################
##### Question 5 #########
##########################

double_MH <- function(X, beta_init,gamma_init,iter1,iter2,beta_sd, gamma_sd, domain_area){
  ## empty seq
  beta <- c()
  gamma <- c()
  n_W <- matrix(0,ncol = iter1, nrow = iter2)
  accept_count <- 0
  
  beta <- append(beta,beta_init)
  gamma <- append(gamma,gamma_init)
  
  # outer MCMC
  for (i in 1:(iter1 -1)) {
    
    if (i %% 100 == 0) {
      print(i / iter1)
    }
    
    # para candidate
    temp_beta <- rnorm(1,mean = beta[i], sd = beta_sd)
    temp_gamma <- rnorm(1,mean = gamma[i], sd = gamma_sd)
    while (temp_beta < 0 | temp_gamma < 0 | temp_gamma > 1 ) {
      temp_beta <-rnorm(1,mean = beta[i], sd = beta_sd)
      temp_gamma <- rnorm(1,mean = gamma[i], sd = gamma_sd)
    }
    
    # auxiliary variable MCMC (inner MCMC)
    tempW <- rpoint(200, win = owin(xrange = c(0,50),yrange = c(0,50)))
    W_result <- birth_death_MCMC(tempW,beta = temp_beta,
                          gamma = temp_gamma,iter = iter2,domain_area = domain_area)
    n_W[,i+1] <- W_result$n
    W <- W_result$X
    
    
    # density of q
    q_n <-  dnorm(x=beta[i], mean = temp_beta, sd = beta_sd, log = TRUE) +
      dnorm(x = gamma[i],mean = temp_gamma, sd = gamma_sd, log = TRUE)
    q_d <-  dnorm(x = temp_beta, mean = beta[i], sd = beta_sd, log = TRUE)+
      dnorm(x = temp_gamma, mean = gamma[i], sd = gamma_sd, log = TRUE)
    
    #prior is always same because they follow uniform dist
    
    #  obs h(x) 
    h_obs_n <- log(h(X,temp_beta,temp_gamma))
    h_obs_d <- log(h(X,beta[i], gamma[i]))
    
    # aux h(x)
    h_aux_n <-log(h(W,beta[i], gamma[i]))
    h_aux_d <-log(h(W,temp_beta,temp_gamma))
    
    # log ratio 
    log_ratio <- q_n + h_obs_n + h_aux_n - (q_d + h_obs_d + h_aux_d)
    
    if (is.na(log_ratio)){
      beta <- append(beta,beta[i])
      gamma <- append(gamma,gamma[i])
      next
    }
        
    # update
    r <- runif(1)
    log_r <- log(r)

    
    if(log_r < log_ratio) {
      beta <- append(beta,temp_beta)
      gamma <- append(gamma,temp_gamma)
      accept_count <- accept_count+1
    } else {
      beta <- append(beta,beta[i])
      gamma <- append(gamma,gamma[i])
    }
  }
  
  # acceptance_ratio
  acceptance_ratio <-  accept_count / iter1
  
  result <- list(beta = beta, gamma = gamma,
                 n_W = n_W, acceptance_ratio = acceptance_ratio)
  
  return(result)
}

result3 <- double_MH(X, 0.1,0.5,10000,500,0.01,0.01,2500)
result3

# trace plot

pdf("5traceplot.pdf", width=8, height=5)
par(mar=c(3,3,3,3),mfrow = c(1,2) )
plot.ts(result3$beta, main = 'Trace plot of Beta')
abline(h = 0.1, col = 'red')
plot.ts(result3$gamma, main = 'Trace plot of Gamma')
abline(h = 0.5, col = 'red')
dev.off()
#acceptance ratio
result3$acceptance_ratio

# posterior mean
mean(result3$beta)
mean(result3$gamma)

#95% HPD
quantile(result3$beta, probs = c(0.025,0.975))
quantile(result3$gamma, probs = c(0.025,0.975))
