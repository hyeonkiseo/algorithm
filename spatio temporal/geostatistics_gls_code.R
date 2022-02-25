library(classInt)
library(fields)
library(maps)
library(sp)
library(gstat)
library(geoR)
library(mvtnorm)
library(MCMCpack)
library(coda)

###### 1 ######
# (b)
library(mvtnorm)
library(fields)

x <- seq(0, 1, length = 500) # fine grid
d <- as.matrix(dist(x)) # create distance matrix
sigma1 <-  Matern(d, range = 1, nu = 0.5) 
# make covariance matrix
mu1 <-  rep(0,500) # make mu vector

y_samp_gen <- function(mu, sigma,seed){
  L <-  t(chol(sigma))
  set.seed(seed)
  Z <-  rnorm(dim(sigma)[1], mean = 0, sd = 1)
  y <- mu + L %*% Z
  return(y)
}

Y <-  y_samp_gen(mu1, sigma1, 2021)
dim(Y)
head(Y)


# (c)
# change range
sigma1 <- Matern(d, range = 1, nu = 0.5) # same as exp
y1 <- y_samp_gen(mu = mu1, sigma = sigma1, seed = 2021)

sigma2 <- Matern(d, range = 0.5, nu = 0.5) # change range
y2 <- y_samp_gen(mu = mu1, sigma = sigma2, seed = 2021)

ylim <- range(c(y1, y2))
par(mfrow = c(1, 2), mar = c(3,3,3,3))
matplot(x, y1, type = "l", ylab = "Y(x)", ylim = ylim, main = expression(rho*"=1"))
matplot(x, y2, type = "l", ylab = "Y(x)", ylim = ylim, main = expression(rho*"=0.5")) 
# scale becomes bigger


# change nu
sigma1 <- Matern(d, range = 1, nu = 0.5) # same as exp
y1 <- y_samp_gen(mu = mu1, sigma = sigma1, seed = 2021)

sigma3 <- Matern(d, range = 1, nu = 2) # change range
y3 <- y_samp_gen(mu = mu1, sigma = sigma3, seed = 2021)

ylim <- range(c(y1, y3))
par(mfrow = c(1, 2), mar = c(3,3,3,3))
matplot(x, y1, type = "l", ylab = "Y(x)", ylim = ylim, main = expression(nu*"=0.5"))
matplot(x, y3, type = "l", ylab = "Y(x)", ylim = ylim, main = expression(nu*"=2"))

# change mu
mu1 <-  rep(0,500) # make mu vector
y1 <- y_samp_gen(mu = mu1, sigma = sigma1, seed = 2021)

mu2 <-  rep(0.5,500)
y4 <- y_samp_gen(mu = mu2, sigma = sigma1, seed = 2021)

ylim <- range(c(y1, y4))
par(mfrow = c(1, 2), mar = c(3,3,3,3))
matplot(x, y1, type = "l", ylab = "Y(x)", ylim = ylim, main = expression(mu*"=0"))
matplot(x, y4, type = "l", ylab = "Y(x)", ylim = ylim, main = expression(mu*"=0.5"))


###### 2 #####
# (a)
library(sp)
library(gstat)
library(fields)
library(classInt)
library(maps)

load("CAtemps.RData")
CAtemp
linmod <- lm(avgtemp ~ lon + lat + elevation, data = CAtemp)
summary(linmod)
linmod$coefficients
fitted <- predict(linmod, newdata = CAtemp, na.action = na.pass)
ehat <- CAtemp$avgtemp - fitted

# plotting
ploteqc <- function(spobj, z, breaks, ...){
  pal <- tim.colors(length(breaks)-1)
  fb <- classIntervals(z, n = length(pal), 
                       style = "fixed", fixedBreaks = breaks)
  col <- findColours(fb, pal)
  plot(spobj, col = col, ...)
  image.plot(legend.only = TRUE, zlim = range(breaks), col = pal)
}



range(ehat)
breaks <- -7:7
x11()
ploteqc(CAtemp, ehat, breaks, pch = 19)
map("county", region = "california", add = TRUE)
title(main = "Average Annual Temperatures residuals using OLS\n,
      1961-1990, Degrees F")



# (b)
CAtemp$ehat <- ehat
CAtemp.sub <- CAtemp[!is.na(ehat),] # Remove lines with missing data
head(CAtemp.sub)

# range(CAtemp.sub$ehat)
vg <- variogram(ehat ~ 1, data = CAtemp.sub, width=30) #  width : set bins
plot(vg, xlab = "Distance", ylab = "Semi-variogram estimate", width=15)


# fit exponential variogram using weighted least square
fitvg <- fit.variogram(vg, vgm(1, "Exp", 500, 0.05))
print(fitvg)
# store estimates
s2.hat <- fitvg$psill[2]
rho.hat <- fitvg$range[2]
tau2.hat <- fitvg$psill[1]

# plotting
plot(vg, fitvg, xlab = "Distance", ylab = "Semi-variogram estimate")


#(c)
Y = CAtemp.sub$avgtemp
d <- rdist.earth(coordinates(CAtemp)) 
cov <- s2.hat * Matern(d, range = rho.hat,
                       nu = 0.5) + tau2.hat *diag(dim(d)[1]) # cov matrix
cov.inv <- solve(cov) # save inverse of covariance matrix
X <- cbind(rep(1, dim(d)[1]),
           CAtemp$lon, CAtemp$lat, CAtemp$elevation) # build X using cbind 
beta.hat.gls <- 
  solve(t(X) %*% cov.inv %*% X) %*% t(X) %*% cov.inv %*% Y  
# calculate beta hat
beta.hat.gls

#(d)
dcross <-  rdist.earth(coordinates(CAtemp), coordinates(CAgrid)) 
Sigmacross <- s2.hat * Matern(dcross, range = rho.hat, nu = 0.5)
Xpred <- cbind(rep(1,dim(CAgrid)[1]),CAgrid$lon, CAgrid$lat, CAgrid$elevation)
Ypred <- Xpred %*% beta.hat.gls +
  t(Sigmacross) %*% cov.inv %*%(Y - X %*% beta.hat.gls)

# plotting Ypred
range(Ypred)
breaks <- 33:75
x11()
ploteqc(CAgrid, Ypred, breaks, pch = 19)
map("county", region = "california", add = TRUE)
title(main = "EBLUP for average temperature, 1961-1990, Degrees F")


# MSE
b <-  t(Xpred) - t(X)%*%cov.inv%*%Sigmacross
vpred <- s2.hat - diag(t(Sigmacross) %*% solve(cov, Sigmacross) +
                         t(b) %*% solve(t(X) %*% solve(cov, X), b))
vpred[vpred<0] <- 0
sepred <- sqrt(vpred)
sepred

# plotting MSE
range(sepred)
breaks <- seq(0.6,1.9,by = 0.01)
x11()
ploteqc(CAgrid, sepred, breaks, pch = 19)
map("county", region = "california", add = TRUE)
title(main = "SE for EBLUP , 1961-1990, Degrees F")

