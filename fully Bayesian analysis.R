#Packages
################################################################################
################################################################################
require(R2jags)
require(mcmcse)
require(bayesplot)
require(TeachingDemos)
require(carData)
#Data
################################################################################
################################################################################
library("carData")
data("Leinhardt")
?Leinhardt
# income: (int)
# Per-capita income in U. S. dollars.
# infant: (num)
# Infant-mortality rate per 1000 live births.
# region:
# A factor with 4 levels: Africa; Americas; Asia, Asia and Oceania; Europe.
# oil: 
# Oil-exporting country. A factor with 2 levels: no, yes.


head(Leinhardt)
summary(Leinhardt)
str(Leinhardt)
pairs(Leinhardt)


plot(infant ~ income, data=Leinhardt, main="Scatter plot of infant and income")
# both are right skewed and positive 
hist((Leinhardt$infant), breaks = 50)
hist((Leinhardt$income), breaks = 50)


#if we check log-scale

Leinhardt$loginfant = log(Leinhardt$infant)
Leinhardt$logincome = log(Leinhardt$income)

plot(loginfant ~ logincome, data=Leinhardt, main="Scatter plot of infant and income in log-log scale")
# Now, we can see a linear model and we use the logarithm scale to write our model
# Since we can transform the x and y variables to get new x's and new y's to have a linear model

################################################################################
################################################################################
################################### Model 1 ####################################
################################################################################
################################################################################
dataset = na.omit(Leinhardt)
na.action(dataset)

model_1 <- function(){
  
  # likelihood
  for(i in 1:N)
  {
    y[i] ~ dnorm(mu[i], tau)
    mu[i] <- beta0 + beta1*logincome[i] # linear model
  }
  
  #prior
  beta0 ~ dnorm(0, 1/1.0e6)
  beta1 ~ dnorm(0, 1/1.0e6)
  
  tau ~ dgamma(1.0E-3, 1.0E-3) # dgamma(shape, rate)
  sig_2 <- 1.0/tau
  sig <- sqrt(sig_2)
}

#data for jags
y = dataset$loginfant
logincome = dataset$logincome
N = nrow(dataset)

data.jags <- list("y", "logincome", "N")

# parameters of interest
model_1.params <- c("beta0", "beta1","sig")

# inits
model_1.inits <- function() {
  inits = list("beta0"=rnorm(1,0,10),
               "beta1"=rnorm(1,0,10),
               "tau"=rgamma(1,1,1))
}

#run jags
set.seed(123)
model_1.fit <- jags(data = data.jags, 
                model.file = model_1, inits = model_1.inits,
                parameters.to.save = model_1.params,                  
                n.chains = 3, n.iter = 9000, n.burnin = 1000, n.thin=10)

#Result and findings
################################################################################
################################################################################


#A trace plot shows the history of a parameter value across iterations of the chain. 
# Here we have iterations across the x-axis. It shows you precisely where the 
# chain has been exploring.
# If the chain is stationary, it should not be showing any long-term trends. 
# The average value for the chain, should be roughly flat. And it should not be 
# wandering(If this is the case, we need to run the chain for many, many 
# more iterations)

# Before we check the inferences from the model, we should perform convergence 
# diagnostics for our Markov chains.
# Plots with BayesPlot
library('bayesplot')
chainArray <- model_1.fit$BUGSoutput$sims.array
mcmc_dens(chainArray)
mcmc_acf(chainArray)
mcmc_combo(chainArray)

# different colors are different chains we ran
traceplot(as.mcmc(model_1.fit))

#Gelman and Rubin's convergence diagnostic
# The ‘potential scale reduction factor’ is calculated for each variable in x, 
# together with upper and lower confidence limits
# Approximate convergence is diagnosed when the upper limit is close to 1. 
gelman.diag(as.mcmc(model_1.fit))
?gelman.diag
# By calculating the shrink factor at several points in time, gelman.plot shows 
# if the shrink factor has really converged, or whether it is still fluctuating.
# In gelman.plot we check that the "black(median) line" should be completely below 
# of the "dotted(97.5%) line"
gelman.plot(as.mcmc(model_1.fit))
?gelman.plot



#Geweke's convergence diagnostic
# Geweke (1992) proposed a convergence diagnostic for Markov chains based on a test
# for equality of the means of the first and last part of a Markov chain (by default 
# the first 10% and the last 50%). If the samples are drawn from the 
# stationary distribution of the chain, the two means are equal and Geweke's statistic 
# has an asymptotically standard normal distribution.
geweke.diag(as.mcmc(model_1.fit))
?geweke.diag
geweke.plot(as.mcmc(model_1.fit))
?geweke.plot

# Autocorrelation is a number between negative 1 and positive 1 which measures
# how linearly dependent the current value of the chain is to past values called lags.
library('coda')
autocorr.diag(as.mcmc(model_1.fit))
?autocorr.diag
autocorr.plot(as.mcmc(model_1.fit))
autocorr.plot(as.mcmc(model_1.fit), lag.max = 10)
?autocorr.diag


# Effective sample size for estimating the mean
# For a time series x of length N, the standard error of the mean is the
# square root of var(x)/n where n is the effective sample size. n = N only when
# there is no autocorrelation.
effectiveSize(as.mcmc(model_1.fit))



#The Raftery, Lewis diagnostic
raftery.diag(as.mcmc(model_1.fit))
?raftery.diag


# Inferential Findings
model_1.fit$BUGSoutput
(DIC_model_1 = model_1.fit$BUGSoutput$DIC)
(Model_1_summary = model_1.fit$BUGSoutput$summary)



# Bayesian point and interval estimation
chainMatrix_model_1 <- model_1.fit$BUGSoutput$sims.matrix


# Point estimates
(theta_model_1.hat.jags <- colMeans(chainMatrix_model_1))


#Equal-Tailed Interval
cred <- 0.95
(theta_model_1.ET.jags <- apply(chainMatrix_model_1, 2, quantile, prob=c((1-cred)/2, 1-(1-cred)/2)))


# Highest Posterior Density intervals
(theta_model_1.HPD.jags <- HPDinterval(as.mcmc(chainMatrix_model_1),prob = 0.95))

?HPDinterval




# Residuals
# Residuals are defined as the difference between the response, the actual observation, 
# and the model's prediction for each value.
logincome = dataset$logincome
N = nrow(dataset)

# data for explanatory variable logincome
X = cbind(rep(1.0, N), logincome)
head(X)

# posterior mean
theta_model_1.hat.jags


# predicted value
y.hat = X %*% theta_model_1.hat.jags[1:2]

# matrix to vector
(y_model_1.hat = drop(y.hat))


# residuals
y = dataset$loginfant

resid_model_1 = y - y_model_1.hat
plot(resid_model_1)

plot(y_model_1.hat, resid_model_1)
qqnorm(resid_model_1)

head(rownames(dataset))
# The countries that have the largest positive residuals
head(rownames(dataset)[order(resid_model_1, decreasing=TRUE)])


# Hypothesis testing
#The marginal likelihood represents the average probability of the data across 
#parameter space, or the average quality of fit of a given model to the data.
# Bayes Factor:  using the ttestBF function, which performs the “JZS” t test 
# described by Rouder, Speckman, Sun, Morey, and Iverson (2009). 
#The ratio of marginal likelihoods is known as the Bayes factor and is an 
#elegant method for comparing models in a Bayesian context.
#Bayes Factors will tell us how much the data changes our belief relative to 
# our prior beliefs.
library(BayesFactor)
BF = ttestBF(y,y_model_1.hat)
BF
exp(BF@bayesFactor$bf)
1/ttestBF(y,y_model_1.hat)#3 – 10	Moderate evidence for H1

################################################################################
################################################################################
################################### Model 2 ####################################
################################################################################
################################################################################
model_2 <- function(){
  
  # likelihood
  for(i in 1:N)
  {
    y[i] ~ dnorm(mu[i], tau)
    mu[i] <- beta0 + beta1*logincome[i] + beta2*is_oil[i]
  }
  
  #prior
  beta0 ~ dnorm(0, 1/1.0e6)
  beta1 ~ dnorm(0, 1/1.0e6)
  beta2 ~ dnorm(0, 1/1.0e6)
  
  tau ~ dgamma(1.0E-3, 1.0E-3) # dgamma(shape, rate)
  sig_2 <- 1.0/tau
  sig <- sqrt(sig_2)
}

#data for jags
y = dataset$loginfant
logincome = dataset$logincome
is_oil=as.numeric(dataset$oil=="yes")
N = nrow(dataset)

data2.jags <- list("y", "logincome","is_oil", "N")

# parameters of interest
model_2.params <- c("beta0", "beta1", "beta2","sig")

# inits
model_2.inits <- function() {
  inits = list("beta0"=rnorm(1,0,10),
               "beta1"=rnorm(1,0,10),
               "beta2"=rnorm(1,0,10),
               "tau"=rgamma(1,1,1))
}

#run jags
set.seed(123)
model_2.fit <- jags(data = data2.jags, 
                    model.file = model_2, inits = model_2.inits,
                    parameters.to.save = model_2.params,                  
                    n.chains = 3, n.iter = 9000, n.burnin = 1000, n.thin=10)

#Result and findings
################################################################################
################################################################################
model_2.fit$BUGSoutput
(DIC_model_2 = model_2.fit$BUGSoutput$DIC)
(Model_2_summary = model_2.fit$BUGSoutput$summary)

chainMatrix_model_2 <- model_2.fit$BUGSoutput$sims.matrix
(theta_model_2.hat.jags <- colMeans(chainMatrix_model_2))

#### Residual check
logincome = dataset$logincome
N = nrow(dataset)
is_oil=as.numeric(dataset$oil=="yes")

X_2 = cbind(rep(1.0, N), logincome, is_oil)
head(X_2)



y_2.hat = X_2 %*% theta_model_2.hat.jags[1:3]
# matrix to vector
y_model_2.hat = drop(y_2.hat)



y = dataset$loginfant

resid_model_2 = y - y_model_2.hat


plot(y_model_2.hat, resid_model_2, main = "model 2")
plot(y_model_1.hat, resid_model_1, main = "model 1")

sd(resid_model_2)
mean(resid_model_2)

qqnorm(resid_model_2,main = "Normal Q-Q Plot - residual model 2")
qqnorm(resid_model_1,main = "Normal Q-Q Plot - residual model 1")


# Hypothesis testing
# Bayes Factor
#Bayes Factors will tell us how much the data changes our belief relative to 
# our prior beliefs.
library(BayesFactor)
BF = ttestBF(y_model_2.hat,y)
BF
exp(BF@bayesFactor$bf)


################################################################################
################################################################################
################################### Model 4 ####################################
################################################################################
################################################################################
model_4 <- function(){
  
  # likelihood
  for(i in 1:N)
  {
    y[i] ~ dnorm(mu[i], tau)
    mu[i] <- alpha[region[i]] + beta1*logincome[i] + beta2*is_oil[i]
  }

  #prior  
  for (j in 1:num_regions) {
    alpha[j] ~ dnorm(mu_alpha, tau_alpha)
  }
  mu_alpha ~ dnorm(0, 1.0e-6)
  tau_alpha ~ dgamma(1.0E-3, 1.0E-3)
  sig_alpha_2 <- 1.0/tau_alpha
  sig_alpha <- sqrt(sig_alpha_2)
  
  

  beta1 ~ dnorm(0, 1/1.0e6)
  beta2 ~ dnorm(0, 1/1.0e6)
  
  tau ~ dgamma(1.0E-3, 1.0E-3) # dgamma(shape, rate)
  sig_2 <- 1.0/tau
  sig <- sqrt(sig_2)
}

#data for jags
y = dataset$loginfant
logincome = dataset$logincome
is_oil=as.numeric(dataset$oil=="yes")
region=as.numeric(dataset$region)
num_regions = length(levels(dataset$region))
N = nrow(dataset)

table(dataset$region,dataset$oil=="yes")

data4.jags <- list("y", "logincome","is_oil","region", "num_regions", "N")

# parameters of interest
model_4.params <- c("beta1", "beta2","sig", "sig_alpha", "mu_alpha", "alpha")


#run jags
set.seed(123)
model_4.fit <- jags(data = data4.jags, 
                    model.file = model_4,
                    parameters.to.save = model_4.params,                  
                    n.chains = 3, n.iter = 9000, n.burnin = 1000, n.thin=10)

#Result and findings
################################################################################
################################################################################
model_4.fit$BUGSoutput
(DIC_model_4 = model_4.fit$BUGSoutput$DIC)
(Model_4_summary = model_4.fit$BUGSoutput$summary)

library('bayesplot')
chainArray_4 <- model_4.fit$BUGSoutput$sims.array
mcmc_dens(chainArray_4)

traceplot(as.mcmc(model_4.fit))


################################################################################
################################################################################
################################### Model 5 ####################################
################################################################################
################################################################################

# T distribution
# The T distribution is similar to the normal distribution, just with fatter tails.
# The probability of getting values very far from the mean is larger with a
# T distribution than a normal distribution, which are better at accommodating outliers.

# df: The smaller the degrees of freedom in t distribution, the heavier the tails of the t distribution and
# t distribution does not have a mean and a variance if the degrees of freedom is less than two.
curve(dnorm(x),from = -10, to=10, col='blue')
curve(dt(x,2),from = -10, to=10, col='green',add=TRUE)


#model with T distribution likelihood
model_5 <- function(){
  
  # likelihood
  for(i in 1:N)
  {
    y[i] ~ dt( mu[i], tau, df )
    mu[i] <- a[region[i]] + beta1*logincome[i] + beta2*is_oil[i]
  }
  
  
  #prior 
  for (j in 1:num_regions) {
    a[j] ~ dnorm(mu_a, tau_a)
  }
  mu_a ~ dnorm(0, 1.0e-6)
  tau_a ~ dgamma(1.0E-3, 1.0E-3)
  sig_a_2 <- 1.0/tau_a
  sig_a <- sqrt(sig_a_2)
  
  beta1 ~ dnorm(0, 1/1.0e6)
  beta2 ~ dnorm(0, 1/1.0e6)
  
  df = nu + 2.0 # we want degrees of freedom > 2 to guarantee existence of mean and variance
  nu ~ dexp(1.0)
  
  tau ~ dgamma(1.0E-3, 1.0E-3) # tau is an inverse scale parameter
  sig = sqrt( 1.0 / tau * df / (df - 2.0) ) # df is degrees of freedom parameter
}

#data for jags
y = dataset$loginfant
logincome = dataset$logincome
is_oil=as.numeric(dataset$oil=="yes")
region=as.numeric(dataset$region)
num_regions=length(levels(dataset$region))
N = nrow(dataset)

data5.jags <- list("y", "logincome","is_oil","region", "num_regions", "N")

# parameters of interest
model_5.params <- c("beta1", "beta2","sig", "sig_a", "mu_a", "a")


#run jags
set.seed(123)
model_5.fit <- jags(data = data5.jags, 
                    model.file = model_5,
                    parameters.to.save = model_5.params,                  
                    n.chains = 3, n.iter = 100e3, n.burnin=floor(100e3/2),
                    n.thin=10)
#Result and findings
################################################################################
################################################################################
library('bayesplot')
chainArray_5 <- model_5.fit$BUGSoutput$sims.array
mcmc_dens(chainArray_5)

traceplot(as.mcmc(model_5.fit))

gelman.diag(as.mcmc(model_5.fit))
gelman.plot(as.mcmc(model_5.fit))

geweke.diag(as.mcmc(model_5.fit))
geweke.plot(as.mcmc(model_5.fit))

library('coda')
autocorr.diag(as.mcmc(model_5.fit))
autocorr.plot(as.mcmc(model_5.fit), lag.max=50)
mcmc_acf(chainArray_5)

effectiveSize(as.mcmc(model_5.fit))

raftery.diag(as.mcmc(model_5.fit))


model_5.fit$BUGSoutput
(DIC_model_5 = model_5.fit$BUGSoutput$DIC)
(Model_5_summary = model_5.fit$BUGSoutput$summary)
