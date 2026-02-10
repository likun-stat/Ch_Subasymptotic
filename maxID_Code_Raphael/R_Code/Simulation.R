#! /usr/bin/Rscript
args <- commandArgs(TRUE)
for (arg in args) eval(parse(text = arg))
rm(arg, args)


.libPaths(c("/scratch/huserrg/R/3.3", .libPaths()))

library(mvtnorm)
library(methods)
library(parallel)
library(fields)

#BASE <- "~/Documents/Work/05_AssistantProfessor-KAUST/Research/Max-id_Thomas-Emeric/Shaheen_Cluster/Max-id/"
BASE <- "/scratch/huserrg/Max-id/"
setwd(BASE)

source('R_Code/Tools.R')
dyn.load("C_Code/pmvnormEmeric.so")


###################################
### Main parameters (as inputs) ###
###################################

# Ds <- c(10,15,20) # dimensions of data (i.e., number of sites)
# ns <- c(50) # numbers of observations
# ranges <- c(0.5,1) # correlation range parameters
# smooths <- c(1) # correlation smoothness parameters
# alphas <- c(1,2,5) # random factor scale parameters
# betas <- c(0,0.5,1) # random factor shape parameters (AD/AI)
# repar <- TRUE # should the model be reparametrized as (alpha,beta,theta(hh),smooth) instead of (alpha,beta,range,smooth), where theta(hh) denotes the bivariate extremal coefficient at distance hh?
# hh <- 0.5 # distance for fixing the extremal coefficient in the reparametrization...
# fixed <- c(FALSE,FALSE,FALSE,TRUE) # whether alpha/beta/range/smooth are fixed to their true values in estimation
# Rs <- c(1:32) # replications
# N <- 10000 # mean number of Poisson points R_i simulated for the approximate simulation...
# sim <- 1 # simulation index
# ncores <- 32 # number of cores for parallel computing
# All <- FALSE # If TRUE, re-simulate even the experiments, which have already been simulated
# start.from.previous.par <- FALSE # If TRUE, the code will start the optimization from the previously estimated value

########################
### Simulation study ###
########################

simul.r <- function(r){
	set.seed(r)
	
	mle.r <- mle.r.beta0 <- array(dim=c(length(Ds),length(ns),length(ranges),length(smooths),length(betas),length(alphas),4))
	pw.nllik.r <- pw.nllik.r.beta0 <- array(dim=c(length(Ds),length(ns),length(ranges),length(smooths),length(betas),length(alphas)))
	conv.r <- conv.r.beta0 <- array(dim=c(length(Ds),length(ns),length(ranges),length(smooths),length(betas),length(alphas)))
	counts.r <- counts.r.beta0 <- array(dim=c(length(Ds),length(ns),length(ranges),length(smooths),length(betas),length(alphas),2))
	time.r <- time.r.beta0 <- array(dim=c(length(Ds),length(ns),length(ranges),length(smooths),length(betas),length(alphas)))

	for(i1 in 1:length(Ds)){
		D <- Ds[i1]
		for(i2 in 1:length(ns)){
			n <- ns[i2]
			for(i3 in 1:length(ranges)){
				range <- ranges[i3]
				for(i4 in 1:length(smooths)){
					smooth <- smooths[i4]
					for(i5 in 1:length(alphas)){
						alpha <- alphas[i5]
						for(i6 in 1:length(betas)){
							beta <- betas[i6]
							if(!file.exists(paste('Outputs/Sim',sim,'/All/mle_r',r,'_D',D,'_n',n,'_al',alpha,'_be',beta,'_ra',range,'_sm',smooth,'.RData',sep='')) | All){
								coord <- matrix(runif(D*2),ncol=2)
								parGauss0 <- c(range,smooth)
								parR0 <- c(alpha,beta)
								theta <- 2*pt(sqrt((alpha+1)*(1-exp(-(hh/range)^smooth))/(1+exp(-(hh/range)^smooth))),df=alpha+1)
								if(repar){
									par0 <- c(alpha,beta,theta,smooth)
								} else{
									par0 <- c(parR0,parGauss0)
								}
								Sigma <- exp(-(as.matrix(dist(coord))/range)^smooth)
								dat <- rmaxidspat(n,coord,parR0,parGauss0,N)
								datU <- apply(dat,c(1,2),pG,parR=parR0)
								
								res <- list(fit1=NA,fit2=NA,time1=NA,time2=NA)
								
								par.file <- paste('Outputs/Sim',sim,'/param_r',r,'_D',D,'_n',n,'_al',alpha,'_be',beta,'_ra',range,'_sm',smooth,'.txt',sep='')
								
								if(start.from.previous.par & file.exists(par.file)){
									init1 <- read.table(file=par.file); init1 <- as.vector(init1[which.min(init1[,ncol(init1)]),-ncol(init1)]); fixed1 <- fixed;
									init2 <- init1; init2[2] <- 0; fixed2 <- fixed; fixed2[2] <- TRUE;
								} else{
									init1 <- par0; fixed1 <- fixed;
									init2 <- init1; init2[2] <- 0; fixed2 <- fixed; fixed2[2] <- TRUE;
									unlink(par.file)
								}
								
								if(repar){
									tr1 <- try( time1 <- system.time( fit1 <- fit.pw.repar(init=init1,datU=datU,coord=coord,cutoff=0.5,fixed=fixed1,optim=TRUE,hessian=FALSE,sandwich=FALSE,eps=10^(-6),print.par.file=par.file,method="Nelder-Mead",control=list(maxit=1000)) ) )
									tr2 <- try( time2 <- system.time( fit2 <- fit.pw.repar(init=init2,datU=datU,coord=coord,cutoff=0.5,fixed=fixed2,optim=TRUE,hessian=FALSE,sandwich=FALSE,eps=10^(-6),print.par.file=par.file,method="Nelder-Mead",control=list(maxit=1000)) ) )
								} else{
									tr1 <- try( time1 <- system.time( fit1 <- fit.pw(init=init1,datU=datU,coord=coord,cutoff=0.5,fixed=fixed1,optim=TRUE,hessian=FALSE,sandwich=FALSE,eps=10^(-6),print.par.file=par.file,method="Nelder-Mead",control=list(maxit=1000)) ) )
									tr2 <- try( time2 <- system.time( fit2 <- fit.pw(init=init2,datU=datU,coord=coord,cutoff=0.5,fixed=fixed2,optim=TRUE,hessian=FALSE,sandwich=FALSE,eps=10^(-6),print.par.file=par.file,method="Nelder-Mead",control=list(maxit=1000)) ) )
								}
								
								if(!is(tr1,"try-error")){
									mle1 <- fit1$mle; 
									if(repar){
										mle1[3] <- -hh*(log((1-(qt(mle1[3]/2,df=mle1[1]+1)^2)/(mle1[1]+1))/(1+(qt(mle1[3]/2,df=mle1[1]+1)^2)/(mle1[1]+1))))^(-1/mle1[4])
									}
									mle.r[i1,i2,i3,i4,i5,i6,] <- mle1
									pw.nllik.r[i1,i2,i3,i4,i5,i6] <- fit1$pw.nllik
									conv.r[i1,i2,i3,i4,i5,i6] <- fit1$convergence
									counts.r[i1,i2,i3,i4,i5,i6,] <- fit1$counts
									time.r[i1,i2,i3,i4,i5,i6] <- time1[3]
									
									res$fit1 <- fit1
									res$time1 <- time1[3]
								}
								
								if(!is(tr2,"try-error")){
									mle2 <- fit2$mle; 
									if(repar){
										mle2[3] <- -hh*(log((1-(qt(mle2[3]/2,df=mle2[1]+1)^2)/(mle2[1]+1))/(1+(qt(mle2[3]/2,df=mle2[1]+1)^2)/(mle2[1]+1))))^(-1/mle2[4])
									}
									mle.r.beta0[i1,i2,i3,i4,i5,i6,] <- mle2
									pw.nllik.r.beta0[i1,i2,i3,i4,i5,i6] <- fit2$pw.nllik
									conv.r.beta0[i1,i2,i3,i4,i5,i6] <- fit2$convergence
									counts.r.beta0[i1,i2,i3,i4,i5,i6,] <- fit2$counts
									time.r.beta0[i1,i2,i3,i4,i5,i6] <- time2[3]
									
                                    res$fit2 <- fit2
									res$time2 <- time2[3]
								}
								save(res,file=paste('Outputs/Sim',sim,'/All/mle_r',r,'_D',D,'_n',n,'_al',alpha,'_be',beta,'_ra',range,'_sm',smooth,'.RData',sep=''))							
							}
						}
					}
				}
			}
		}
	}
	
	return(list(mle.r=mle.r,mle.r.beta0=mle.r.beta0,nllik.r=nllik.r,nllik.r.beta0=nllik.r.beta0,conv.r=conv.r,conv.r.beta0=conv.r.beta0,counts.r=counts.r,counts.r.beta0=counts.r.beta0,time.r=time.r,time.r.beta0=time.r.beta0))
}

simul.list <- mclapply(Rs,FUN=simul.r,mc.cores=ncores)









