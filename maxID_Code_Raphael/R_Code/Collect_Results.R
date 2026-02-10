#####################################################
#####################################################
### Load and Source Necessary Libraries and Files ###
#####################################################
#####################################################

library(mvtnorm)
library(methods)
library(parallel)
library(fields)

BASE <- "~/Documents/Work/05_AssistantProfessor-KAUST/Research/Max-id_Thomas-Emeric/Shaheen_Cluster/Max-id/"
setwd(BASE)

source("R_Code/Tools.R")

###################################
### Main parameters (as inputs) ###
###################################

sim <- 1 # simulation index

if(sim==1){
	Ds <- c(10,15,20,30,50) # dimensions of data (i.e., number of sites)
	ns <- c(50) # numbers of observations
	ranges <- c(0.5) # correlation range parameters
	smooths <- c(1) # correlation smoothness parameters
	alphas <- c(1) # random factor scale parameters
	betas <- c(0,0.5,1) # random factor shape parameters (AD/AI)
	fixed <- c(TRUE,FALSE,FALSE,TRUE) # where the parameters (alpha,beta,range,smooth) fixed to their true values?
	Rs <- c(1:1024) # replications
} else if(sim==2){
	Ds <- c(20) # dimensions of data (i.e., number of sites)
	ns <- c(50) # numbers of observations
	ranges <- c(0.5) # correlation range parameters
	smooths <- c(1) # correlation smoothness parameters
	alphas <- c(1,2,5) # random factor scale parameters
	betas <- c(0,0.5,1) # random factor shape parameters (AD/AI)
	fixed <- c(FALSE,FALSE,FALSE,TRUE) # where the parameters (alpha,beta,range,smooth) fixed to their true values?
	Rs <- c(1:1024) # replications
}

##############################
### Collecting all results ###
##############################

### Parameters
mle <- mle.beta.pos <- mle.beta0 <- array(dim=c(length(Rs),length(Ds),length(ns),length(alphas),length(betas),length(ranges),length(smooths),4))
mle.true <- array(dim=c(length(Ds),length(ns),length(alphas),length(betas),length(ranges),length(smooths),4))
pw.nllik <- pw.nllik.beta.pos <- pw.nllik.beta0 <- array(dim=c(length(Rs),length(Ds),length(ns),length(alphas),length(betas),length(ranges),length(smooths)))
conv <- conv.beta.pos <- conv.beta0 <- array(dim=c(length(Rs),length(Ds),length(ns),length(alphas),length(betas),length(ranges),length(smooths)))
counts <- counts.beta.pos <- counts.beta0 <- array(dim=c(length(Rs),length(Ds),length(ns),length(alphas),length(betas),length(ranges),length(smooths),2))
time <- time.beta.pos <- time.beta0 <- array(dim=c(length(Rs),length(Ds),length(ns),length(alphas),length(betas),length(ranges),length(smooths)))

dimnames(mle) <- dimnames(mle.beta.pos) <- dimnames(mle.beta0) <- list(paste('R=',Rs,sep=''),paste('D=',Ds,sep=''),paste('n=',ns,sep=''),paste('alpha=',alphas,sep=''),paste('beta=',betas,sep=''),paste('range=',ranges,sep=''),paste('smooth=',smooths,sep=''),c("alpha","beta","range","smooth"))
dimnames(mle.true) <- list(paste('D=',Ds,sep=''),paste('n=',ns,sep=''),paste('alpha=',alphas,sep=''),paste('beta=',betas,sep=''),paste('range=',ranges,sep=''),paste('smooth=',smooths,sep=''),c("alpha","beta","range","smooth"))
dimnames(pw.nllik) <- dimnames(pw.nllik.beta.pos) <- dimnames(pw.nllik.beta0) <- dimnames(conv) <- dimnames(conv.beta.pos) <- dimnames(conv.beta0) <- dimnames(time) <- dimnames(time.beta.pos) <- dimnames(time.beta0) <- list(paste('R=',Rs,sep=''),paste('D=',Ds,sep=''),paste('n=',ns,sep=''),paste('alpha=',alphas,sep=''),paste('beta=',betas,sep=''),paste('range=',ranges,sep=''),paste('smooth=',smooths,sep=''))
dimnames(counts) <- dimnames(counts.beta.pos) <- dimnames(counts.beta0) <- list(paste('R=',Rs,sep=''),paste('D=',Ds,sep=''),paste('n=',ns,sep=''),paste('alpha=',alphas,sep=''),paste('beta=',betas,sep=''),paste('range=',ranges,sep=''),paste('smooth=',smooths,sep=''),c("function","gradient"))


for(i1 in 1:length(Ds)){
	D <- Ds[i1]
	for(i2 in 1:length(ns)){
		n <- ns[i2]
		for(i3 in 1:length(alphas)){
			alpha <- alphas[i3]
			for(i4 in 1:length(betas)){
				beta <- betas[i4]
				for(i5 in 1:length(ranges)){
					range <- ranges[i5]
					for(i6 in 1:length(smooths)){
						smooth <- smooths[i6]
						par0 <- c(alpha,beta,range,smooth)
						mle.true[i1,i2,i3,i4,i5,i6,] <- par0
						for(r in 1:length(Rs)){
							file <- paste('Outputs/Sim',sim,'/All/mle_r',r,'_D',D,'_n',n,'_al',alpha,'_be',beta,'_ra',range,'_sm',smooth,'.RData',sep='')
							if(file.exists(file)){	
								tr <- try(load(file))
								if(!is(tr,"is-error")){
									if(!any(is.na(res$fit1))){
										mle1 <- res$fit1$mle
										conv1 <- res$fit1$convergence
										if(!(max(mle1[1:3]) > 19.9 | mle1[4]>2) & (conv1==0 | conv1==10)){
										#if(!(max(mle1[1:3]) > 19.9 | mle1[4]>2)){
											mle.beta.pos[r,i1,i2,i3,i4,i5,i6,] <- mle1
											pw.nllik.beta.pos[r,i1,i2,i3,i4,i5,i6] <- res$fit1$pw.nllik
											conv.beta.pos[r,i1,i2,i3,i4,i5,i6] <- conv1
											counts.beta.pos[r,i1,i2,i3,i4,i5,i6,] <- res$fit1$counts
											time.beta.pos[r,i1,i2,i3,i4,i5,i6] <- res$time1
										}
									}
									if(!any(is.na(res$fit2))){
										mle2 <- res$fit2$mle
										conv2 <- res$fit2$convergence
										if(!(max(mle2[1:3]) > 19.9 | mle2[4]>2) & (conv2==0 | conv2==10)){
										#if(!(max(mle2[1:3]) > 19.9 | mle2[4]>2)){
											mle.beta0[r,i1,i2,i3,i4,i5,i6,] <- mle2
											pw.nllik.beta0[r,i1,i2,i3,i4,i5,i6] <- res$fit2$pw.nllik
											conv.beta0[r,i1,i2,i3,i4,i5,i6] <- conv2
											counts.beta0[r,i1,i2,i3,i4,i5,i6,] <- res$fit2$counts
											time.beta0[r,i1,i2,i3,i4,i5,i6] <- res$time2
										}
									}
								} 
							}
						}
					}
				}
			}
		}
	}
}

ind0 <- pw.nllik.beta0<pw.nllik.beta.pos
ind0 <- ind0[!is.na(ind0)]

mle <- mle.beta.pos; mle[ind0] <- as.vector(mle.beta0[ind0])
pw.nllik <- pw.nllik.beta.pos; pw.nllik[ind0] <- pw.nllik.beta0[ind0]
conv <- conv.beta.pos; conv[ind0] <- conv.beta0[ind0]
counts <- counts.beta.pos + counts.beta0
time <- time.beta.pos + time.beta0

save(mle,file=paste("Outputs/Sim",sim,"/mle.RData",sep=""))
save(mle.beta.pos,file=paste("Outputs/Sim",sim,"/mle-beta-pos.RData",sep=""))
save(mle.beta0,file=paste("Outputs/Sim",sim,"/mle-beta0.RData",sep=""))
save(mle.true,file=paste("Outputs/Sim",sim,"/mle-true.RData",sep=""))
save(pw.nllik,file=paste("Outputs/Sim",sim,"/nllik.RData",sep=""))
save(pw.nllik.beta.pos,file=paste("Outputs/Sim",sim,"/nllik-beta-pos.RData",sep=""))
save(pw.nllik.beta0,file=paste("Outputs/Sim",sim,"/nllik-beta0.RData",sep=""))
save(conv,file=paste("Outputs/Sim",sim,"/conv.RData",sep=""))
save(conv.beta.pos,file=paste("Outputs/Sim",sim,"/conv-beta-pos.RData",sep=""))
save(conv.beta0,file=paste("Outputs/Sim",sim,"/conv-beta0.RData",sep=""))
save(counts,file=paste("Outputs/Sim",sim,"/counts.RData",sep=""))
save(counts.beta.pos,file=paste("Outputs/Sim",sim,"/counts-beta-pos.RData",sep=""))
save(counts.beta0,file=paste("Outputs/Sim",sim,"/counts-beta0.RData",sep=""))
save(time,file=paste("Outputs/Sim",sim,"/time.RData",sep=""))
save(time.beta.pos,file=paste("Outputs/Sim",sim,"/time-beta-pos.RData",sep=""))
save(time.beta0,file=paste("Outputs/Sim",sim,"/time-beta0.RData",sep=""))


my.mean <- function(x,p=0,na.rm=TRUE){ return(mean(x[x>quantile(x,p/2,na.rm=na.rm) & x<quantile(x,1-p/2,na.rm=na.rm)],na.rm=na.rm)) }
my.var <- function(x,p=0,na.rm=TRUE){ return(var(x[x>quantile(x,p/2,na.rm=na.rm) & x<quantile(x,1-p/2,na.rm=na.rm)],na.rm=na.rm)) }
my.sd <- function(x,p=0,na.rm=TRUE){ return(sqrt(my.var(x,p,na.rm))) }

BIAS <- apply(mle,2:8,my.mean,p=0.05,na.rm=TRUE) - mle.true
VAR <- apply(mle,2:8,my.var,p=0.05,na.rm=TRUE)
SD <- apply(mle,2:8,my.sd,p=0.05,na.rm=TRUE)
RMSE <- sqrt(BIAS^2+VAR)

dimnames(BIAS) <- dimnames(VAR) <- dimnames(SD) <- dimnames(RMSE) <- list(paste('D=',Ds,sep=''),paste('n=',ns,sep=''),paste('alpha=',alphas,sep=''),paste('beta=',betas,sep=''),paste('range=',ranges,sep=''),paste('smooth=',smooths,sep=''),c("alpha","beta","range","smooth"))

save(BIAS,file=paste("Outputs/Sim",sim,"/BIAS.RData",sep=""))
save(VAR,file=paste("Outputs/Sim",sim,"/VAR.RData",sep=""))
save(SD,file=paste("Outputs/Sim",sim,"/SD.RData",sep=""))
save(RMSE,file=paste("Outputs/Sim",sim,"/RMSE.RData",sep=""))

fac <- 100
rnd <- 0
order.par <- c(2,3)

show.BIAS <- matrix(paste(round(fac*BIAS[,1,1,,1,1,order.par[1]],rnd),"/",round(fac*BIAS[,1,1,,1,1,order.par[2]],rnd),sep=""),length(Ds),length(betas)); rownames(show.BIAS) <- paste("Ds=",Ds,sep=""); colnames(show.BIAS) <- paste("beta=",betas,sep="")
show.VAR <- matrix(paste(round(fac*VAR[,1,1,,1,1,order.par[1]],rnd),"/",round(fac*VAR[,1,1,,1,1,order.par[2]],rnd),sep=""),length(Ds),length(betas)); rownames(show.VAR) <- paste("Ds=",Ds,sep=""); colnames(show.VAR) <- paste("beta=",betas,sep="")
show.SD <- matrix(paste(round(fac*SD[,1,1,,1,1,order.par[1]],rnd),"/",round(fac*SD[,1,1,,1,1,order.par[2]],rnd),sep=""),length(Ds),length(betas)); rownames(show.SD) <- paste("Ds=",Ds,sep=""); colnames(show.SD) <- paste("beta=",betas,sep="")
show.RMSE <- matrix(paste(round(fac*RMSE[,1,1,,1,1,order.par[1]],rnd),"/",round(fac*RMSE[,1,1,,1,1,order.par[2]],rnd),sep=""),length(Ds),length(betas)); rownames(show.RMSE) <- paste("Ds=",Ds,sep=""); colnames(show.RMSE) <- paste("beta=",betas,sep="")


##########################
### QQPLOTS OF RESULTS ###
##########################

sim <- 1

if(sim==1){
	Ds <- c(10,15,20,30,50) # dimensions of data (i.e., number of sites)
	n <- 50 
	alpha <- 1
	betas <- c(0,0.5,1) # random factor shape parameters (AD/AI)
	range <- 0.5
	smooth <- 1
	
	pdf(paste("Figures/Boxplot_sim",sim,"_alpha",alpha,".pdf",sep=""),width=9,height=6)
	par(mfrow=c(2,3),mgp=c(2,1,0),mar=c(3.1,3.1,3.1,1))
	mle.boxplot <- mle[,,which(ns==n),which(alphas==alpha),,which(ranges==range),which(smooths==smooth),]
	for(i in 1:length(betas)){
		mle.boxplot.beta <- mle.boxplot[,,i,2]
		boxplot(mle.boxplot.beta,col=1+i,ylab=expression("Estimated parameter"~beta),main=bquote(beta~"="~.(betas[i])))
		abline(h=betas[i],col="orange",lwd=2)
	}
	for(i in 1:length(betas)){
		mle.boxplot.range <- mle.boxplot[,,i,3]
		boxplot(mle.boxplot.range,col=1+i,ylab=expression("Estimated range parameter"~lambda),main=bquote(beta~"="~.(betas[i])))
		abline(h=range,col="orange",lwd=2)
	}
	dev.off()
} else{
	D <- 20
	n <- 50
	alphas <- c(1,2,5)
	betas <- c(0,0.5,1)
	range <- 0.5
	smooth <- 1	
	
	pdf(paste("Figures/Boxplot_sim",sim,"_D",D,".pdf",sep=""),width=9,height=9)
	par(mfrow=c(3,3),mgp=c(2,1,0),mar=c(3.1,3.1,3.1,1))
	mle.boxplot <- mle[,which(Ds==D),which(ns==n),,,which(ranges==range),which(smooths==smooth),]
	for(i in 1:length(betas)){
		mle.boxplot.alpha <- mle.boxplot[,,i,1]
		boxplot(mle.boxplot.alpha,col=1+i,ylab=expression("Estimated parameter"~alpha),main=bquote(beta~"="~.(betas[i])))
		segments(x0=c(1:length(alphas))-0.5,x1=c(1:length(alphas))+0.5,y0=alphas,y1=alphas,col="orange",lwd=2)
	}
	for(i in 1:length(betas)){
		mle.boxplot.beta <- mle.boxplot[,,i,2]
		boxplot(mle.boxplot.beta,col=1+i,ylab=expression("Estimated parameter"~beta),main=bquote(beta~"="~.(betas[i])))
		abline(h=betas[i],col="orange",lwd=2)
	}
	for(i in 1:length(betas)){
		mle.boxplot.range <- mle.boxplot[,,i,3]
		boxplot(mle.boxplot.range,col=1+i,ylab=expression("Estimated range parameter"~lambda),main=bquote(beta~"="~.(betas[i])))
		abline(h=range,col="orange",lwd=2)
	}
	dev.off()
}





































