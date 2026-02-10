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

#############################
### EXTREMAL COEFFICIENTS ###
#############################

rho <- 0.5
alphas <- c(1,2,5)
betas <- c(0,0.5,1,2)

z <- -1/log(seq(0.001,0.999,by=0.001))
theta.z <- array(dim=c(length(alphas),length(betas),length(z)))
for(i in 1:length(alphas)){
	alpha <- alphas[i]
	for(j in 1:length(betas)){
		beta <- betas[j]
		parR <- c(alpha,beta)
		print(paste("alpha=",alpha,"; beta=",beta,sep=""))
		z.RW <- qG(exp(-1/z),parR=parR,log=FALSE)
		#theta.z[i,j,] <- z*V(cbind(z.RW,z.RW),rho=rep(rho,length(z)),parR=parR,log=FALSE)
		for(k in 1:length(z.RW)){
			print(k)
			theta.z[i,j,k] <- z[k]*V(c(z.RW[k],z.RW[k]),rho=rho,parR=parR,log=FALSE)
		}
	}
}

pdf(paste("Figures/ExtrCoefs.pdf",sep=""),width=7,height=2.5)
par(mfrow=c(1,length(alphas)),mgp=c(2,1,0),mar=c(3.1,3.1,3.1,1))
for(i in 1:length(alphas)){
	plot(z,theta.z[i,1,],type="l",log="x",ylim=c(1,2),xlab="Level z",ylab=expression("Extremal coefficient "*theta[2]*"(z)=zV(z,z)"),main=bquote(alpha*"="*.(alphas[i])))
	lines(z,theta.z[i,2,],col=2)
	lines(z,theta.z[i,3,],col=3)
	lines(z,theta.z[i,4,],col=4)
	abline(h=c(1,2),col="lightgrey")
}
legend(x="bottomright",legend=c(expression(beta*"=0"),expression(beta*"=0.5"),expression(beta*"=1"),expression(beta*"=2")),lty=1,col=c(1:4))
dev.off()


##########################
### QQPLOTS OF RESULTS ###
##########################
sim <- 1
load(file=paste("Outputs/Sim",sim,"/mle.RData",sep=""))

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
