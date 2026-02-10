#! /usr/bin/Rscript
args <- commandArgs(TRUE)
for (arg in args) eval(parse(text = arg))
rm(arg, args)


########################
########################
### INPUT PARAMETERS ###
########################
########################
# Ds <- c(10,15,20) # dimensions of data (i.e., number of sites)
# ns <- c(50) # numbers of observations
# ranges <- c(0.5,1) # correlation range parameters
# smooths <- c(1) # correlation smoothness parameters
# alphas <- c(1,2,5) # random factor scale parameters
# betas <- c(0,0.5,1) # random factor shape parameters (AD/AI)
# repar <- TRUE # should the model be reparametrized as (alpha,beta,theta(hh),smooth) instead of (alpha,beta,range,smooth), where theta(hh) denotes the bivariate extremal coefficient at distance hh?
# hh <- 0.5 # distance for fixing the extremal coefficient in the reparametrization...
# fixed <- c(FALSE,FALSE,FALSE,TRUE) # whether alpha/beta/range/smooth are fixed to their true values in estimation
# Replics <- c(1:1024) # replications (1024=32*32)
# N <- 10000 # mean number of Poisson points R_i simulated for the approximate simulation...
# sim <- 1 # simulation index
# All <- FALSE # If TRUE, re-simulate even the experiments, which have already been simulated
# start.from.previous.par <- FALSE # If TRUE, the code will start the optimization from the previously estimated value

ncores <- 32 # number of cores for parallel computing
Replic <- length(Replics)
njobs <- ceiling(Replic/ncores)

for (i in 1:njobs) {
	Rs <- Replics[(ncores*(i-1)+1):min(Replic,ncores*i)]
	
    line1 <- "#!/bin/bash -l\n"
    line2 <- "#SBATCH --account=k1241\n"
    line3 <- paste("#SBATCH --output MAXID-SIM",sim,"_D",min(Ds),"-",max(Ds),"_R",min(Rs),"-",max(Rs),"_%J.out\n",sep="")
    line4 <- paste("#SBATCH --error MAXID-SIM",sim,"_D",min(Ds),"-",max(Ds),"_R",min(Rs),"-",max(Rs),"_%J.err\n",sep="")
    line5 <- "#SBATCH --nodes=1\n"
    line6 <- paste("#SBATCH -n ", ncores,"\n", sep = "")
    line7 <- "#SBATCH --time 23:59:59\n"
    line8 <- paste("#SBATCH -J \"MAXID-SIM",sim,"_D",min(Ds),"-",max(Ds),"_R",min(Rs),"-",max(Rs),"\"\n\n", sep = "")
    line9 <- "module load r\n\n"
    line10 <- paste("Rscript /scratch/huserrg/Max-id/R_Code/Simulation.R \"Ds<-c(",paste(Ds,collapse=","),")\" \"ns<-c(",paste(ns,collapse=","),")\" \"ranges<-c(",paste(ranges,collapse=","),")\" \"smooths<-c(",paste(smooths,collapse=","),")\" \"alphas<-c(",paste(alphas,collapse=","),")\" \"betas<-c(",paste(betas,collapse=","),")\" \"repar<-",repar,"\" \"hh<-",hh,"\" \"fixed<-c(",paste(fixed,collapse=","),")\" \"Rs<-c(",paste(Rs,collapse=","),")\" \"N<-",N,"\" \"sim<-",sim,"\" \"All<-",All,"\" \"start.from.previous.par <-", start.from.previous.par,"\" \"ncores<-",ncores,"\" \n",sep="") 
    						
  	sub <- paste(line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, sep = "")
    if (file.exists("Simulation.sub")) {
		unlink("Simulation.sub")
	}
	cat(sub, file = "Simulation.sub")
	system("sbatch Simulation.sub")
}


