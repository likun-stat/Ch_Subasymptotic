#######################################################################################################
### intensity of the radial variable R, random number generator
#######################################################################################################

#Density function (PDF) of the point process of R
dF <- function(r,parR,log=FALSE){
	alpha <- parR[1]
	beta <- parR[2]
	logval <- c()
	logval[is.na(r)] <- NA
	ind <- r>0 & !is.na(r)
	if(beta==0){
		logval[ind] <- -(alpha+1)*log(r[ind])+log(alpha)
	} else{
		logval[ind] <- -alpha/beta*(r[ind]^beta-1)+log(beta*r[ind]^(-beta-1)+alpha*r[ind]^(-1))
	}
	logval[r<=0 & !is.na(r)] <- -Inf
	if(log){
		return( logval )	
	} else{
		return( exp(logval) )
	}
}

#the intensity dF integrated over [x,infty)
upperF <- function(x,parR,log=FALSE){
	alpha <- parR[1]
	beta <- parR[2]
	logval <- c()
	logval[is.na(x)] <- NA
	ind <- x>0 & !is.na(x)
	if(beta==0){
		logval[ind] <- -alpha*log(x[ind])
	} else{
		logval[ind] <- -beta*log(x[ind]) - alpha/beta*(x[ind]^beta-1)
	}
	logval[x<=0 & !is.na(x)] <- Inf
	if(log){
		return( logval )	
	} else{
		return( exp(logval) )
	}
}

#inverse function of the intensity dF integrated over [x,infty)
upperFinv <- function(y,parR,log=FALSE){
	alpha <- parR[1]
	beta <- parR[2]
	logval <- c()
	for(i in 1:length(y)){
		if(!is.na(y[i]) & y[i]>0){
			fun <- function(x){
				return(log(y[i])-upperF(x=exp(x),parR=parR,log=TRUE))
			}
			logval[i] <- uniroot(f=fun,interval=c(-3,3),extendInt='yes')$root	
		}
	}
	if(log){
		return( logval )	
	} else{
		return( exp(logval) )
	}
}

#generate n points from the point process R_1>R_2>R_3>... (in decreasing order) on [eps,infty)
rF <- function(N, parR){
	return(sort( upperFinv(y=N*runif(N),parR=parR,log=FALSE) ,decreasing = TRUE))
}

#generate (bivariate) realizations from the max-id model (generate points of the point process and then take the componentwise max)
rmaxid <- function(n, parR, rho, N=1000){ # N is the number of Poisson points to simulate...
	Z <- matrix(ncol=2,nrow=n)
	A <- t(chol(matrix(c(1,rho,rho,1),2,2)))
	for (k in 1:n){
		R <- rF(N,parR)
		Z[k,] <- apply(R*t(A%*%matrix(rnorm(2*N),ncol=N,nrow=2)),2,max)
	}
	return(Z)
}

#################################################################################################################################################
### function V and its derivatives
#################################################################################################################################################
	
#density function of the point process (also -V12)
library(mvtnorm)
mV12 <- function(x,rho,parR,log=FALSE){
	if(!is.matrix(x)){
		x <- matrix(x,nrow=1)
	}
	dGi <- function(xi){
		fun <- function(r,parR){
			X <- matrix(xi,ncol=2,nrow=length(r),byrow=TRUE)
			return(exp(dmvnorm(sign(X)*exp(log(abs(X))-log(r)),sigma=matrix(c(1,rho,rho,1),2,2),log=TRUE)-2*log(r)+dF(r,parR,log=TRUE)))
		}
		val <- integrate(fun,lower=0,upper=Inf,parR=parR,rel.tol=10^(-3),stop.on.error=FALSE)$value
		return(val)
	}
	I <- apply(is.na(x),1,sum)==0
	val <- c()
	val[I] <- apply(matrix(x[I,],nrow=sum(I)),1,dGi)
	val[!I] <- NA
	if(log){
		return( log(val) )	
	} else{
		return( val )
	}
}

#Partial derivatives of -V with respect to the k=1,2 element
mVk <- function(x,k,rho,parR,log=FALSE){
	if(!is.list(x)){
		if(!is.matrix(x)){
			x <- matrix(x,nrow=1)
		}
		x <- as.list(data.frame(t(x)))
	}
	#k is the vector that contains the index for partial derivatives: I will make k a list
	if(!is.list(k)){
		k <- as.list(k)
	}
	
	g <- function(xi,ki){
		#function of r to be integrated (needs to be defined for different values of r (= r is a vector))
		fun <- function(r,parR){
			return( exp( pnorm(sign(xi[-ki]-rho*xi[ki])*exp(log(abs(xi[-ki]-rho*xi[ki]))-log(r)),mean=0,sd=sqrt(1-rho^2),log.p=TRUE) + dnorm(sign(xi[ki])*exp(log(abs(xi[ki]))-log(r)),log=TRUE) -log(r) + dF(r,parR,log=TRUE) ) )
		}
		val <- integrate(fun,lower=0,upper=Inf,parR=parR,rel.tol=10^(-3),stop.on.error=FALSE)$value
		return(val)
	}
	val <- c()
	I <- mapply(function(x) sum(is.na(x))==0,x)
	val[I] <- mapply(g,xi=x[I],ki=k)
	val[!I] <- NA
	
	if(log){
		return( log(val) )	
	} else{
		return( val )
	}
}

# V (integral of point process density over outer region)
V <- function(x,rho,parR,log=FALSE){
	if(!is.list(x)){
		if(!is.matrix(x)){
			x <- matrix(x,nrow=1)
		}
		x <- as.list(data.frame(t(x)))
	}

	g <- function(xi){
		if (any(xi<=0)){return(Inf)}
		#function of r to be integrated (needs to be defined for different values of r (= r is a vector))
		fun <- function(r,parR){
			X <- matrix(xi,ncol=2,nrow=length(r),byrow=TRUE)
			logpmv <- log( 1-apply(sign(X)*exp(log(abs(X))-log(r)),1,function(u) pmvnorm(upper=u,sigma=matrix(c(1,rho,rho,1),2,2))) )
			# logpmv <- log( 1-apply(sign(X)*exp(log(abs(X))-log(r)),1,function(u){
				# set.seed(1987264)
				# return(.C("pmvnormE", as.double(u), as.integer(2), as.double(c(matrix(c(1,rho,rho,1),2,2))), as.double(0), as.double(0), as.integer(100), as.integer(1))[[4]]) ### MEANS I USE ONLY 100 POINTS IN THE MONTE CARLO ALGORITHM
				# }
				# ) )		
			return( exp( logpmv + dF(r,parR,log=TRUE) ) )
		}
		val <- integrate(fun,lower=0,upper=Inf,parR=parR,rel.tol=10^(-3),stop.on.error=FALSE)$value
		return(val)
	}
	val <- c()
	I <- mapply(function(x) sum(is.na(x))==0,x)
	val[I] <- mapply(g,xi=x[I])
	val[!I] <- NA
	
	if(log){
		return( log(val) )	
	} else{
		return( val )
	}
}

#################################################################################################################################################
### marginal distribution of the max-id model, its density and quantile function: pG, dG, qG
#################################################################################################################################################
pG <- function(x,parR,log=FALSE){
	g <- function(xi){
		if (xi<=0){return(Inf)}
		#function of r to be integrated (needs to be defined for different values of r (= r is a vector))
		fun <- function(r,parR){
			logp <- log( 1-pnorm(sign(xi)*exp(log(abs(xi))-log(r))) )
			return( exp( logp + dF(r,parR,log=TRUE) ) )
		}
		val <- integrate(fun,lower=0,upper=Inf,parR=parR,rel.tol=10^(-3),stop.on.error=FALSE)$value
		return(val)
	}
	val <- c()
	I <- !is.na(x) 
	val[I] <- exp(-apply(matrix(x[I],ncol=1),1,g))
	val[!I] <- NA
	logval <- log(val)
	
	if(log){
		return( logval )	
	} else{
		return( exp(logval) )
	}	
}

dG <- function(x,parR,log=FALSE){
	g <- function(xi){
		if (xi<=0){return(0)}
		fun <- function(r,parR){
			return(exp(dnorm(sign(xi)*exp(log(abs(xi))-log(r)),log=TRUE)-log(r)+dF(r,parR,log=TRUE)))
		}
		val <- integrate(fun,lower=0,upper=Inf,parR=parR,rel.tol=10^(-3),stop.on.error=FALSE)$value
		return(val)
	}
	I <- !is.na(x) 
	val0 <- c()
	val0[I] <- apply(matrix(x[I],ncol=1),1,g)
	val0[!I] <- NA
	logval <- log(val0)+pG(x,parR,log=TRUE)  #density is -V_1 exp(-V)
	if(log){
		return( logval )
	} else{
		return( exp(logval) )
	}	
}

qG <- function(p,parR,log=FALSE){
	fun <- function(x,p,parR){
		return( pG(exp(x),parR,log=TRUE)-log(p) )
	}
	I <- !is.na(p) & (p>0) & (p<1)
	logval <- c()
	logval[!is.na(p) & p==0] <- -Inf
	logval[!is.na(p) & p==1] <- Inf
	logval[I] <- apply(matrix(p[I],ncol=1),1,function(pi) uniroot(fun,interval=c(-3,3),p=pi,parR=parR,extendInt='yes')$root)
	if(log){
		return( logval )
	} else{
		return( exp(logval) )
	}
}


##################################################################################################################################
### spatial simulations of max-id model (multivariate based on a stable correlation function, using coordinates of stations) 
##################################################################################################################################
rmaxidspat <- function(n, coord, parR, parGauss, N=1000){
  library(doParallel)
  registerDoParallel(8)
	D <- nrow(coord)
	Z <- matrix(ncol=D,nrow=n)
	Sigma <- exp(-(as.matrix(dist(coord))/parGauss[1])^parGauss[2])
	A <- t(chol(Sigma))
	Z = foreach (k = 1:n, .combine=rbind) %dopar% {
		R <- rF(N,parR)
		apply(R*t(A%*%matrix(rnorm(D*N),ncol=N,nrow=D)),2,max)
	}
	return(Z)
}

#####################################
### Pairwise likelihood inference ###
#####################################

#pairwise copula likelihood
pw.nllik <- function(par,datU,coord,doSum=TRUE,cutoff=Inf,print.par.file=NULL){ #negative pairwise log likelihood (for the copula)
	parR <- par[1:2]
	parGauss <- par[3:4]
	if (parR[1]<=0 | parR[1]>20 | parR[2]<0 | parR[2]>20 | parGauss[1]<=0 | parGauss[1]>20 | parGauss[2]<=0 | parGauss[2]>2){return(Inf)}	
	
	#marginal transformation
	XDAT <- apply(datU,c(1,2),qG,parR)
	
	#pairs and corresponding rho: put everything in a matrix to which I will apply the likelihood of bivariate data
	D <- nrow(coord)
	pair <- expand.grid(1:D,1:D)
    pair <- pair[,2:1]
    pair <- pair[pair[,1]<pair[,2],]
    pair <- matrix(c(pair[[1]],pair[[2]]),ncol=2)
    dist.pair <- as.matrix(dist(coord))[pair]
    ind <- which(dist.pair<cutoff)
    new.pair <- pair[ind,]
    new.dist.pair <- dist.pair[ind]
    if(is.null(nrow(new.pair))) new.pair <- matrix(new.pair, nrow=1)
	rho.pair <- exp(-(new.dist.pair/parGauss[1])^parGauss[2])
	
	#fix random seed (and save the current random seed to restore it at the end)
	# oldSeed <- get(".Random.seed", mode="numeric", envir=globalenv())
	# set.seed(65413721)
	
	#I use a loop over the pairs (each function uses apply in it, so loops are OK, I can't make faster)
	val <- matrix(nrow=nrow(XDAT),ncol=nrow(new.pair)) ### stores pairwise likelihood contributions for each replicate (rows) and each pair (columns)
	for (k in 1:nrow(new.pair)){
	  # if(k %% 50 == 0) cat("k=", k, '\n')
		val[,k] <- V(XDAT[,new.pair[k,]],rho.pair[k],parR)-log( mVk(XDAT[,new.pair[k,]],k=rep(1,nrow(XDAT)),rho.pair[k],parR)*mVk(XDAT[,new.pair[k,]],k=rep(2,nrow(XDAT)),rho.pair[k],parR) + mV12(XDAT[,new.pair[k,]],rho.pair[k],parR) ) + apply(apply(XDAT[,new.pair[k,]],c(1,2),dG,parR=parR,log=TRUE),1,sum)
	}
	
	#restore random seed to its previous value
	# assign(".Random.seed", oldSeed, envir=globalenv())
	
	if(!is.null(print.par.file)){
		cat(c(par,sum(val),"\n"),file=print.par.file,append=TRUE)
	}
	
	if(doSum){
		return(sum(val)) ### sums all contributions over replicates and pairs
	} else{
		return(apply(val,1,sum)) ### sums all contributions over pairs, but NOT over replicates
	}
}


pw.nllik_fixR <- function(par,datU,coord,doSum=TRUE,cutoff=Inf,print.par.file=NULL){ #negative pairwise log likelihood (for the copula)
  parR <- par[1:2]
  parGauss <- par[3:4]
  if (parR[1]<=0 | parR[1]>20 | parR[2]<0 | parR[2]>20 | parGauss[1]<=0 | parGauss[1]>20 | parGauss[2]<=0 | parGauss[2]>2){return(Inf)}	
  
  #marginal transformation
  XDAT <- apply(datU,c(1,2),qG,parR)
  
  #pairs and corresponding rho: put everything in a matrix to which I will apply the likelihood of bivariate data
  D <- nrow(coord)
  pair <- expand.grid(1:D,1:D)
  pair <- pair[,2:1]
  pair <- pair[pair[,1]<pair[,2],]
  pair <- matrix(c(pair[[1]],pair[[2]]),ncol=2)
  dist.pair <- as.matrix(dist(coord))[pair]
  ind <- which(dist.pair<cutoff)
  new.pair <- pair[ind,]
  new.dist.pair <- dist.pair[ind]
  rho.pair <- exp(-(new.dist.pair/parGauss[1])^parGauss[2])
  
  #fix random seed (and save the current random seed to restore it at the end)
  # oldSeed <- get(".Random.seed", mode="numeric", envir=globalenv())
  # set.seed(65413721)
  
  #I use a loop over the pairs (each function uses apply in it, so loops are OK, I can't make faster)
  val <- matrix(nrow=nrow(XDAT),ncol=nrow(new.pair)) ### stores pairwise likelihood contributions for each replicate (rows) and each pair (columns)
  for (k in 1:nrow(new.pair)){
    # if(k %% 50 == 0) cat("k=", k, '\n')
    val[,k] <- V(XDAT[,new.pair[k,]],rho.pair[k],parR)-log( mVk(XDAT[,new.pair[k,]],k=rep(1,nrow(XDAT)),rho.pair[k],parR)*mVk(XDAT[,new.pair[k,]],k=rep(2,nrow(XDAT)),rho.pair[k],parR) + mV12(XDAT[,new.pair[k,]],rho.pair[k],parR) ) + apply(apply(XDAT[,new.pair[k,]],c(1,2),dG,parR=parR,log=TRUE),1,sum)
  }
  
  #restore random seed to its previous value
  # assign(".Random.seed", oldSeed, envir=globalenv())
  
  if(!is.null(print.par.file)){
    cat(c(par,sum(val),"\n"),file=print.par.file,append=TRUE)
  }
  
  if(doSum){
    return(sum(val)) ### sums all contributions over replicates and pairs
  } else{
    return(apply(val,1,sum)) ### sums all contributions over pairs, but NOT over replicates
  }
}


#fit the copula model using pairwise likelihood
fit.pw <- function(init,datU,coord,cutoff=Inf,fixed=rep(FALSE,4),optim=TRUE,hessian=TRUE,sandwich=TRUE,eps=10^(-6),print.par.file=NULL,...){ #fit max-id copula model by negative pairwise log likelihood
	if(optim==TRUE){
		pw.nllik2 <- function(par,datU,coord){
			par2 <- init
			par2[which(!fixed)] <- par
			val <- pw.nllik(par2,datU,coord,doSum=TRUE,cutoff=cutoff,print.par.file=print.par.file)
			return(val)
		}
		init2 <- init[which(!fixed)]
		fit <- optim(init2,pw.nllik2,datU=datU,coord=coord,hessian=hessian,...)
		cat(fit$par)
		
		res <- list()
		mle <- c(); mle[!fixed] <- fit$par; mle[fixed] <- init[fixed]; res$mle <- mle
		res$pw.nllik <- fit$val
		res$convergence <- fit$convergence
		res$hessian <- fit$hessian
		res$counts <- fit$counts
		if(sandwich & hessian){
			res$inv.hess <- solve(res$hessian)
			grad.pw.nllik <- function(par,datU,coord){
				par2 <- init
				par2[which(!fixed)] <- par
				grad <- matrix(nrow=nrow(datU),ncol=sum(!fixed))
				for(i in 1:sum(!fixed)){
					par2p <- par2; par2p[i] <- par2[i]+eps
					par2m <- par2; par2m[i] <- par2[i]-eps
					grad[,i] <- (pw.nllik(par2p,datU,coord,doSum=FALSE,cutoff=cutoff)-pw.nllik(par2m,datU,coord,doSum=FALSE,cutoff=cutoff))/(2*eps)
				}
				return(grad)
			}
			res$grad <- grad.pw.nllik(fit$par,datU,coord)
			# res$var.grad <- apply(res$grad,2,var)
			res$var.grad <- cov(res$grad)
			res$sandwich <- res$inv.hess%*%res$var.grad%*%res$inv.hess
			res$sd.mle <- sqrt(diag(res$sandwich))
		}
		return(res)
	} else{
		pw.nllik(init,datU,coord,doSum=TRUE,cutoff=cutoff)	
	}
}








#pairwise copula likelihood, when reparametrizing as (alpha,beta,theta(h),smooth) instead of (alpha,beta,range,smooth) for some fixed distance h
pw.nllik.repar <- function(par,datU,coord,doSum=TRUE,cutoff=Inf,hh=0.5,print.par.file=NULL){ #negative pairwise log likelihood (for the copula)
	parR <- par[1:2]
	parGauss <- par[3:4]
	parGauss[1] <- -hh*(log((1-(qt(par[3]/2,df=parR[1]+1)^2)/(parR[1]+1))/(1+(qt(par[3]/2,df=parR[1]+1)^2)/(parR[1]+1))))^(-1/parGauss[2]) ### getting the range parameter from extremal coefficient...
	if(any(is.na(parR)) | any(is.na(parGauss))){
		return(Inf)
	} else{
		if (parR[1]<=0 | parR[1]>20 | parR[2]<0 | parR[2]>20 | parGauss[1]<=0 | parGauss[1]>20 | parGauss[2]<=0 | parGauss[2]>2 ){
			return(Inf)
		}
	}	
	
	#marginal transformation
	XDAT <- apply(datU,c(1,2),qG,parR)
	
	#pairs and corresponding rho: put everything in a matrix to which I will apply the likelihood of bivariate data
	D <- nrow(coord)
	pair <- expand.grid(1:D,1:D)
    pair <- pair[,2:1]
    pair <- pair[pair[,1]<pair[,2],]
    pair <- matrix(c(pair[[1]],pair[[2]]),ncol=2)
    dist.pair <- as.matrix(dist(coord))[pair]
    ind <- which(dist.pair<cutoff)
    new.pair <- pair[ind,]
    new.dist.pair <- dist.pair[ind]
	rho.pair <- exp(-(new.dist.pair/parGauss[1])^parGauss[2])
	
	#fix random seed (and save the current random seed to restore it at the end)
	oldSeed <- get(".Random.seed", mode="numeric", envir=globalenv())
	set.seed(65413721)
	
	#I use a loop over the pairs (each function uses apply in it, so loops are OK, I can't make faster)
	val <- matrix(nrow=nrow(XDAT),ncol=nrow(new.pair)) ### stores pairwise likelihood contributions for each replicate (rows) and each pair (columns)
	for (k in 1:nrow(new.pair)){
		val[,k] <- V(XDAT[,new.pair[k,]],rho.pair[k],parR)-log( mVk(XDAT[,new.pair[k,]],k=rep(1,nrow(XDAT)),rho.pair[k],parR)*mVk(XDAT[,new.pair[k,]],k=rep(2,nrow(XDAT)),rho.pair[k],parR) + mV12(XDAT[,new.pair[k,]],rho.pair[k],parR) ) + apply(apply(XDAT[,new.pair[k,]],c(1,2),dG,parR=parR,log=TRUE),1,sum)
	}
	
	#restore random seed to its previous value
	assign(".Random.seed", oldSeed, envir=globalenv())
	
	if(!is.null(print.par.file)){
		cat(c(par,sum(val),"\n"),file=print.par.file,append=TRUE)
	}
	
	if(doSum){
		return(sum(val)) ### sums all contributions over replicates and pairs
	} else{
		return(apply(val,1,sum)) ### sums all contributions over pairs, but NOT over replicates
	}
}

#fit the copula model using pairwise likelihood, when reparametrizing as (alpha,beta,theta(h),smooth) instead of (alpha,beta,range,smooth) for some fixed distance h
fit.pw.repar <- function(init,datU,coord,cutoff=Inf,hh=0.5,fixed=rep(FALSE,4),optim=TRUE,hessian=TRUE,sandwich=TRUE,eps=10^(-6),print.par.file=NULL,...){ #fit max-id copula model by negative pairwise log likelihood
	if(optim==TRUE){
		pw.nllik2 <- function(par,datU,coord){
			par2 <- init
			par2[which(!fixed)] <- par
			val <- pw.nllik.repar(par2,datU,coord,doSum=TRUE,cutoff=cutoff,hh=hh,print.par.file=print.par.file)
			return(val)
		}
		init2 <- init[which(!fixed)]
		fit <- optim(init2,pw.nllik2,datU=datU,coord=coord,hessian=hessian,...)
		
		res <- list()
		mle <- c(); mle[!fixed] <- fit$par; mle[fixed] <- init[fixed]; res$mle <- mle
		res$pw.nllik <- fit$val
		res$convergence <- fit$convergence
		res$hessian <- fit$hessian
		res$counts <- fit$counts
		if(sandwich & hessian){
			res$inv.hess <- solve(res$hessian)
			grad.pw.nllik <- function(par,datU,coord){
				par2 <- init
				par2[which(!fixed)] <- par
				grad <- matrix(nrow=nrow(datU),ncol=sum(!fixed))
				for(i in 1:sum(!fixed)){
					par2p <- par2; par2p[i] <- par2[i]+eps
					par2m <- par2; par2m[i] <- par2[i]-eps
					grad[,i] <- (pw.nllik(par2p,datU,coord,doSum=FALSE,cutoff=cutoff)-pw.nllik(par2m,datU,coord,doSum=FALSE,cutoff=cutoff))/(2*eps)
				}
				return(grad)
			}
			res$grad <- grad.pw.nllik(fit$par,datU,coord)
			res$var.grad <- apply(res$grad,2,var)
			res$sandwich <- res$inv.hess%*%res$var.grad%*%res$inv.hess
			res$sd.mle <- sqrt(diag(res$sandwich))
		}
		return(res)
	} else{
		pw.nllik(init,datU,coord,doSum=TRUE,cutoff=cutoff)	
	}
}




















