library(ismev)
library(evd)
library(maps)
theta.z <- function(dataFrech,level.z=NULL,nlevel=1000,xlim=NULL,ylim=NULL,PLOT=TRUE,add=FALSE,sd=TRUE,addLS=TRUE,col="black",colsd="black",polysd=TRUE,...){
  S <- ncol(dataFrech)
  n <- nrow(dataFrech)
  maxFrech <- apply(dataFrech,1,max)
  if(is.null(level.z)){
    level.z <- seq(min(maxFrech),max(maxFrech),length=nlevel)		
  }
  if(is.null(xlim)){
    xlim <- range(level.z)
  }
  if(is.null(ylim)){
    ylim <- c(1,S)
  }
  theta.z <- sd.z <- c()	
  for(i in 1:length(level.z)){
    p.z <- mean(maxFrech<=level.z[i])
    theta.z[i] <- min(S,max(1,-level.z[i]*log(p.z)))
    sd.z[i] <- level.z[i]*sqrt((1-p.z)/(p.z*n)) ## Delta method...
  }
  if(is.null(colsd)){
    colsd <- col
  }
  lmfit <- lm(theta.z~log(level.z))
  if(PLOT){
    if(!add){
      plot(level.z,theta.z,type="l",xlab="Level z",ylab=expression("Level-dependent extremal coefficient"~theta[D]*"(z)"),xlim=xlim,ylim=ylim,log="x",col=col,...)
      abline(h=c(1,S),col="lightgrey",lty=2)
    } else{
      lines(level.z,theta.z,col=col,...)
    }
    if(sd){
      if(polysd){
        polygon(c(level.z,level.z[length(level.z):1]),c(theta.z-qnorm(0.975)*sd.z,theta.z[length(level.z):1]+qnorm(0.975)*sd.z[length(level.z):1]),col=colsd,border=NA)
        lines(level.z,theta.z,col=col)
      } else{
        lines(level.z,theta.z-qnorm(0.975)*sd.z,lty=2,col=colsd,...)
        lines(level.z,theta.z+qnorm(0.975)*sd.z,lty=2,col=colsd,...)
      }
    }
    if(addLS){
      lines(exp(seq(min(log(level.z))-1,max(log(level.z))+1,length=1000)),lmfit$coef[1]+lmfit$coef[2]*seq(min(log(level.z))-1,max(log(level.z))+1,length=1000),col="red")
    }
  }
  
  return(list(theta=theta.z,intercept=lmfit$coef[1],slope=lmfit$coef[2]))
}

######### INDIVIDUAL PARAMETRIC GEV FITS ##########
GEV.to.Frech <- function(dat){
  fit <- gev.fit(dat,show=FALSE)
  return(-1/log(pgev(dat,loc=fit$mle[1],scale=fit$mle[2],shape=fit$mle[3])))
}
daily.data.Frech <- apply(daily.data,2,GEV.to.Frech)
weekly.data.Frech <- apply(weekly.data,2,GEV.to.Frech)
monthly.data.Frech <- apply(monthly.data,2,GEV.to.Frech)
### This sometimes doesn't work well because of the small sample size of monthly data (and also the shape parameters may be quite different).......

######### NON-PARAMETRIC ESTIMATION + 'EXTREMAL INDEX' (constant for all station) ##########
dist.cdfs <- function(theta,dat1,dat2){ ### Here we assume that F_2 = \hat F_1^theta, where theta is the same for all stations...
  unif2.a <- unif2.b <- matrix(nrow=nrow(dat2),ncol=ncol(dat2))
  for(j in 1:ncol(dat2)){
    cdf1 <- ecdf(dat1[,j])
    cdf2 <- ecdf(dat2[,j])
    unif2.a[,j] <- cdf2(dat2[,j])*(nrow(dat2)/(nrow(dat2)+1))
    unif2.b[,j] <- (cdf1(dat2[,j])*(nrow(dat1)/(nrow(dat1)+1)))^theta		
  }
  return(sum((unif2.a-unif2.b)^2))
}

######### NON-PARAMETRIC ESTIMATION + 'EXTREMAL INDEX' (specific to each station) ##########
dist.cdfs <- function(theta,dat1,dat2){ ### Here we assume that F_2 = \hat F_1^theta, where theta is different for each station...
  cdf1 <- ecdf(dat1)
  cdf2 <- ecdf(dat2)
  unif2.a <- cdf2(dat2)*(length(dat2)/(length(dat2)+1))
  unif2.b <- (cdf1(dat2)*(length(dat1)/(length(dat1)+1)))^theta		
  
  return(sum((unif2.a-unif2.b)^2))
}

######### GEV FITS + 'EXTREMAL INDEX' (specific to each station) ##########
fitGEVs <- function(daily.dat,weekly.dat,monthly.dat){
  S <- ncol(daily.dat)
  mles <- matrix(ncol=ncol(daily.dat),nrow=4)
  for(j in 1:S){
    daily.dat.j <- daily.dat[,j]
    weekly.dat.j <- weekly.dat[,j]
    monthly.dat.j <- monthly.dat[,j]
    
    jointGEV.negloglik <- function(par,daily.dat.j,weekly.dat.j,monthly.dat.j){
      mu <- par[1]; sig <- par[2]; xi <- par[3]
      th <- par[4]
      th.weekly <- th*7 ## weekly blocks are of size approx 7
      th.monthly <- th*30 ## monthly blocks are of size approx 30
      if(sig > 0 & th.weekly>1 & th.monthly>th.weekly){
        loglik.daily <- dgev(daily.dat.j,loc=mu,scale=sig,shape=xi,log=TRUE)
        loglik.weekly <- dgev(weekly.dat.j,loc=mu-sig*(1-th.weekly^xi)/xi,scale=sig*th.weekly^xi,shape=xi,log=TRUE)
        loglik.monthly <- dgev(monthly.dat.j,loc=mu-sig*(1-th.monthly^xi)/xi,scale=sig*th.monthly^xi,shape=xi,log=TRUE)
        loglik <- sum(loglik.daily)+sum(loglik.weekly)+sum(loglik.monthly)
        return(-loglik)
      } else{
        return(Inf)
      }
    }
    fitGEV <- function(daily.dat.j,weekly.dat.j,monthly.dat.j){
      fit0 <- gev.fit(daily.dat.j,show=FALSE)
      init <- c(fit0$mle,1)
      fit <- optim(par=init,fn=jointGEV.negloglik,daily.dat.j=daily.dat.j,weekly.dat.j=weekly.dat.j,monthly.dat.j=monthly.dat.j,method="Nelder-Mead",control=list(maxit=1000),hessian=FALSE)
      return(fit$par)
    }
    mles[,j] <- fitGEV(daily.dat.j,weekly.dat.j,monthly.dat.j)
  }
  return(mles)
}

######### GEV FITS + 'EXTREMAL INDICES' for weeks/months (specific to each station) ##########
fitGEVs <- function(daily.dat,weekly.dat,monthly.dat){
  S <- ncol(daily.dat)
  mles <- matrix(ncol=ncol(daily.dat),nrow=5)
  for(j in 1:S){
    daily.dat.j <- daily.dat[,j]
    weekly.dat.j <- weekly.dat[,j]
    monthly.dat.j <- monthly.dat[,j]
    
    jointGEV.negloglik <- function(par,daily.dat.j,weekly.dat.j,monthly.dat.j){
      mu <- par[1]; sig <- par[2]; xi <- par[3]
      th.weekly <- par[4]; th.monthly <- par[5]
      if(sig > 0 & th.weekly>1 & th.monthly>th.weekly){
        loglik.daily <- dgev(daily.dat.j,loc=mu,scale=sig,shape=xi,log=TRUE)
        loglik.weekly <- dgev(weekly.dat.j,loc=mu-sig*(1-th.weekly^xi)/xi,scale=sig*th.weekly^xi,shape=xi,log=TRUE)
        loglik.monthly <- dgev(monthly.dat.j,loc=mu-sig*(1-th.monthly^xi)/xi,scale=sig*th.monthly^xi,shape=xi,log=TRUE)
        loglik <- sum(loglik.daily)+sum(loglik.weekly)+sum(loglik.monthly)
        return(-loglik)
      } else{
        return(Inf)
      }
    }
    fitGEV <- function(daily.dat.j,weekly.dat.j,monthly.dat.j){
      fit0 <- gev.fit(daily.dat.j,show=FALSE)
      init <- c(fit0$mle,7,30)
      fit <- optim(par=init,fn=jointGEV.negloglik,daily.dat.j=daily.dat.j,weekly.dat.j=weekly.dat.j,monthly.dat.j=monthly.dat.j,method="Nelder-Mead",control=list(maxit=1000),hessian=FALSE)
      return(fit$par)
    }
    mles[,j] <- fitGEV(daily.dat.j,weekly.dat.j,monthly.dat.j)
  }
  return(mles)
}

######### GEV FITS with constant shape parameter but individual location/scale parameters (specific to each station) ##########
fitGEVs <- function(daily.dat,weekly.dat,monthly.dat){
  S <- ncol(daily.dat)
  mles <- matrix(ncol=ncol(daily.dat),nrow=7)
  for(j in 1:S){
    daily.dat.j <- daily.dat[,j]
    weekly.dat.j <- weekly.dat[,j]
    monthly.dat.j <- monthly.dat[,j]
    
    jointGEV.negloglik <- function(par,daily.dat.j,weekly.dat.j,monthly.dat.j){
      mu.daily <- par[1]; sig.daily <- par[2]; 
      mu.weekly <- par[3]; sig.weekly <- par[4]; 
      mu.monthly <- par[5]; sig.monthly <- par[6]; 
      xi <- par[7]
      if(sig.daily>0 & sig.weekly>0 & sig.monthly>0){
        loglik.daily <- dgev(daily.dat.j,loc=mu.daily,scale=sig.daily,shape=xi,log=TRUE)
        loglik.weekly <- dgev(weekly.dat.j,loc=mu.weekly,scale=sig.weekly,shape=xi,log=TRUE)
        loglik.monthly <- dgev(monthly.dat.j,loc=mu.monthly,scale=sig.monthly,shape=xi,log=TRUE)
        loglik <- sum(loglik.daily)+sum(loglik.weekly)+sum(loglik.monthly)
        return(-loglik)
      } else{
        return(Inf)
      }
    }
    fitGEV <- function(daily.dat.j,weekly.dat.j,monthly.dat.j){
      fit0.daily <- gev.fit(daily.dat.j,show=FALSE)
      fit0.weekly <- gev.fit(weekly.dat.j,show=FALSE)
      fit0.monthly <- gev.fit(monthly.dat.j,show=FALSE)
      init <- c(fit0.daily$mle[1:2],fit0.weekly$mle[1:2],fit0.monthly$mle[1:2],mean(fit0.daily$mle[3],fit0.weekly$mle[3],fit0.monthly$mle[3]))
      fit <- optim(par=init,fn=jointGEV.negloglik,daily.dat.j=daily.dat.j,weekly.dat.j=weekly.dat.j,monthly.dat.j=monthly.dat.j,method="Nelder-Mead",control=list(maxit=1000),hessian=FALSE)
      return(fit$par)
    }
    mles[,j] <- fitGEV(daily.dat.j,weekly.dat.j,monthly.dat.j)
  }
  return(mles)
}



wd <- "~/Dropbox/maxid-sharedfolder/R_Code_Raphael/R_Code/Wind_Analysis/"
setwd(wd)
load("winddata.Rdata")

daily.data <- winddata$datamat
dates <- strptime(as.character(winddata$dates),format="%Y%m%d")
all.dates <- strptime(as.character(seq(from=min(as.Date(dates)),to=max(as.Date(dates)),by=1)),format="%Y-%m-%d")
all.years <- 1900+all.dates$year
all.months <- 1+all.dates$mon
all.days <- all.dates$mday

S <- ncol(daily.data)
n <- length(all.dates)

all.daily.data <- matrix(nrow=n,ncol=S)
for(i in 1:n){
	ind <- which(winddata$years==all.dates$year[i]+1900 & winddata$months==all.dates$mon[i]+1 & winddata$days==all.dates$mday[i])
	if(length(ind)>0){
		all.daily.data[ind,] <- daily.data[ind,]
	}
}
n.daily <- nrow(daily.data)

all.weekly.data <- matrix(nrow=ceiling(n/7),ncol=S)
time.weekly.maxima <- matrix(nrow=0,ncol=S)
for(i in 1:ceiling(n/7)){
	data.week.i <- all.daily.data[((i-1)*7+1):min(n,(i*7)),]
	if(any(is.na(data.week.i))){
		all.weekly.data[i,] <- rep(NA,S)
		time.weekly.maxima <- rbind(time.weekly.maxima,rep(NA,S))
	} else{
		all.weekly.data[i,] <- apply(data.week.i,2,max)
		time.weekly.maxima <- rbind(time.weekly.maxima,sapply(apply(data.week.i,2,which.max),FUN=function(x){a<-((i-1)*7+1):min(n,(i*7)); return(a[x])}))
	}
}
weekly.data <- all.weekly.data[!apply(is.na(all.weekly.data),1,any),]
time.weekly.maxima <- time.weekly.maxima[!apply(is.na(all.weekly.data),1,any),]
n.weekly <- nrow(weekly.data)

distinct.months <- unique(cbind(all.years,all.months))
n.distinct.months <- nrow(distinct.months)
all.monthly.data <- matrix(nrow=n.distinct.months,ncol=S)
time.monthly.maxima <- matrix(nrow=0,ncol=S)
for(i in 1:n.distinct.months){
	data.month.i <- all.daily.data[all.years==distinct.months[i,1] & all.months==distinct.months[i,2],]
	ndays.month.i <- 31*(distinct.months[i,2]%in%c(1,3,5,7,8,10,12)) + 30*(distinct.months[i,2]%in%c(4,6,9,11)) + 28*(distinct.months[i,2]==2)
	if(sum(!is.na(data.month.i))/(ndays.month.i*S)<0.95){
		all.monthly.data[i,] <- rep(NA,S)
		time.monthly.maxima <- rbind(time.monthly.maxima,rep(NA,S))
	} else{
		all.monthly.data[i,] <- apply(data.month.i,2,max,na.rm=TRUE)
		time.monthly.maxima <- rbind(time.monthly.maxima,sapply(apply(data.month.i,2,which.max),FUN=function(x){a<-which(all.years==distinct.months[i,1] & all.months==distinct.months[i,2]); return(a[x])}))
	}
}
monthly.data <- all.monthly.data[!apply(is.na(all.monthly.data),1,any),]
time.monthly.maxima <- time.monthly.maxima[!apply(is.na(all.monthly.data),1,any),]
n.monthly <- nrow(monthly.data)

distinct.years <- unique(all.years)
n.distinct.years <- length(distinct.years)
all.yearly.data <- matrix(nrow=n.distinct.years,ncol=S)
time.yearly.maxima <- matrix(nrow=0,ncol=S)
for(i in 1:n.distinct.years){
	data.year.i <- all.daily.data[all.years==distinct.years[i],]
	ndays.year <- 365
	if(sum(!is.na(data.year.i))/ndays.year<0.95){
		all.yearly.data[i,] <- rep(NA,S)
		time.yearly.maxima <- rbind(time.yearly.maxima,rep(NA,S))
	} else{
		all.yearly.data[i,] <- apply(data.year.i,2,max,na.rm=TRUE)
		time.yearly.maxima <- rbind(time.yearly.maxima,sapply(apply(data.year.i,2,which.max),FUN=function(x){a<-which(all.years==distinct.years[i]); return(a[x])}))
	}
}
yearly.data <- all.yearly.data[!apply(is.na(all.yearly.data),1,any),]
time.yearly.maxima <- time.yearly.maxima[!apply(is.na(all.yearly.data),1,any),]
n.yearly <- nrow(yearly.data)
### we don't have so many years available, so we skip yearly data in the following...

stat <- 5
par(mfrow=c(1,3))
acf(daily.data[,stat])
acf(weekly.data[,stat])
acf(monthly.data[,stat])

######### NON-PARAMETRIC MARGINAL ESTIMATION ##########
daily.data.Frech <- qfrechet(apply(daily.data,2,rank,ties.method="random")/(n.daily+1))
weekly.data.Frech <- qfrechet(apply(weekly.data,2,rank,ties.method="random")/(n.weekly+1))
monthly.data.Frech <- qfrechet(apply(monthly.data,2,rank,ties.method="random")/(n.monthly+1))
### This doesn't work well because of the small sample size of monthly data (and also it's difficult to compare dependence results when margins are not estimated similarly).......

######### NON-PARAMETRIC MARGINAL ESTIMATION FOR DAILY DATA + UPSCALING TO HIGHER BLOCK SIZES ##########
daily.data.Frech <- qfrechet(apply(daily.data,2,rank,ties.method="random")/(n.daily+1))
weekly.data.Frech <- matrix(nrow=n.weekly,ncol=S)
for(j in 1:S){
	weekly.data.Frech[,j] <- daily.data.Frech[time.weekly.maxima[,j]]/7
}
monthly.data.Frech <- matrix(nrow=n.monthly,ncol=S)
for(j in 1:S){
	monthly.data.Frech[,j] <- daily.data.Frech[time.monthly.maxima[,j]]/30
}
### This doesn't work well because of temporal dependence, which is not taken care of when going from daily to weekly and monthly scales.......


theta.weekly <- optim(par=1,fn=dist.cdfs,dat1=daily.data,dat2=weekly.data,method="Brent",lower=0,upper=500)$par
theta.monthly <- optim(par=1,fn=dist.cdfs,dat1=daily.data,dat2=monthly.data,method="Brent",lower=0,upper=500)$par
daily.data.Frech <- daily.data
weekly.data.Frech <- weekly.data
monthly.data.Frech <- monthly.data
for(j in 1:S){
	cdf.daily <- ecdf(daily.data[,j])
	daily.data.Frech[,j] <- qfrechet(cdf.daily(daily.data[,j])*(n.daily/(n.daily+1)))
	weekly.data.Frech[,j] <- qfrechet((cdf.daily(weekly.data[,j])*(n.daily/(n.daily+1)))^theta.weekly)
	monthly.data.Frech[,j] <- qfrechet((cdf.daily(monthly.data[,j])*(n.daily/(n.daily+1)))^theta.monthly)
}
### This seems to work more or less OK....... (but some lack of fit for weekly and monthly data; perhaps we should have a single theta for each station...)


daily.data.Frech <- daily.data
weekly.data.Frech <- weekly.data
monthly.data.Frech <- monthly.data
for(j in 1:S){
	theta.weekly <- optim(par=1,fn=dist.cdfs,dat1=daily.data[,j],dat2=weekly.data[,j],method="Brent",lower=0,upper=500)$par
	theta.monthly <- optim(par=1,fn=dist.cdfs,dat1=daily.data[,j],dat2=monthly.data[,j],method="Brent",lower=0,upper=500)$par
	
	cdf.daily <- ecdf(daily.data[,j])
	daily.data.Frech[,j] <- qfrechet(cdf.daily(daily.data[,j])*(n.daily/(n.daily+1)))
	weekly.data.Frech[,j] <- qfrechet((cdf.daily(weekly.data[,j])*(n.daily/(n.daily+1)))^theta.weekly)
	monthly.data.Frech[,j] <- qfrechet((cdf.daily(monthly.data[,j])*(n.daily/(n.daily+1)))^theta.monthly)
}
### This does not seem to work better than assuming theta constant for all stations, so better to use the latter approach......




GEV.mles <- fitGEVs(daily.data,weekly.data,monthly.data)
daily.data.Frech <- daily.data
weekly.data.Frech <- weekly.data
monthly.data.Frech <- monthly.data

for(j in 1:S){
	mu <- GEV.mles[1,j]; sig <- GEV.mles[2,j]; xi <- GEV.mles[3,j]
	th <- GEV.mles[4,j]
	th.weekly <- th*7
	th.monthly <- th*30
	daily.data.Frech[,j] <- -1/log(pgev(daily.data[,j],loc=mu,scale=sig,shape=xi))
	weekly.data.Frech[,j] <- -1/log(pgev(weekly.data[,j],loc=mu-sig*(1-th.weekly^xi)/xi,scale=sig*th.weekly^xi,shape=xi))
	monthly.data.Frech[,j] <- -1/log(pgev(monthly.data[,j],loc=mu-sig*(1-th.monthly^xi)/xi,scale=sig*th.monthly^xi,shape=xi))
}
### This seems to work reasonably well, but maybe better to have block-specific extreme value indices?......



GEV.mles <- fitGEVs(daily.data,weekly.data,monthly.data)
daily.data.Frech <- daily.data
weekly.data.Frech <- weekly.data
monthly.data.Frech <- monthly.data

for(j in 1:S){
	mu <- GEV.mles[1,j]; sig <- GEV.mles[2,j]; xi <- GEV.mles[3,j]
	th.weekly <- GEV.mles[4,j]; th.monthly <- GEV.mles[5,j]
	daily.data.Frech[,j] <- -1/log(pgev(daily.data[,j],loc=mu,scale=sig,shape=xi))
	weekly.data.Frech[,j] <- -1/log(pgev(weekly.data[,j],loc=mu-sig*(1-th.weekly^xi)/xi,scale=sig*th.weekly^xi,shape=xi))
	monthly.data.Frech[,j] <- -1/log(pgev(monthly.data[,j],loc=mu-sig*(1-th.monthly^xi)/xi,scale=sig*th.monthly^xi,shape=xi))
}
### This seems to work quite well...... This is chosen for the paper.



GEV.mles <- fitGEVs(daily.data,weekly.data,monthly.data)
daily.data.Frech <- daily.data
weekly.data.Frech <- weekly.data
monthly.data.Frech <- monthly.data

for(j in 1:S){
	mu.daily <- GEV.mles[1,j]; sig.daily <- GEV.mles[2,j]; 
	mu.weekly <- GEV.mles[3,j]; sig.weekly <- GEV.mles[4,j]; 
	mu.monthly <- GEV.mles[5,j]; sig.monthly <- GEV.mles[6,j]; 
	xi <- GEV.mles[7,j]
	daily.data.Frech[,j] <- -1/log(pgev(daily.data[,j],loc=mu.daily,scale=sig.daily,shape=xi))
	weekly.data.Frech[,j] <- -1/log(pgev(weekly.data[,j],loc=mu.weekly,scale=sig.weekly,shape=xi))
	monthly.data.Frech[,j] <- -1/log(pgev(monthly.data[,j],loc=mu.monthly,scale=sig.monthly,shape=xi))
}
### This also seems to work quite well......


stat <- 1
par(mfrow=c(1,3))
qqplot(qfrechet((1:n.daily)/(n.daily+1)), daily.data.Frech[,stat],log="xy",main="Daily maxima")
abline(0,1,col="red")
qqplot(qfrechet((1:n.weekly)/(n.weekly+1)), weekly.data.Frech[,stat],log="xy",main="Weekly maxima")
abline(0,1,col="red")
qqplot(qfrechet((1:n.monthly)/(n.monthly+1)), monthly.data.Frech[,stat],log="xy",main="Monthly maxima")
abline(0,1,col="red")
stat <- stat+1

save(daily.data.Frech,file="daily_wind_Frechet.RData")
save(weekly.data.Frech,file="weekly_wind_Frechet.RData")
save(monthly.data.Frech,file="monthly_wind_Frechet.RData")

daily.data.Unif <- exp(-1/daily.data.Frech)
weekly.data.Unif <- exp(-1/weekly.data.Frech)
monthly.data.Unif <- exp(-1/monthly.data.Frech)

save(daily.data.Unif,file="daily_wind_Unif.RData")
save(weekly.data.Unif,file="weekly_wind_Unif.RData")
save(monthly.data.Unif,file="monthly_wind_Unif.RData")

thetaD.daily <- mean(apply(daily.data.Frech,1,max)^(-1))^(-1)
thetaD.weekly <- mean(apply(weekly.data.Frech,1,max)^(-1))^(-1)
thetaD.monthly <- mean(apply(monthly.data.Frech,1,max)^(-1))^(-1)

ICthetaD.daily <- thetaD.daily + c(-2,2)*thetaD.daily/sqrt(n.daily)
ICthetaD.weekly <- thetaD.weekly + c(-2,2)*thetaD.weekly/sqrt(n.weekly)
ICthetaD.monthly <- thetaD.monthly + c(-2,2)*thetaD.monthly/sqrt(n.monthly)



nlevel <- 1000
level.z <- qfrechet(seq(0.3,0.99,length=nlevel))
par(mfrow=c(1,3))
the <- theta.z(daily.data.Frech,level.z=level.z,sd=TRUE,addLS=FALSE,ylim=c(1,10),col="black",colsd="lightgrey") ### CI computed by Delta method
the <- theta.z(weekly.data.Frech,level.z=level.z,sd=TRUE,addLS=FALSE,ylim=c(1,10),col="black",colsd="lightgrey") ### CI computed by Delta method
the <- theta.z(monthly.data.Frech,level.z=level.z,sd=TRUE,addLS=FALSE,ylim=c(1,10),col="black",colsd="lightgrey") ### CI computed by Delta method

pdf(file="~/Dropbox/maxid-sharedfolder/tex/V9/art/extcoefWind_D.pdf",width=7,height=6,onefile=TRUE)
par(mfrow=c(1,1)) ## same but on a single graph... (without yearly maxima)
the <- theta.z(monthly.data.Frech,level.z=level.z,sd=TRUE,addLS=FALSE,ylim=c(1,15),col="red",colsd=rgb(col2rgb("pink")[1]/255,col2rgb("pink")[2]/255,col2rgb("pink")[3]/255,alpha=0.5)) ### CI computed by Delta method
the <- theta.z(weekly.data.Frech,level.z=level.z,sd=TRUE,addLS=FALSE,ylim=c(1,15),add=TRUE,col="blue",colsd=rgb(col2rgb("lightblue")[1]/255,col2rgb("lightblue")[2]/255,col2rgb("lightblue")[3]/255,alpha=0.5)) ### CI computed by Delta method
the <- theta.z(daily.data.Frech,level.z=level.z,sd=TRUE,addLS=FALSE,ylim=c(1,15),add=TRUE,col="black",colsd=rgb(col2rgb("lightgrey")[1]/255,col2rgb("lightgrey")[2]/255,col2rgb("lightgrey")[3]/255,alpha=0.5)) ### CI computed by Delta method
legend(x="topleft",legend=c("Daily","Weekly","Monthly"),col=c("black","blue","red"),lty=1)
dev.off()



dat <- weekly.data.Frech
nlevel <- 1000
level.z <- qfrechet(seq(0.01,0.99,length=nlevel))
#level.z <- sort(unique(apply(dat,1,max)))
#level.z <- level.z[level.z<qfrechet(0.99)]
nlevel <- length(level.z)
replic <- 300
the.boot <- matrix(nrow=nlevel,ncol=replic)
lmfit.boot <- matrix(nrow=2,ncol=replic)
for(i in 1:replic){
	# dat.boot <- dat[sample(1:nrow(dat),nrow(dat),replace=TRUE),] ### simple non-parametric bootstrap
	dat.boot <- c() 
	nboot <- 0
	while(nboot<nrow(dat)){ ### stationary block bootstrap
		len <- rgeom(1,1/4)
		start <- sample(1:nrow(dat),1)
		dat.boot <- rbind(dat.boot,dat[start:min(start+len,nrow(dat)),])
		nboot <- nrow(dat.boot)
	}
	dat.boot <- dat.boot[1:nrow(dat),]
	the <- theta.z(dat.boot,level.z=level.z,PLOT=FALSE)
	the.boot[,i] <- the$theta
	lmfit.boot[,i] <- c(the$intercept,the$slope)
}
the.LB <- apply(the.boot,1,quantile,0.025)
the.UB <- apply(the.boot,1,quantile,0.975)

lmfit.LB <- apply(lmfit.boot,1,quantile,0.025)
lmfit.UB <- apply(lmfit.boot,1,quantile,0.975)

the <- theta.z(dat,level.z=level.z,sd=TRUE,addLS=FALSE,ylim=c(1,10)) ### CI computed by Delta method
lines(level.z,the.LB,lty=2,col="blue") ### Bootstrap
lines(level.z,the.UB,lty=2,col="blue") ### Bootstrap

hist(lmfit.boot[2,],xlab="Slope value",freq=F,main="Histogram of the slope") ### For the histogram to make sense, we need to take a high enough level.z (>0.5, say), otherwise the estimator doesn't work and the slope becomes negative...
abline(v=the$slope,col="blue")














