setwd("~/Desktop/Textbook_subsymptotic_extremes/")




#############################################################################################################################
#############################################################################################################################
### ------------------------------------------------------ PNW east ----------------------------------------------------- ###
#############################################################################################################################
#############################################################################################################################
load("./PNW_east.RData")
load("./PNW_10Day_copula.RData")
load("./PNW_10Day_maxima.RData")
stationDF_PNW <- stationDF_PNW[,-1]
library(sp)
PNW_east_poly <- data.frame(x = PNW_east$x, y = PNW_east$y)
PNW_east_poly <- rbind(PNW_east_poly, PNW_east_poly[1,])
which.in.polygon.east <- which(point.in.polygon(stationDF_PNW[,1], stationDF_PNW[,2], PNW_east_poly$x, PNW_east_poly$y)==1)
stations_east <- stationDF_PNW[which.in.polygon.east, ]
U <- U[which.in.polygon.east, ]
PNW_east_10Day_maxima <- PNW_JJA_10day[which.in.polygon.east, ]


library(sp)
library(gstat) 
Range <- rep(NA, ncol(PNW_east_10Day_maxima))
for(time in 1:ncol(PNW_east_10Day_maxima)){
  dat_variog <- cbind(stations_east, X= PNW_east_10Day_maxima[,time])
  coordinates(dat_variog) = ~longitude + latitude
  varg <- variogram(X ~ 1, data = dat_variog, cressie=TRUE)
  varg.fit.matern = fit.variogram(varg, vgm(psill = 10, "Mat", range = 2, nugget = 0, kappa=1.5), fit.kappa = FALSE)
  Range[time] <- varg.fit.matern$range[2]
}
mean(Range[Range<5]) #0.4886055

library(invgamma)
source("./utils.R")
delta_m = 0.389
tau_sq_m = 10
corr_m = fields::Matern(d=0.3, range  = 0.342, nu=1.5)
res_true<-chi_u(N=1e6,delta=delta_m, tau=tau_sq_m, corr=corr_m) 


range_m = 1.05
nu_m=1.88
res_true_IM<-chi_u_invertedMaxStab(N = 1e6, d=0.3, range=range_m, nu=nu_m)


plot(dat$x, dat$truth, type='l', ylim=c(0,1))
lines(dat$x, dat$truth_upper)
lines(dat$x, dat$truth_lower)
lines(res_true$U, res_true$chi, col='red')
lines(res_true_IM$U, res_true_IM$chi, col='blue')

plot(dat_bar$x, dat_bar$truth, type='l', ylim=c(0,1))
lines(dat_bar$x, dat_bar$truth_upper)
lines(dat_bar$x, dat_bar$truth_lower)
lines(res_true$U, res_true$chibar, col='red')
lines(res_true_IM$U, res_true_IM$chibar, col='blue')



save(dat, dat_bar, res_true, res_true_IM, file='h=0.3.RData')
chi_data <- data.frame(x=res_true$U, chi=res_true$chi, chibar = res_true$chibar, model = "Huser-Wadsworth")
chi_data <- rbind(chi_data,
            data.frame(x=res_true_IM$U, chi=res_true_IM$chi, chibar = res_true_IM$chibar, model = "Inverted max-stable"))


plt <- ggplot(dat,aes(x=x,y=truth)) +
  geom_line(color="#0047fa",linewidth=0.3) +
  geom_line(aes(x=x, y=truth_upper),linewidth=0.2,color="#0047fa") +
  geom_line(aes(x=x, y=truth_lower),linewidth=0.2,color="#0047fa") +
  geom_ribbon(data=dat,aes(ymin=truth_lower,ymax=truth_upper),alpha=0.2,fill="#0266d9") +
  geom_line(data=chi_data, aes(x=x, y=chi, color=model),linewidth=0.8) +
  ylab(expression(chi(u))) + xlab("Quantile") +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = 'none',
        panel.background = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA)) + 
  scale_x_continuous(expand = c(0, 0), limits=c(0.947,1)) + ggtitle("PNW_east") +
  scale_y_continuous(expand = c(0, 0), limits = c(0,1)) + 
  force_panelsizes(rows = unit(3.05, "in"),
                   cols = unit(3.05, "in"))

plt
ggsave("./chi_fit_PNWeast.pdf", width=3.7, height=3.7, unit = 'in')



chi_data$chibar[chi_data$chibar < 0] <- NA
plt <- ggplot(dat_bar,aes(x=x,y=truth)) +
  geom_line(color="#0047fa",linewidth=0.3) +
  geom_line(aes(x=x, y=truth_upper),linewidth=0.2,color="#0047fa") +
  geom_line(aes(x=x, y=truth_lower),linewidth=0.2,color="#0047fa") +
  geom_ribbon(data=dat_bar,aes(ymin=truth_lower,ymax=truth_upper),alpha=0.2,fill="#0266d9") +
  geom_line(data=chi_data, aes(x=x, y=chibar, color=model),linewidth=0.8) +
  ylab(expression(bar(chi)(u))) + xlab("Quantile") +
  theme(plot.title = element_text(hjust = 0.5),
        panel.background = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA)) + 
  scale_x_continuous(expand = c(0, 0), limits=c(0.947,1)) + ggtitle("PNW_east") +
  scale_y_continuous(expand = c(0, 0), limits = c(0,1)) + 
  force_panelsizes(rows = unit(3.05, "in"),
                   cols = unit(3.05, "in"))

plt
ggsave("./chi_bar_fit_PNWeast.pdf", width=5.3, height=3.7, unit = 'in')


## -------------------------------------------------------
##                        h=0.3
## -------------------------------------------------------
Dist <- fields::rdist(stations_east[,1:2])
h <- 0.3; tol <- 0.05
pairs <- which(Dist < h+tol & Dist > h-tol, arr.ind = TRUE)
pairs <- pairs[pairs[,1] < pairs[,2], ]
nrow(pairs)

plot(stationDF_PNW[,1:2])
points(stations_east[,1:2], pch=20)
for(iter in 1:nrow(pairs)){
  points(stations_east[pairs[iter,],1:2], type='l')
}

U_pairs <- matrix(NA,nrow = ncol(U)*round(nrow(pairs)),ncol = 2)
ind <- 1

for(time in 1:ncol(U)){
  for(npair in 1:nrow(pairs)){
    U_pairs[ind, ] <- c(U[pairs[npair, 1], time], U[pairs[npair, 2], time])
    ind<- ind + 1
  }
}

Min_sim <- apply(U_pairs, 1, min)
all_sim <- as.vector(U_pairs)
u_vec=c(seq(0.95,0.98,0.01),seq(0.9801,0.9997,0.0001))

## -- chi(u) 
EmpIntv_sim <- matrix(NA, nrow = length(u_vec), ncol=3)

for(i in 1:length(u_vec)){
  p_tmp_sim <- mean(Min_sim>u_vec[i])
  p_tmp1_sim <- mean(U_pairs[,1]>u_vec[i])
  if(p_tmp_sim==0|p_tmp1_sim==0){
    EmpIntv_sim[i,]<-c(-2,2,0)
  } else{
    var_sim <- 2*p_tmp_sim^2/p_tmp1_sim^2*{(1-p_tmp_sim)/p_tmp_sim - (1-p_tmp1_sim)/p_tmp1_sim}/length(Min_sim) 
    EmpIntv_sim[i,]<-c(exp(log(p_tmp_sim/p_tmp1_sim)) - qnorm(0.975)*sqrt(var_sim),
                       exp(log(p_tmp_sim/p_tmp1_sim)) - qnorm(0.025)*sqrt(var_sim), p_tmp_sim/p_tmp1_sim)
  }
}
dat <- data.frame(x=u_vec, truth=EmpIntv_sim[,3], truth_upper=EmpIntv_sim[,2],truth_lower=EmpIntv_sim[,1], region = "PNW_east")
Dat <- dat

dat[dat>1] <- 1
dat[dat<0] <- 0



## -- chi_bar(u)
u_vec=c(seq(0.95,0.98,0.01),seq(0.9801,0.9997,0.0001))
EmpIntv_sim_bar <- matrix(NA, nrow = length(u_vec), ncol=3)

for(i in 1:length(u_vec)){
  p_tmp_sim <- mean(Min_sim>u_vec[i]) #joint
  p_tmp1_sim <- mean(U_pairs[,1]>u_vec[i]) #marginal
  if(p_tmp_sim==0|p_tmp1_sim==0){
    EmpIntv_sim_bar[i,]<-c(-2,2,0)
  } else{
    var_sim <- 8*(log(p_tmp1_sim)/log(p_tmp_sim))^2*{(1-p_tmp_sim)/(p_tmp_sim*log(p_tmp_sim)^2)+(1-p_tmp1_sim)/(p_tmp1_sim*log(p_tmp1_sim)^2) -
        2*(1-p_tmp1_sim)/(p_tmp1_sim*log(p_tmp_sim)*log(p_tmp1_sim))}/length(Min_sim) 
    EmpIntv_sim_bar[i,]<-c(2*log(p_tmp1_sim)/log(p_tmp_sim)-1 - qnorm(0.975)*sqrt(var_sim),
                           2*log(p_tmp1_sim)/log(p_tmp_sim)-1 - qnorm(0.025)*sqrt(var_sim), 2*log(p_tmp1_sim)/log(p_tmp_sim)-1)
  }
}
dat_bar <- data.frame(x=u_vec, truth=EmpIntv_sim_bar[,3], truth_upper=EmpIntv_sim_bar[,2],truth_lower=EmpIntv_sim_bar[,1], region = "PNW_east")
Dat_bar <- dat_bar



## -------------------------------------------------------
##                        h=1
## -------------------------------------------------------
Dist <- fields::rdist(stations_east[,1:2])
h <- 0.7; tol <- 0.02
pairs <- which(Dist < h+tol & Dist > h-tol, arr.ind = TRUE)
pairs <- pairs[pairs[,1] < pairs[,2], ]
nrow(pairs)

plot(stationDF_PNW[,1:2])
points(stations_east[,1:2], pch=20)
for(iter in 1:nrow(pairs)){
  points(stations_east[pairs[iter,],1:2], type='l')
}

U_pairs <- matrix(NA,nrow = ncol(U)*round(nrow(pairs)),ncol = 2)
ind <- 1

for(time in 1:ncol(U)){
  for(npair in 1:nrow(pairs)){
    U_pairs[ind, ] <- c(U[pairs[npair, 1], time], U[pairs[npair, 2], time])
    ind<- ind + 1
  }
}

Min_sim <- apply(U_pairs, 1, min)
all_sim <- as.vector(U_pairs)
u_vec=c(seq(0.95,0.98,0.01),seq(0.9801,0.9997,0.0001))

## -- chi(u) 
EmpIntv_sim <- matrix(NA, nrow = length(u_vec), ncol=3)

for(i in 1:length(u_vec)){
  p_tmp_sim <- mean(Min_sim>u_vec[i])
  p_tmp1_sim <- mean(U_pairs[,1]>u_vec[i])
  if(p_tmp_sim==0|p_tmp1_sim==0){
    EmpIntv_sim[i,]<-c(-2,2,0)
  } else{
    var_sim <- 2*p_tmp_sim^2/p_tmp1_sim^2*{(1-p_tmp_sim)/p_tmp_sim - (1-p_tmp1_sim)/p_tmp1_sim}/length(Min_sim) 
    EmpIntv_sim[i,]<-c(exp(log(p_tmp_sim/p_tmp1_sim)) - qnorm(0.975)*sqrt(var_sim),
                       exp(log(p_tmp_sim/p_tmp1_sim)) - qnorm(0.025)*sqrt(var_sim), p_tmp_sim/p_tmp1_sim)
  }
}
dat <- data.frame(x=u_vec, truth=EmpIntv_sim[,3], truth_upper=EmpIntv_sim[,2],truth_lower=EmpIntv_sim[,1], region = "PNW_east")
Dat <- dat

dat[dat>1] <- 1
dat[dat<0] <- 0



## -- chi_bar(u)
u_vec=c(seq(0.95,0.98,0.01),seq(0.9801,0.9997,0.0001))
EmpIntv_sim_bar <- matrix(NA, nrow = length(u_vec), ncol=3)

for(i in 1:length(u_vec)){
  p_tmp_sim <- mean(Min_sim>u_vec[i]) #joint
  p_tmp1_sim <- mean(U_pairs[,1]>u_vec[i]) #marginal
  if(p_tmp_sim==0|p_tmp1_sim==0){
    EmpIntv_sim_bar[i,]<-c(-2,2,0)
  } else{
    var_sim <- 8*(log(p_tmp1_sim)/log(p_tmp_sim))^2*{(1-p_tmp_sim)/(p_tmp_sim*log(p_tmp_sim)^2)+(1-p_tmp1_sim)/(p_tmp1_sim*log(p_tmp1_sim)^2) -
        2*(1-p_tmp1_sim)/(p_tmp1_sim*log(p_tmp_sim)*log(p_tmp1_sim))}/length(Min_sim) 
    EmpIntv_sim_bar[i,]<-c(2*log(p_tmp1_sim)/log(p_tmp_sim)-1 - qnorm(0.975)*sqrt(var_sim),
                           2*log(p_tmp1_sim)/log(p_tmp_sim)-1 - qnorm(0.025)*sqrt(var_sim), 2*log(p_tmp1_sim)/log(p_tmp_sim)-1)
  }
}
dat_bar <- data.frame(x=u_vec, truth=EmpIntv_sim_bar[,3], truth_upper=EmpIntv_sim_bar[,2],truth_lower=EmpIntv_sim_bar[,1], region = "PNW_east")
Dat_bar <- dat_bar


