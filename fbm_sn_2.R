#set.seed(120)
library(MASS)
H_T=0.1
g_T=1.0
T=5
n=100

#H[l_star[3]]
#[1] 0.09452916
#> g[l_star[3]]
#[1] 0.8562708

H_sim = numeric()
g_sim = numeric()



  

delta1 = (T/n)
k1=seq(delta1,T,delta1)
n2=length(k1)


  
mu=rep(0,(n2-1))
delta2=1
k2=seq(1,T,delta2)
n3=length(k2)
k4=seq(2,T,delta2)
k5=seq(1,(T-1),delta2)
b_T=2-2*H_T
k6=k4^b_T
k7=k5^b_T
k8=k6-k7
M_mu=rep(0,(n3-1))

M = 100 #1000

for(k in 1:M )
{
lc2_T=lgamma((3/2)-H_T)-log(2)-log(H_T)-log(b_T)-lgamma(H_T+(1/2))-lgamma(b_T)
c2_T=exp(lc2_T)
M_T=rnorm((n3-1),M_mu,sqrt(c2_T*k8))
lk_H1T=log(2)+log(H_T)+lgamma((3/2)-H_T)+lgamma((1/2)+H_T)
k_H1T=exp(lk_H1T)
a_T=3/2-H_T
ll_HT=log(2)+log(H_T)+lgamma(3-(2*H_T))+lgamma((1/2)+H_T)-lgamma((3/2)-H_T)
l_HT=exp(ll_HT)
Q_T=numeric()
dw_T=numeric()
Z_T=numeric()
M=numeric() 
M_sum=numeric() 
for(i1 in 1:n3) 
{
  k3=seq(.001,k2[i1],delta1)
  n4=length(k3)
  Q1_T=numeric()
  for(j1 in 1:(n4-1))
  {
    Q1_T[j1]=(l_HT/b_T)*(k_H1T)^(-1)*g_T*beta(a_T,a_T)
    
  }
  Q_T[i1]=sum(Q1_T)
  dw_T[i1]=b_T*(k2[i1]^(b_T-1))*(l_HT)^(-1)*delta2
}
for(i2 in 1:(n3-1))
{
  Z_T[i2]=M_T[i2]+Q_T[i2]*dw_T[i2]  
}
N=20000
H=rep(0.15,N)
g=rep(1.15,N)

H=rep(0.15,N)
g=rep(4.670473,N)
g=rep(-1,N)

#H=rep(0.1,N)
#g=rep(0.1,N)
# H=rep(1.920746e-01,N)
# g=rep(2.740235e-01,N)
# H=rep(1.947590e-01,N)
# g=rep(2.824058e-01,N)
# H=rep(1.960164e-01,N)
# g=rep(2.863209e-01,N)
# H=rep(1.966035e-01,N)
# g=rep(2.881316e-01,N)
#Final: H=1.966180e-01 beta=2.881805e-01

L=numeric()
L1=numeric()
L2=numeric()
b=numeric()
T=numeric()
t=seq(2,(N+1))


  
for(l in 1:N)
{
  k_H1=2*H[l]*gamma((3/2)-H[l])*gamma((1/2)+H[l])
  l_H=2*H[l]*gamma(3-(2*H[l]))*gamma((1/2)+H[l])/gamma((3/2)-H[l])
  a=3/2-H[l]
  b=2-2*H[l]
  c2=gamma((3/2)-H[l])/(2*H[l]*b*gamma(H[l]+(1/2))*gamma(b))
  Q=rep(0,n3)
  M1=numeric()
  sd1=numeric()
  w1=numeric()
  for(i1 in 1:n3) 
  {
    k3=seq(.001,k2[i1],delta1)
    n4=length(k3)
    #Z1=rep(0,n4)
    Q1=numeric()
    for(j1 in 1:(n4-1))
    {
      Q1[j1]=(l_H/b)*(k_H1)^(-1)*g[l]*beta(a,a)
      
    }
    Q[i1]=sum(Q1)
    w1[i1]=(k2[i1]^b)*(l_H)^(-1)
  }
  k_6=k4^b
  k_7=k5^b
  k_8=k_6-k_7
  sd1=sqrt(c2*k_8)
  for(i2 in 1:(n3-1))
  {
    M1[i2]=Z_T[i2]-Q[i2]*(w1[i2+1]-w1[i2])  
    L1[i2]=-log((sd1[i2]))-((M1[i2]^2)/(2*sd1[i2]^2))
  }
  L[l]=sum(L1)
  T[l]=1/(log(log(t[l])))
  if(l>100) T[l]=1/log(log(log(t[l])))
  u1=runif(2,0,1)
  for(j in 1:2)
  {
    if(u1[j]<(1/2)){
      b[j]=1
    } else {
      b[j]=-1
    }
  }
  epsilon=rnorm(1,0,1)
  #a1=(.001)^(1/2)
  #a2=(.001)^(1/2)
  #a1 = 0.0003
  #a2 = 0.0004
  a1 = 0.001
  a2 = 0.0 #0.001
  
  H[l+1]=H[l]+b[1]*a1*abs(epsilon)
  g[l+1]=g[l]+b[2]*a2*abs(epsilon)
  if (H[l+1]>0 && H[l+1]<1)
  {
    H[l+1]=H[l+1]
    g[l+1]=g[l+1]
    k_H1=2*H[l+1]*gamma((3/2)-H[l+1])*gamma((1/2)+H[l+1])
    l_H=2*H[l+1]*gamma(3-(2*H[l+1]))*gamma((1/2)+H[l+1])/gamma((3/2)-H[l+1])
    a=3/2-H[l+1]
    b=2-2*H[l+1]
    c2=gamma((3/2)-H[l+1])/(2*H[l+1]*b*gamma(H[l+1]+(1/2))*gamma(b))
    Q2=rep(0,n3)
    M2=numeric()
    sd2=numeric()
    w2=numeric()
    for(i1 in 1:n3) 
    {
      k3=seq(.001,k2[i1],delta1)
      n4=length(k3)
      #Z1=rep(0,n4)
      Q3=numeric()
      for(j1 in 1:(n4-1))
      {
        Q3[j1]=(l_H/b)*(k_H1)^(-1)*g[l+1]*beta(a,a)
        
      }
      Q2[i1]=sum(Q3)
      w2[i1]=(k2[i1]^b)*(l_H)^(-1)
    }
    k_6=k4^b
    k_7=k5^b
    k_8=k_6-k_7
    sd2=sqrt(c2*k_8)
    for(i2 in 1:(n3-1))
    {
      M2[i2]=Z_T[i2]-Q2[i2]*(w2[i2+1]-w2[i2])  
      L2[i2]=-log((sd2[i2]))-((M2[i2]^2)/(2*sd2[i2]^2))
    }
    L[l+1]=sum(L2)
    p=min(0,((L[l+1]-L[l])/T[l]))
    u2=runif(1,0,1)
    if(log(u2)>=p){
      H[l+1]=H[l]
      g[l+1]=g[l]
    } else {
      H[l+1]=H[l+1]
      g[l+1]=g[l+1]
    }
    
  }else{
    H[l+1]=H[l]
    g[l+1]=g[l]
  }
  #x=c(l,H[l],g[l])
  #print(x)
    
}

max_L=max(L)
l_star=which(L==max_L)

H_sim[k] = H[l_star[1]]
g_sim[k] = g[l_star[1]]
x2=c(k,H_sim[k],g_sim[k])
print(x2)
}

xboot=cbind(H_sim,g_sim)
write(t(xboot),file="bootstrap_mle2",ncol=2)

