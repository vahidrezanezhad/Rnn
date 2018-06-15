import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib as mpl
mpl.rcParams.update({'font.size':20})
mpl.rcParams.update({'figure.autolayout':True})

def tanh(arr_int):
    return 2.00/(1+np.exp(-2*arr_int))-1
def likelihood(dtrue,d_est):
    return np.dot( dtrue-d_est, dtrue-d_est)
def sigmoid(arr_int):
    return 1.00/(1.00+np.exp(-arr_int))
#npoint=100
#t=np.linspace(0,100,npoint)
#t_ext=np.linspace(0,140,140) 
#x=np.sin(10*np.pi*t)
#x_ext=np.sin(10*np.pi*t_ext)

#tt=range(140)
x_ext=np.loadtxt('stock_price.txt')
x_ext=x_ext[500:]
ntot=len(x_ext)
ntest=200
ntrain=ntot-ntest
npoint=ntrain
tt=range(ntot)
x_ext=x_ext[:ntot]
#x_ext=2*(x_ext-np.min(x_ext))/np.float(np.max(x_ext)-np.min(x_ext))-1

x_ext_mean=np.mean(x_ext)
x_ext_std=np.std(x_ext)
x_ext=(x_ext-np.mean(x_ext))/np.std(x_ext)


x=x_ext[:ntrain]
print x
#plt.plot(t,x,'o')
#plt.plot(x_ext,'-o',label='Estimated data',markersize=8,linewidth=5)
#plt.show()
iteration=250000
sigma=1e-3#5e-6#1e-6
num_of_in=40#17#20#16#12
sigma_par=0.03#0.02
Nt=ntrain-num_of_in-1
num_out=1
num_of_hid=17#9#9#9#8#8
num_of_param= num_of_hid*num_of_in+num_of_hid+num_of_hid*num_of_hid+num_of_hid+1
dtrue=x[num_of_in:npoint-1]

#print len(t),len(dtrue),'len'

par_ih=[]
par_hh=[]
par_ho=[]
bias_h=[]
bias_o=[]
for un in xrange(num_of_hid*num_of_in):
    par_ih.append([])
for un2 in xrange(num_of_hid*num_of_hid):
    par_hh.append([])
for un3 in xrange(num_of_hid):
    par_ho.append([])
    bias_h.append([])


y_predict_par=[]
for pp in xrange(ntest):
    y_predict_par.append([])
  
cost=[]
sm=[]

num_of_par=num_of_hid*num_of_in+num_of_hid*num_of_hid+num_of_hid+num_of_hid+1

#initial point of parameters
for j in xrange(num_of_par):
    sm.append(0.001)
for jjjj in xrange(ntest):
    sm.append(1)
# first arg
input_mult=np.array(sm[:num_of_hid*num_of_in]).reshape(num_of_hid,num_of_in)
    
rnn_mul=np.array(sm[num_of_hid*num_of_in:num_of_hid*num_of_in+num_of_hid*num_of_hid ] ).reshape(num_of_hid,num_of_hid)

bias_hid=np.array( sm[num_of_hid*num_of_in+num_of_hid**2:num_of_hid*num_of_in+num_of_hid**2+num_of_hid ])

wo=np.array( sm[num_of_hid*num_of_in+num_of_hid**2+num_of_hid:num_of_hid*num_of_in+num_of_hid**2+num_of_hid+num_of_hid ])
bo=sm[num_of_par-1]

res_hat=np.zeros(len(dtrue))
htold=np.zeros(num_of_hid)
for jj in xrange(Nt):
    xx=x[jj:jj+num_of_in]
    htnew=np.dot( input_mult,xx)+np.dot(rnn_mul,htold)+bias_hid
    ##htnew=tanh(htnew)
    ###htnew=sigmoid(htnew)

    ot=np.dot( htnew, wo) +bo

    #ot=tanh(ot)
    #ot=sigmoid(ot) 

    res_hat[jj]=ot
    htold[:]=htnew[:]



num_est=ntest
htold_predict=np.zeros(num_of_hid)
x_start=x[len(x)-num_of_in:]
x_start=list(x_start)
x_be_estimated=[]
y_predict=np.zeros(num_est)
for vz in xrange(num_est):
    htnew_predict=np.dot( input_mult,np.array(x_start))+np.dot(rnn_mul,htold_predict)+bias_hid
    ot_predict=np.dot( htnew_predict, wo) +bo
    y_predict[vz]=ot_predict
    x_start.pop(0)
    x_start.append(y_predict[vz])
    htold_predict[:]=htnew_predict[:]



arg=likelihood( dtrue, res_hat)
arg_predict=likelihood( np.array( sm[num_of_par:]) ,y_predict   )

for i in xrange(iteration):
    nrr=random.randint(0,num_of_par+ntest-1)
    print i
    smnew=sm[:]
    ##print np.array(smnew)
    if nrr<=num_of_par-1:
        smnew[nrr]=random.gauss(sm[nrr],sigma_par)
    elif nrr>num_of_par-1:
        smnew[nrr]=random.gauss(sm[nrr],0.4)
    #smnew[nrr]=random.gauss(sm[nrr],sigma_par)

    input_mult=np.array(smnew[:num_of_hid*num_of_in]).reshape(num_of_hid,num_of_in)
    rnn_mul=np.array(smnew[num_of_hid*num_of_in:num_of_hid*num_of_in+num_of_hid*num_of_hid ] ).reshape(num_of_hid,num_of_hid)

    bias_hid=np.array( smnew[num_of_hid*num_of_in+num_of_hid**2:num_of_hid*num_of_in+num_of_hid**2+num_of_hid ])

    wo=np.array( smnew[num_of_hid*num_of_in+num_of_hid**2+num_of_hid:num_of_hid*num_of_in+num_of_hid**2+num_of_hid+num_of_hid ])

    bo=smnew[num_of_par-1]

    res_hat=np.zeros(len(dtrue))
    htold=np.zeros(num_of_hid)
    for j in xrange(Nt):
        xx=x[j:j+num_of_in]
        htnew=np.dot( input_mult,xx)+np.dot(rnn_mul,htold)+bias_hid
        ##htnew=tanh(htnew)
        ###htnew=sigmoid(htnew)
       
        ot=np.dot( htnew, wo) +bo

        #ot=tanh(ot)
        ###ot=sigmoid(ot) 

        res_hat[j]=ot
        htold[:]=htnew[:]

    argnew=likelihood( dtrue, res_hat)
    """
    if nrr<=num_of_par-1:
       min_of_dis=min(1,np.exp(-(argnew-arg)/(sigma)))
    elif nrr>num_of_par-1:
       min_of_dis=min(1,np.exp(-(argnew-arg)/(sigma)))
    """




    num_est=ntest
    htold_predict=np.zeros(num_of_hid)
    x_start=x[len(x)-num_of_in:]
    x_start=list(x_start)
    x_be_estimated=[]
    y_predict=np.zeros(num_est)
    for vz in xrange(num_est):
        ####print x_start
        htnew_predict=np.dot( input_mult,np.array(x_start))+np.dot(rnn_mul,htold_predict)+bias_hid
        ########htnew=tanh(htnew)
        ########htnew=sigmoid(htnew)
 
        ot_predict=np.dot( htnew_predict, wo) +bo

        #######ot=tanh(ot)
        #########ot=sigmoid(ot)    
        y_predict[vz]=ot_predict
        #x_be_estimated.append( x_start[:])
        x_start.pop(0)
        x_start.append(y_predict[vz])
        ###x_start[:]=x_be_estimated[:]
        htold_predict[:]=htnew_predict[:]
    

    argnew_predict=likelihood( np.array( smnew[num_of_par:]) , y_predict)
    
    if nrr<=num_of_par-1:
        min_of_dis=min(1,np.exp(-(argnew-arg)/(sigma)))
    elif nrr>num_of_par-1:
        min_of_dis=min(1,np.exp(-(argnew-arg)/(sigma))*np.exp(-(argnew_predict-arg_predict)/(1))  )







    dran=random.random()
    print argnew,arg,argnew-arg,'argdiff'
    if min_of_dis>=dran:
        print i,'i'
        sm[nrr]=smnew[nrr]
        arg=argnew
        arg_predict=argnew_predict
        for uh in xrange(num_of_in*num_of_hid):
            par_ih[uh].append(sm[uh])

        for uh2 in xrange(num_of_hid*num_of_hid):
            par_hh[uh2].append(sm[uh2+num_of_in*num_of_hid])
        
        for uh3 in xrange(num_of_hid):
            #print len(sm),uh3+num_of_in*num_of_hid+num_of_hid*num_of_hid+num_of_hid

            bias_h[uh3].append(sm[uh3+num_of_in*num_of_hid+num_of_hid*num_of_hid])     
            par_ho[uh3].append(sm[uh3+num_of_in*num_of_hid+num_of_hid*num_of_hid+num_of_hid])
        
        for uh4 in xrange(ntest):
            y_predict_par[ uh4].append( sm[uh4+num_of_par] )
        #print num_of_in*num_of_hid+num_of_hid*num_of_hid+num_of_hid+1,'diz'
        bias_o.append(sm[num_of_in*num_of_hid+num_of_hid*num_of_hid+num_of_hid+1 ] )
        cost.append(arg)



par_ih_mean=np.zeros( num_of_in*num_of_hid)
par_hh_mean=np.zeros( num_of_hid*num_of_hid)

par_ho_mean=np.zeros(num_of_hid)
bias_h_mean=np.zeros( num_of_hid)

y_predict_par_var=np.zeros(ntest)
y_predict_par_mean=np.zeros(ntest)
for umm in xrange(num_of_in*num_of_hid):
    par_ih_arr=np.array(par_ih[umm])
    par_ih_mean[umm]=np.mean(par_ih_arr[12000:])
                           
par_ih_mean=par_ih_mean.reshape(num_of_hid,num_of_in)

for umm2 in xrange(num_of_hid*num_of_hid):
    par_hh_arr=np.array(par_hh[umm2])
    par_hh_mean[umm2]=np.mean(par_hh_arr[12000:])

par_hh_mean=par_hh_mean.reshape(num_of_hid,num_of_hid)

for umm3 in xrange(num_of_hid):
    par_ho_arr=np.array(par_ho[umm3])
    bias_h_arr=np.array(bias_h[umm3])
    par_ho_mean[umm3]=np.mean(par_ho_arr[12000:])
    bias_h_mean[umm3]=np.mean(bias_h_arr[12000:])

for umm4 in xrange(ntest):
    y_predict_arr=np.array( y_predict_par[umm4])
    y_predict_par_mean[umm4]=np.mean(y_predict_arr[ 12000:] )
    y_predict_par_var[umm4]=np.std(y_predict_arr[ 12000:] )
bias_o_arr=np.array(bias_o)
bias_o_mean=np.mean( bias_o_arr[12000:])

###num_est=40
###x_start=x[:num_of_in]
###x_start=list(x_start)
####x_be_estimated=[]
###print x
htold=np.zeros(num_of_hid)
y_fit=np.zeros(Nt)
for vz in xrange(Nt):
    ###print x_start
    x_start=x[vz:vz+num_of_in]
    #x_start=x[:num_of_in]
    htnew=np.dot( par_ih_mean,np.array(x_start))+np.dot(par_hh_mean,htold)+bias_h_mean
    #####htnew=tanh(htnew)
    #####htnew=sigmoid(htnew)

    ot=np.dot( htnew, par_ho_mean) +bias_o_mean

    ####ot=tanh(ot)
    ######ot=sigmoid(ot)    
    y_fit[vz]=ot
    ####x_be_estimated.append( x_start[:])
    ###x_start.pop(0)
    ###x_start.append(y_est[vz])
    ####x_start[:]=x_be_estimated[:]
    htold[:]=htnew[:]

###plt.plot(y_predict_par_var ,'o')
###plt.show()

###plt.plot(y_predict_par[0][:],'o')
###plt.show()

#print len(tt[140-40:]),y_predict_par_mean
##plt.figure(0)
##ax=plt.gca()

###plt.plot(tt[ntrain:],y_predict_par_mean*x_ext_std+x_ext_mean,'-o',label='Estimated data',markersize=8,linewidth=5)
####ax.errorbar(tt[140-40:],y_predict_par_mean*x_ext_std+x_ext_mean,yerr=y_predict_par_var*x_ext_std,fmt='-o',label='Estimated data',markersize=8,linewidth=5)
####plt.plot(tt[140-40:],y_est*x_ext_std+x_ext_mean,'-o',label='Estimated data',markersize=8,linewidth=5)
###plt.plot(tt[ntrain:],x_ext[ntrain:]*x_ext_std+x_ext_mean,'-*',color='red',label='Test set',markersize=8,linewidth=5)
###plt.plot(tt[:ntrain],x_ext[:ntrain]*x_ext_std+x_ext_mean,'-*',color='green',label='Training set',markersize=8,linewidth=5)
###plt.xlabel('Time [day]')
###plt.ylabel('Number of passengers')
###plt.legend(loc='best')

###plt.show()


##plt.figure(1)
##ax=plt.gca()

##plt.plot(tt[num_of_in:Nt+num_of_in],y_fit*x_ext_std+x_ext_mean,'-o',label='Estimated data',markersize=8,linewidth=5)

##plt.plot(tt[:ntrain],x_ext[:ntrain]*x_ext_std+x_ext_mean,'-*',color='green',label='Training set',markersize=8,linewidth=5)
##plt.xlabel('Time [day]')
##plt.ylabel('Number of passengers')
##plt.legend(loc='best')

##plt.show()

np.savetxt('y1.txt',y_predict_par_mean*x_ext_std+x_ext_mean)
np.savetxt('x1.txt',tt[ntrain:])

np.savetxt('y1p.txt',x_ext[ntrain:]*x_ext_std+x_ext_mean)
np.savetxt('x1p.txt',tt[ntrain:])


np.savetxt('y1t.txt',x_ext[:ntrain]*x_ext_std+x_ext_mean)
np.savetxt('x1t.txt',tt[:ntrain])

np.savetxt('y2.txt',y_fit*x_ext_std+x_ext_mean)
np.savetxt('x2.txt',tt[num_of_in:Nt+num_of_in])





#plt.plot(np.log(cost),'o')
#plt.show()
"""
plt.plot(bias_h[0][:],'o')
plt.show()

plt.plot(par_hh[0][:],'o')
plt.show()

plt.plot(par_ih[0][:],'o')
plt.show()

plt.plot(par_ih[4][:],'o')
plt.show()

plt.plot(bias_h[0][:],'o')
plt.show()

plt.plot(bias_o[:],'o')
plt.show()

plt.plot(par_ho[0][:],'o')
plt.show()
"""
