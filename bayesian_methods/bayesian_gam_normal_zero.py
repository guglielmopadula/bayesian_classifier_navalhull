import os
import multiprocessing
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

from jax import vmap
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SA
import dill
import jax
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import pairwise_distances
import scipy
import argparse
import time

import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from jax.lib import xla_bridge
import os
import multiprocessing


print(xla_bridge.get_backend().platform)
print(jax.device_count())
print(jax.local_device_count())
numpyro.set_host_device_count(4)
NUM_BAYES_SAMPLES=4000
NUM_WARMUP=24000
NUM_CHAINS=4

def invert_permutation(p):
    p = np.asanyarray(p) # in case p is a tuple, etc.
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s

def run_inference(model, rng_key, X, Y):
    start = time.time()
    kernel = NUTS(model,step_size=0.0001)
    mcmc = MCMC(
        kernel,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_BAYES_SAMPLES,
        num_chains=NUM_CHAINS,
        progress_bar=True,
    )
    mcmc.run(rng_key, X, Y)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples(group_by_chain=True)

'''
# helper function for prediction
def predict(model, rng_key, samples, X):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(X=X, Y=None)
    return model_trace["Y"]["value"]
'''


# create artificial regression dataset


def eta_constructor(m,d):
    if d//2==0:
        def simple_eta(x):
            return (-1)**(m+1+d//2)/(2**(2*m-1)*np.pi**(d//2)*np.math.factorial(m-1)*np.math.factorial(m+d//2-1))*x**(2*m-d)*np.log(x)
    
    if d//2!=0:
        def simple_eta(x):
            return scipy.special.gamma(d/2-m)/(2**(2*m)*np.pi**(d//2)*np.math.factorial(m-1))*x**(2*m-d)
        
    return simple_eta   

positive_data=np.load("positive_data_red.npy")
negative_data=np.load("negative_data_red.npy")
NUM_FEATURES=positive_data.shape[1]
m=4 #must be 2*m>d+1
simple_eta=eta_constructor(m,NUM_FEATURES)


true=np.load("positive_data_red.npy")
false=np.load("negative_data_red.npy")
train_true = true[:300]     
train_false = false[:300]
test_true=true[300:]
test_false=false[300:]
train_perm=np.random.permutation(600)
test_perm=np.random.permutation(600)
train=np.concatenate((train_true,train_false))
test=np.concatenate((test_true,test_false))
target=np.concatenate((np.ones(300,dtype=np.int32),np.zeros(300,dtype=np.int32)))
target_train=target
train=train[train_perm]
target_train=target[train_perm]
test=test[test_perm]
target_test=target[test_perm]
NUM_DATA=train.shape[0]




def partition0(max_range, S):
    K = len(max_range)
    return np.array([i for i in itertools.product(*(range(i+1) for i in max_range)) if sum(i)<=S])            
part=partition0([m,m,m,m,m],m-1)
part=part.T
part=part.reshape(1,part.shape[0],part.shape[1])
part=part.repeat(NUM_DATA,axis=0)
M=part.shape[2]
k=M+1



Distance_train=pairwise_distances(train)
E_train=simple_eta(Distance_train)
tmp=train.reshape(NUM_DATA,NUM_FEATURES,-1).repeat(M,axis=2)
T=np.prod(tmp**part,axis=1)
w,v=np.linalg.eig(E_train)
u=v.T
uk=u[:,:k]
dk=np.diag(w[:k])
print(dk)
Zk=scipy.linalg.null_space(T.T@uk)
delta_coeff=uk@dk@Zk
alpha_coeff=T
X_train=np.concatenate((delta_coeff,alpha_coeff),axis=1)



Distance_test=pairwise_distances(test)
E_test=simple_eta(Distance_test)
tmp=test.reshape(NUM_DATA,NUM_FEATURES,-1).repeat(M,axis=2)
T=np.prod(tmp**part,axis=1)
w,v=np.linalg.eig(E_test)
u=v.T
uk=u[:,:k]
dk=np.diag(w[:k])
print(dk)
Zk=scipy.linalg.null_space(T.T@uk)
delta_coeff=uk@dk@Zk
alpha_coeff=T
X_test=np.concatenate((delta_coeff,alpha_coeff),axis=1)

scaler = StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)



classic_coeffs=np.load("gam_coef.npy")
alpha_mean=classic_coeffs[-1]
beta_mean=classic_coeffs[:-1]
alpha_mean=jnp.array(alpha_mean.reshape(1,1))
beta_mean=jnp.array(beta_mean.reshape(-1,1))


def model(X,Y):
    N,D_X=X.shape
    alpha = numpyro.sample("alpha", dist.Normal(0, 5*np.ones((1, 1))))
    beta = numpyro.sample("beta", dist.Normal(0,5*jnp.ones((D_X, 1))))
    logit=jnp.matmul(X,beta)+alpha
    y=numpyro.sample("Y",dist.BernoulliLogits(logit).to_event(1),obs=Y)
    return y
rng_key, rng_key_predict = random.split(random.PRNGKey(0))

shape=X_train[0].shape[0]
posterior_samples=run_inference(model,rng_key,X_train,target_train)
alpha=posterior_samples["alpha"].reshape(NUM_CHAINS,NUM_BAYES_SAMPLES,-1)
beta=posterior_samples["beta"].reshape(NUM_CHAINS,NUM_BAYES_SAMPLES,-1)


predictive = Predictive(model, posterior_samples, return_sites=["Y"])
y_fitted_dist=predictive(random.PRNGKey(0), X_train,None)["Y"]
y_fitted_dist=y_fitted_dist.reshape(NUM_CHAINS*NUM_BAYES_SAMPLES,-1)
y_fitted_prob=np.mean(y_fitted_dist,axis=0)
y_fitted=np.round(y_fitted_prob)
y_predictive_dist=predictive(random.PRNGKey(0), X_test,None)["Y"]
y_predictive_dist=y_predictive_dist.reshape(NUM_CHAINS*NUM_BAYES_SAMPLES,-1)
y_predictive_prob=np.mean(y_predictive_dist,axis=0)
y_predictive=np.round(y_predictive_prob)
print(confusion_matrix(target_train,y_fitted))
print(confusion_matrix(target_test,y_predictive))
