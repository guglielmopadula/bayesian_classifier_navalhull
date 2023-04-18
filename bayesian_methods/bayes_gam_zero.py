import os
import multiprocessing

import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import pairwise_distances
import scipy
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import multiprocessing
import stan
import arviz as az
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




def compute_features(train):
    NUM_DATA=train.shape[0]
    NUM_FEATURES=train.shape[1]

    m=4 #must be 2*m>d+1
    simple_eta=eta_constructor(m,NUM_FEATURES)
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
    Zk=scipy.linalg.null_space(T.T@uk)
    delta_coeff=uk@dk@Zk
    alpha_coeff=T
    X_train=np.concatenate((delta_coeff,alpha_coeff),axis=1)
    return X_train




def partition0(max_range, S):
    K = len(max_range)
    return np.array([i for i in itertools.product(*(range(i+1) for i in max_range)) if sum(i)<=S])            
if __name__ == "__main__":
    m=4 #must be 2*m>d+1
    data=np.load("npy_files/data.npy",allow_pickle=True).item()
    target_train=data["target_train"]
    train=data["train"]
    test=data["test"]
    target_test=data["target_test"]
    NUM_DATA=train.shape[0]
    NUM_FEATURES=train.shape[1]
    X_train=compute_features(train)
    X_test=compute_features(test)



    model = """
    data {
    int<lower=0> N;
    int<lower=1> K;  
    array[N] int y;           // estimated treatment effects
    matrix[N,K] x; // standard error of effect estimates
    matrix[N,K] x_new; // standard error of effect estimates

    }
    parameters {
    real beta;                // population treatment effect
    vector[K] alpha;          // unscaled deviation from mu by school
    }
    model {
    alpha ~ normal(0, 5);
    beta ~ normal(0, 5);
    y ~ bernoulli_logit(x*alpha+beta);
    }
    generated quantities {
    array[N] int y_new;
    array[N] real logit_new;
    for (n in 1:N){
    logit_new[n]=x_new[n] * alpha+beta;
    y_new[n] = bernoulli_logit_rng(logit_new[n]);}
    array[N] int y_pred;
    array[N] real logit_pred;
    array[N] real like_pred;

    for (n in 1:N){
    logit_pred[n]=x[n] * alpha+beta;
    y_pred[n] = bernoulli_logit_rng(logit_pred[n]);
    like_pred[n]=bernoulli_logit_lpmf(y_pred[n]|logit_pred[n]);}
    }"""

    data={"N":600,"K":X_train.shape[1],"y":target_train,"x":X_train,"x_new":X_test}
    posterior = stan.build(model, data=data)
    fit = posterior.sample(num_chains=4, num_samples=1000)
    y_fitted_dist=fit["y_pred"]
    y_predictive_dist=fit["y_new"]
    y_fitted_prob=np.mean(y_fitted_dist,axis=1).reshape(-1)
    y_fitted=np.zeros_like(y_fitted_prob)*(y_fitted_prob<0.5)+np.ones_like(y_fitted_prob)*(y_fitted_prob>=0.5)
    y_predictive_prob=np.mean(y_predictive_dist,axis=1)
    y_predictive=np.zeros_like(y_predictive_prob)*(y_predictive_prob<0.5)+np.ones_like(y_predictive_prob)*(y_predictive_prob>=0.5)
    print(az.summary(fit,var_names=["~y_pred","~y_new","~logit_pred","~logit_new","~like_pred"]))
    print(confusion_matrix(target_train,y_fitted))
    print(confusion_matrix(target_test,y_predictive))
    df = fit.to_frame()
    divergences=np.sum(df["divergent__"]>=1)
    np.save("./npy_files/bayes_gam_zero.npy",{"fit":fit,"y_fitted_dist":y_fitted_dist,"y_predictive_dist":y_predictive_dist,"target_train":target_train,"target_test":target_test,"num_chains":4,"num_bayes_samples":1000,"num_warmup":1000,"divergences":divergences})
