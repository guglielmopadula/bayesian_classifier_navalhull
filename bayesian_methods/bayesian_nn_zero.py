import os
import multiprocessing
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)
import arviz as az
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics import confusion_matrix
import time
import stan
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import StandardScaler

NUM_BAYES_SAMPLES=1000
NUM_WARMUP=3000
NUM_CHAINS=4



# create artificial regression dataset




if __name__ == "__main__":

  data=np.load("./npy_files/data.npy",allow_pickle=True).item()
  target_train=data["target_train"]
  train=data["train"]
  test=data["test"]
  target_test=data["target_test"]
  scaler=StandardScaler()
  scaler.fit(train)
  train=scaler.transform(train)
  test=scaler.transform(test)
  model = """
  data {
  int<lower=0> N;
  int<lower=1> K;  
  array[N] int y;           // estimated treatment effects
  matrix[N,K] x; // standard error of effect estimates
  matrix[N,K] x_new; // standard error of effect estimates

  }
  parameters {
  matrix[K,4] alpha;
  row_vector[4] beta;
  matrix[4,3] gamma;
  row_vector[3] delta;
  matrix[3,2] epsilon;
  row_vector[2] zeta;
  matrix[2,1] eta;
  row_vector[1] theta;
  }

  model {
  to_vector(alpha) ~ normal(0,2);
  to_vector(beta) ~ normal(0,2);
  to_vector(gamma) ~ normal(0,2);
  to_vector(delta) ~ normal(0,2);
  to_vector(epsilon) ~ normal(0,2);
  to_vector(zeta) ~ normal(0,2);
  to_vector(eta) ~ normal(0,2);
  to_vector(theta) ~ normal(0,2);
  for (n in 1:N)
  y[n] ~ bernoulli_logit(to_vector(tanh(tanh(tanh(x[n]*alpha+beta)*gamma+delta)*epsilon+zeta)*eta+theta));
  }
  generated quantities {
  array[N] int y_new;
  array[N] int y_pred;
  array[N] real logit_new;
  array[N] real logit_pred;
  array[N] real like_pred;




  for (n in 1:N){
  logit_new[n]=(tanh(tanh(tanh(x_new[n]*alpha+beta)*gamma+delta)*epsilon+zeta)*eta+theta)[1];
  y_new[n] = bernoulli_logit_rng(logit_new[n]);
  }
  for (n in 1:N){
  logit_pred[n]=(tanh(tanh(tanh(x[n]*alpha+beta)*gamma+delta)*epsilon+zeta)*eta+theta)[1];
  y_pred[n] = bernoulli_logit_rng(logit_pred[n]);
  like_pred[n]=bernoulli_logit_lpmf(y_pred[n]|logit_pred[n]);
  }

  }
  """

  data={"N":600,"K":5,"y":target_train,"x":train,"x_new":test}
  posterior = stan.build(model, data=data)
  fit = posterior.sample(num_chains=4, num_samples=1000,num_warmup=4000)
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
  np.save("./npy_files/bayesian_nn_zero.npy",{"fit":fit,"y_fitted_dist":y_fitted_dist,"y_predictive_dist":y_predictive_dist,"target_train":target_train,"target_test":target_test,"num_chains":4,"num_bayes_samples":1000,"num_warmup":1000,"divergences":divergences})
