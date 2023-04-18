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

  model = """
  data {
  int<lower=0> N;
  int<lower=1> K;  
  array[N] int y;           // estimated treatment effects
  matrix[N,K] x; // standard error of effect estimates
  matrix[N,K] x_new; // standard error of effect estimates

  }
  parameters {
  matrix[2, K] mu;                // population treatment effect
  array[K] real<lower=0> sigma;          // unscaled deviation from mu by school
  }
  model {
  to_vector(mu) ~ normal(0, 5);
  sigma ~ gamma(1, 1);
  for (n in 1:N)
      x[n] ~ normal(to_vector(mu[y[n]+1]),sigma);
  }
  generated quantities {
  array[N] real p_pred;
  array[N] real p_new;
  array[N] int y_pred;
  array[N] int y_new;
  array[N] real like_pred;


  for (n in 1:N){
  p_new[n]=exp(normal_lpdf(x_new[n]|to_vector(mu[2]),sigma))/(exp(normal_lpdf(x_new[n]|mu[1],sigma))+exp(normal_lpdf(x_new[n]|mu[2],sigma)));
  y_new[n] = bernoulli_rng(p_new[n]);}
  for (n in 1:N){
  p_pred[n]=exp(normal_lpdf(x[n]|to_vector(mu[2]),sigma))/(exp(normal_lpdf(x[n]|mu[1],sigma))+exp(normal_lpdf(x[n]|mu[2],sigma)));
  y_pred[n] = bernoulli_rng(p_pred[n]);
  like_pred[n]=bernoulli_lpmf(y_pred[n]|p_pred[n]);

  }
  }

  """

  """
  generated quantities {
  array[N] int y_new;
  for (n in 1:N)
  y_new[n] = bernoulli_logit_rng(x_new[n] * alpha+beta);
  array[N] int y_pred;
  for (n in 1:N)
  y_pred[n] = bernoulli_logit_rng(x[n] * alpha+beta);
  }
  """
  data={"N":600,"K":5,"y":target_train,"x":train,"x_new":test}
  posterior = stan.build(model, data=data)
  fit = posterior.sample(num_chains=4, num_samples=1000)
  y_fitted_dist=fit["y_pred"]
  y_fitted_prob=np.mean(y_fitted_dist,axis=1).reshape(-1)
  y_fitted=np.zeros_like(y_fitted_prob)*(y_fitted_prob<0.5)+np.ones_like(y_fitted_prob)*(y_fitted_prob>=0.5)
  y_predictive_dist=fit["y_new"]
  y_predictive_prob=np.mean(y_predictive_dist,axis=1)
  y_predictive=np.zeros_like(y_predictive_prob)*(y_predictive_prob<0.5)+np.ones_like(y_predictive_prob)*(y_predictive_prob>=0.5)
  print(az.summary(fit,var_names=["~y_pred","~y_new","~p_new","~p_pred","~like_pred"]))
  print(confusion_matrix(target_train,y_fitted))
  print(confusion_matrix(target_test,y_predictive))
  df = fit.to_frame()
  divergences=np.sum(df["divergent__"]>=1)
  np.save("./npy_files/bayes_nbpp_zero.npy",{"fit":fit,"y_fitted_dist":y_fitted_dist,"y_predictive_dist":y_predictive_dist,"target_train":target_train,"target_test":target_test,"num_chains":4,"num_bayes_samples":1000,"num_warmup":1000,"divergences":divergences})
