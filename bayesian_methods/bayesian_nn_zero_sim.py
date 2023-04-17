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
  matrix[K,3] alpha;
  matrix[3,1] gamma;
  row_vector[1] delta;
}

model {
    to_vector(alpha) ~ normal(0, 0.05);
    to_vector(gamma) ~ normal(0, 0.05);
    to_vector(delta) ~ normal(0, 0.05);
    for (n in 1:N)
    y[n] ~ bernoulli_logit(to_vector(tanh(x[n]*alpha)*gamma+delta));
}
generated quantities {
  array[N] int y_new;
  array[N] int y_pred;



  for (n in 1:N){
    y_new[n] = bernoulli_logit_rng(to_vector(tanh(x_new[n]*alpha)*gamma+delta))[1];
    }
  for (n in 1:N){
    y_pred[n] = bernoulli_logit_rng(to_vector(tanh(x[n]*alpha)*gamma+delta))[1];
    }
    
    }
"""

data={"N":600,"K":5,"y":target_train,"x":train,"x_new":test}
posterior = stan.build(model, data=data)
fit = posterior.sample(num_chains=4, num_samples=1000)
y_fitted_dist=fit["y_pred"]
y_predictive_dist=fit["y_new"]
y_fitted_prob=np.mean(y_fitted_dist,axis=1).reshape(-1)
y_fitted=np.zeros_like(y_fitted_prob)*(y_fitted_prob<0.5)+np.ones_like(y_fitted_prob)*(y_fitted_prob>=0.5)
y_predictive_prob=np.mean(y_predictive_dist,axis=1)
y_predictive=np.zeros_like(y_predictive_prob)*(y_predictive_prob<0.5)+np.ones_like(y_predictive_prob)*(y_predictive_prob>=0.5)
print(az.summary(fit,var_names=["~y_pred","~y_new"]))

print(confusion_matrix(target_train,y_fitted))
print(confusion_matrix(target_test,y_predictive))
#np.save("./npy_files/bayes_logreg_zero.npy",{"posterior_samples":posterior_samples,"y_fitted_dist":y_fitted_dist,"y_predictive_dist":y_predictive_dist,"target_train":target_train,"target_test":target_test,"num_chains":NUM_CHAINS,"num_bayes_samples":NUM_BAYES_SAMPLES,"num_warmup":NUM_WARMUP,"divergences":divergences})
