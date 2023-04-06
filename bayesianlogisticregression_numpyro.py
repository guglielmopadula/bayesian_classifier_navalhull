from jax import vmap
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import dill
import jax
from sklearn.metrics import confusion_matrix
import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
NUM_BAYES_SAMPLES=6000
NUM_CHAINS=4
print(jax.device_count())

def invert_permutation(p):
    p = np.asanyarray(p) # in case p is a tuple, etc.
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s

def run_inference(model, rng_key, X, Y):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=NUM_BAYES_SAMPLES,
        num_samples=NUM_BAYES_SAMPLES,
        num_chains=NUM_CHAINS
    )
    mcmc.run(rng_key, X, Y)
    #mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples(group_by_chain=True)


# helper function for prediction
def predict(model, rng_key, samples, X):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(X=X, Y=None)
    return model_trace["Y"]["value"]


# create artificial regression dataset


true=np.load("positive_data.npy")
false=np.load("negative_data.npy")
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

classic_coeffs=np.load("classiclogisticregression_coef.npy")
alpha_mean=classic_coeffs[-1]
beta_mean=classic_coeffs[:-1]
alpha_mean=jnp.array(alpha_mean.reshape(1,1))
beta_mean=jnp.array(beta_mean.reshape(-1,1))

def model(X,Y):
    N,D_X=X.shape
    alpha = numpyro.sample("alpha", dist.Normal(alpha_mean, 5*jnp.ones((1, 1))))
    beta = numpyro.sample("beta", dist.Normal(beta_mean, 5*jnp.ones((D_X, 1))))
    logit=jnp.matmul(X,beta)+alpha
    y=numpyro.sample("Y",dist.BernoulliLogits(logit).to_event(1),obs=Y)
    return y
rng_key, rng_key_predict = random.split(random.PRNGKey(0))

shape=train[0].shape[0]

samples=run_inference(model,rng_key,train,target_train)
samples["alpha"]=samples["alpha"].reshape(NUM_CHAINS*NUM_BAYES_SAMPLES,-1,1)
samples["beta"]=samples["beta"].reshape(NUM_CHAINS*NUM_BAYES_SAMPLES,-1,1)

print(samples["alpha"].shape)
print(samples["beta"].shape)
output_dict = {}
output_dict['model']=model
output_dict['samples']=samples
with open('file.pkl', 'wb') as handle:
    dill.dump(output_dict, handle)
'''
with open('file.pkl', 'rb') as in_strm:
    output_dict = dill.load(in_strm)
model=output_dict['model']
samples=output_dict['samples']
'''
print(numpyro.diagnostics.summary(samples["beta"].reshape(NUM_CHAINS,NUM_BAYES_SAMPLES,-1)))

# predict Y_test at inputs X_test
vmap_args = (
    samples,
    random.split(rng_key_predict, NUM_BAYES_SAMPLES * NUM_CHAINS),
)
predictions = vmap(
    lambda samples, rng_key: predict(model, rng_key, samples, test)
)(*vmap_args)
predictions = predictions[..., 0]

fitted = vmap(
    lambda samples, rng_key: predict(model, rng_key, samples, train)
)(*vmap_args)
fitted = fitted[..., 0]

predictions=predictions[:,invert_permutation(test_perm)]
target_test=target_test[invert_permutation(test_perm)]
fitted=fitted[:,invert_permutation(train_perm)]
target_train=target_train[invert_permutation(train_perm)]

# compute mean prediction and confidence interval around median
mean_prediction = jnp.mean(predictions, axis=0)
percentiles_pred = np.percentile(predictions, [5.0, 95.0], axis=0)


# make plots
fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

ax.fill_between(
    test[:, 1], percentiles_pred[0, :], percentiles_pred[1, :], color="lightblue"
)
# plot mean prediction
ax.plot(np.arange(len(mean_prediction)), mean_prediction, "blue", ls="solid", lw=2.0)
ax.plot(np.arange(len(mean_prediction)), target_test, "red", ls="solid", lw=2.0)

ax.set(xlabel="X", ylabel="Y", title="Mean predictions with 90% CI")

plt.savefig("bnn_plot_pred.pdf")


# compute mean prediction and confidence interval around median
mean_fit = jnp.mean(fitted, axis=0)
percentiles_fit = np.percentile(fitted, [5.0, 95.0], axis=0)

# make plots
fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

ax.fill_between(
    test[:, 1], percentiles_fit[0, :], percentiles_fit[1, :], color="lightblue"
)
# plot mean prediction
ax.plot(np.arange(len(mean_fit)), mean_fit, "blue", ls="solid", lw=2.0)
ax.plot(np.arange(len(mean_fit)), target_train, "red", ls="solid", lw=2.0)

ax.set(xlabel="X", ylabel="Y", title="Mean predictions with 90% CI")

plt.savefig("bnn_plot_fit.pdf")

mean_fit=np.ones_like(mean_fit)*(mean_fit>0.5)+np.zeros_like(mean_fit)*(mean_fit<=0.5)
mean_prediction=np.ones_like(mean_prediction)*(mean_prediction>0.5)+np.zeros_like(mean_prediction)*(mean_prediction<=0.5)
print(confusion_matrix(target_train,mean_fit))
print(confusion_matrix(target_test,mean_prediction))