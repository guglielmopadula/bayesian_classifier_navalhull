from jax import vmap
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

NUM_BAYES_SAMPLES=10
NUM_CHAINS=4




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
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()


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
test_true=true[300:]
train_false = false[:300]
test_false=false[300:]
train=np.concatenate((train_true,train_false))
test=np.concatenate((test_true,test_false))
target=np.concatenate((np.ones(300,dtype=np.int32),np.zeros(300,dtype=np.int32)))

def model(X,Y):
    N,D_X=X.shape
    w1 = numpyro.sample("w1", dist.Normal(jnp.zeros((D_X, 1)), jnp.ones((D_X, 1))))
    logit=jnp.matmul(X,w1)
    y=numpyro.sample("Y",dist.BernoulliLogits(logit).to_event(1),obs=Y)
rng_key, rng_key_predict = random.split(random.PRNGKey(0))
samples=run_inference(model,rng_key,train,target)


# predict Y_test at inputs X_test
vmap_args = (
    samples,
    random.split(rng_key_predict, NUM_BAYES_SAMPLES * NUM_CHAINS),
)
predictions = vmap(
    lambda samples, rng_key: predict(model, rng_key, samples, test)
)(*vmap_args)
predictions = predictions[..., 0]

# compute mean prediction and confidence interval around median
mean_prediction = jnp.mean(predictions, axis=0)
percentiles = np.percentile(predictions, [5.0, 95.0], axis=0)

# make plots
fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

ax.fill_between(
    test[:, 1], percentiles[0, :], percentiles[1, :], color="lightblue"
)
# plot mean prediction
ax.plot(np.arange(len(mean_prediction)), mean_prediction, "blue", ls="solid", lw=2.0)
ax.plot(np.arange(len(mean_prediction)), target, "red", ls="solid", lw=2.0)

ax.set(xlabel="X", ylabel="Y", title="Mean predictions with 90% CI")

plt.savefig("bnn_plot.pdf")


