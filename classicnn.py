from flax import linen as nn
import numpy as np
import tensorflow as tf
from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct                # Flax dataclasses
import optax                           # Common loss functions and optimizers
from flax import linen as nn  # Linen API
import jax
import jax.numpy as jnp




class Classifier(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(features=1000)(x)
    x = nn.relu(x)
    x = nn.Dense(features=500)(x)
    x = nn.relu(x)
    x = nn.Dense(features=250)(x)
    x = nn.relu(x)
    x = nn.Dense(features=125)(x)
    x = nn.relu(x)
    x = nn.Dense(features=50)(x)
    x = nn.relu(x)
    x = nn.Dense(features=20)(x)
    x = nn.relu(x)
    x = nn.Dense(features=2)(x)
    return x
  

@struct.dataclass
class Metrics(metrics.Collection):
  accuracy: metrics.Accuracy
  loss: metrics.Average.from_output('loss')

class TrainState(train_state.TrainState):
  metrics: Metrics

def create_train_state(module, rng, learning_rate, momentum):
  """Creates an initial `TrainState`."""
  params = module.init(rng, jnp.ones([1, train[0].shape[0]]))['params'] # initialize parameters by passing a template image
  tx = optax.adamw(learning_rate)
  return TrainState.create(
      apply_fn=module.apply, params=params, tx=tx,
      metrics=Metrics.empty())

@jax.jit
def train_step(state, batch):
  """Train for a single step."""
  def loss_fn(params):
    x,y=batch
    logits = state.apply_fn({'params': params}, x)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=y).mean()
    return loss
  grad_fn = jax.grad(loss_fn)
  grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state

@jax.jit
def compute_metrics(*, state, batch):
  x,y=batch
  logits = state.apply_fn({'params': state.params}, x)
  loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=y).mean()
  metric_updates = state.metrics.single_from_model_output(
    logits=logits, labels=y, loss=loss)
  metrics = state.metrics.merge(metric_updates)
  state = state.replace(metrics=metrics)
  return state




BATCH_SIZE=20
NUM_EPOCHS=1000
NUM_SAMPLES=600
true=np.load("positive_data.npy")
false=np.load("negative_data.npy")
train_true = true[:300]     
test_true=true[300:]
train_false = false[:300]
test_false=false[300:]
train_perm=np.random.permutation(600)
test_perm=np.random.permutation(600)

tf.random.set_seed(0)
train=np.concatenate((train_true,train_false))
test=np.concatenate((test_true,test_false))
target=np.concatenate((np.ones(300,dtype=np.int32),np.zeros(300,dtype=np.int32)))
train_dataset = tf.data.Dataset.from_tensor_slices((train, target))
test_dataset = tf.data.Dataset.from_tensor_slices((test, target))
BATCH_SIZE = 600
SHUFFLE_BUFFER_SIZE = 100
train_ds = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_ds = test_dataset.batch(BATCH_SIZE)

classifier=Classifier()
init_rng = jax.random.PRNGKey(0)
learning_rate = 0.0001
momentum = 0.5
state = create_train_state(classifier, init_rng, learning_rate, momentum)
del init_rng  # Must not be used anymore.

num_steps_per_epoch = train_ds.cardinality().numpy() // NUM_EPOCHS
metrics_history = {'train_loss': [],
                   'train_accuracy': [],
                   'test_loss': [],
                   'test_accuracy': []}

for i in range(NUM_EPOCHS):
    for step,batch in enumerate(train_ds.as_numpy_iterator()):

    # Run optimization steps over training batches and compute batch metrics
        state = train_step(state, batch) # get updated train state (which contains the updated parameters)
        state = compute_metrics(state=state, batch=batch) # aggregate batch metrics

    for metric,value in state.metrics.compute().items(): # compute metrics
        metrics_history[f'train_{metric}'].append(value) # record metrics
        state = state.replace(metrics=state.metrics.empty()) # reset train_metrics for next training epoch

            # Compute metrics on the test set after each training epoch
    test_state = state
    for test_batch in test_ds.as_numpy_iterator():
        test_state = compute_metrics(state=test_state, batch=test_batch)

    for metric,value in test_state.metrics.compute().items():
        metrics_history[f'test_{metric}'].append(value)

    print(f"train epoch: {i}, "
        f"loss: {metrics_history['train_loss'][-1]}, "
        f"accuracy: {metrics_history['train_accuracy'][-1] * 100}")
    print(f"test epoch: {i}, "
        f"loss: {metrics_history['test_loss'][-1]}, "
        f"accuracy: {metrics_history['test_accuracy'][-1] * 100}")
    
import matplotlib.pyplot as plt  # Visualization

# Plot loss and accuracy in subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.set_title('Loss')
ax2.set_title('Accuracy')
for dataset in ('train','test'):
  ax1.plot(metrics_history[f'{dataset}_loss'], label=f'{dataset}_loss')
  ax2.plot(metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')
ax1.legend()
ax2.legend()
plt.show()
plt.clf()

@jax.jit
def pred_step(state, batch):
  x,y=batch
  logits = state.apply_fn({'params': state.params}, x)
  return logits.argmax(axis=1)

test_batch = test_ds.as_numpy_iterator().next()
pred = pred_step(state, test_batch)
print(np.sum(np.abs(pred-target))/(np.sum(target)))