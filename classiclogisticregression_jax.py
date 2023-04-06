from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import jax.numpy as jnp
from sklearn.metrics import classification_report, confusion_matrix
from jax import grad
from tqdm import trange
import matplotlib.pyplot as plt
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
train=train[train_perm]
target_train=target[train_perm]
test=test[test_perm]
target_test=target[test_perm]
print(target_train)
print(target_test)
def logistic(r):
    return 1 / (1 + jnp.exp(-r))

def predict(c, w, X):
    return logistic(jnp.dot(X, w) + c)

def cost(c, w, X, y, eps=1e-14, lmbd=0): #0.05
    n = y.size
    p = predict(c, w, X)
    p = jnp.clip(p, eps, 1 - eps)  # bound the probabilities within (0,1) to avoid ln(0)
    return -jnp.sum(y * jnp.log(p) + (1 - y) * jnp.log(1 - p)) / n + 0.5 * lmbd * (
        jnp.dot(w, w) + c * c
    )

c_0 = 1.0
w_0 = 1.0e-5 * jnp.ones_like(true[0])
n_iter = 1000
eta = 5e-2
tol = 1e-6
w = w_0
c = c_0

new_cost = float(cost(c, w, train, target_train))
cost_hist = [new_cost]
for i in trange(n_iter):
    c_current = c
    c -= eta * grad(cost, argnums=0)(c_current, w, train, target_train)
    w -= eta * grad(cost, argnums=1)(c_current, w, train, target_train)
    new_cost = float(cost(c, w, train, target_train))
    cost_hist.append(new_cost)
    if (i > 20) and (i % 10 == 0):
        if jnp.abs(cost_hist[-1] - cost_hist[-20]) < tol:
            print(f"Exited loop at iteration {i}")
            break

_, ax = plt.subplots()
plt.semilogy(cost_hist)
ax.grid()
_ = ax.set(xlabel="Iteration", ylabel="Cost value", title="Convergence history")

y_fit_proba = predict(c, w, train)
y_fit = jnp.array(y_fit_proba)
y_fit = jnp.where(y_fit < 0.5, y_fit, 1.0)
y_fit = jnp.where(y_fit >= 0.5, y_fit, 0.0)
print(confusion_matrix(target_train, y_fit))


y_pred_proba = predict(c, w, test)
y_pred = jnp.array(y_pred_proba)
y_pred = jnp.where(y_pred < 0.5, y_pred, 1.0)
y_pred = jnp.where(y_pred >= 0.5, y_pred, 0.0)

print(confusion_matrix(target_test, y_pred))