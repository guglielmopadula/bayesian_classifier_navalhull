from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
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
clf = LogisticRegression(random_state=0,penalty=None,max_iter=300).fit(train, target_train)
print(confusion_matrix(target_train,clf.predict(train)))
print(confusion_matrix(target_test,clf.predict(test)))