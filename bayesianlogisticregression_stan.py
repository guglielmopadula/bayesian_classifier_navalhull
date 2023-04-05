import stan                                                                                                                                        
import pandas as pd                                                                                  
import numpy as np                                                                                   
                                                                                                     

with open('test.stan') as f:
    file = f.readlines()

code=""
for i in range(len(file)):
  code=code+str(file[i])
print(code)
true=np.load("positive_data.npy")
false=np.load("negative_data.npy")
train_true = true[:300]     
test_true=true[300:]
train_false = false[:300]
test_false=false[300:]
train=np.concatenate((train_true,train_false))
test=np.concatenate((test_true,test_false))
target=np.concatenate((np.ones(300,dtype=np.int32),np.zeros(300,dtype=np.int32)))
data = {                                                                                             
    'N': 600,                                                                                        
    'N2': 600,                                                                                     
    'K': train[0].shape[0],                                                                                        
    'y': target,                                                                                     
    'X': train,                                                                                      
    'new_X': test,                                                                                   
}                                                                                                    
                                                                                                     
posterior = stan.build(code,data)                                                               
fit = posterior.sample(num_chains=4, num_samples=1000)                                                        
target = np.mean(fit['y_pred'], axis=0)                                                               

