import stan                                                                                                                                        
import pandas as pd                                                                                  
import numpy as np                                                                                   
                                                                                                     
code = """                                                                                           
data {                                                                                               
  int N; //the number of training observations                                                       
  int N2; //the number of test observations                                                          
  int K; //the number of features                                                                    
  array[N] int y; //the response                                                                           
  matrix[N,K] X; //the model matrix                                                                  
  matrix[N2,K] new_X; //the matrix for the predicted values                                          
}                                                                                                    
parameters {                                                                                         
  real alpha;                                                                                        
  vector[K] beta; //the regression parameters                                                        
}                                                                                                    
transformed parameters {                                                                             
  vector[N] linpred;                                                                                 
  linpred = alpha+X*beta;                                                                            
}                                                                                                    
model {                                                                                              
  alpha ~ cauchy(0,10); //prior for the intercept following Gelman 2008                              
                                                                                                     
  for(i in 1:K)                                                                                      
    beta[i] ~ student_t(1, 0, 0.03);                                                                 
                                                                                                     
  y ~ bernoulli_logit(linpred);                                                                      
}                                                                                                    
generated quantities {                                                                        
  vector[N2] y_pred;                                                                               
  y_pred = alpha+new_X*beta; //the y values predicted by the model                             
}                                                                                                    
"""               

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

