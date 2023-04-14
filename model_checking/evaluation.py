import sys
import numpyro
import numpy as np
from sklearn.metrics import confusion_matrix



def compute_bayesian_p_value(y_fitted_dist,y):
    mean_dist=np.mean(y_fitted_dist,axis=-1)
    print(mean_dist)
    mean=np.mean(y)
    std_dist=np.std(y_fitted_dist,axis=-1)
    std=np.std(y)
    print("Mean pvalue is", np.mean(mean_dist>=mean))
    print("Std pvalue is", np.mean(std_dist>=std))

    
    

for name in ["bayes_logreg_mle","bayes_nb_zero","bayes_nb_mle","bayes_logreg_zero","bayesian_nn_zero","bayesian_nn_mle"]:
    values=np.load("./npy_files/"+name+".npy",allow_pickle=True).item()
    sys.stdout=open("./model_checking/"+name+".txt","w+")
    posterior_samples=values["posterior_samples"]
    y_fitted_dist=values["y_fitted_dist"]
    y_predictive_dist=values["y_predictive_dist"]
    target_train=values["target_train"]
    target_test=values["target_test"]
    NUM_CHAINS=values["num_chains"]
    NUM_BAYES_SAMPLES=values["num_bayes_samples"]
    y_fitted_prob=np.mean(y_fitted_dist,axis=0)
    y_fitted=np.round(y_fitted_prob)
    y_predictive_dist=y_predictive_dist.reshape(NUM_CHAINS*NUM_BAYES_SAMPLES,-1)
    y_predictive_prob=np.mean(y_predictive_dist,axis=0)
    y_predictive=np.round(y_predictive_prob)
    numpyro.diagnostics.print_summary(posterior_samples,prob=0.95)
    compute_bayesian_p_value(y_fitted_dist,target_train)
    print(confusion_matrix(target_train,y_fitted))
    print(confusion_matrix(target_test,y_predictive))
    sys.stdout.close()
    sys.stdout=sys.__stdout__

    
