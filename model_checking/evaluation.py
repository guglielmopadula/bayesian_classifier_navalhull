import sys
import numpy as np
import sys
from pathlib import Path
from sklearn.metrics import confusion_matrix,accuracy_score
import arviz as az


##############REMEMBER TO DO export PYTHONPATH="${PYTHONPATH}:." && export XLA_PYTHON_CLIENT_MEM_FRACTION=.10

def compute_bayesian_p_value(y_fitted_dist,y):
    mean_dist=np.mean(y_fitted_dist,axis=0)
    mean=np.mean(y)
    print("Mean pvalue is", np.mean(mean_dist>=mean))



def compute_rse(posterior):
    l=[]
    for k,v in posterior.items():
        v=v.reshape(-1,int(np.prod(v.shape[2:])))
        v=v.reshape(v.shape[0],-1)
        l.append(np.max(np.std(v[:,np.mean(v,axis=0)!=0],axis=0)/np.mean(v[:,np.mean(v,axis=0)!=0],axis=0)))
    return np.max(l)

def compute_rhat(posterior):
    rhat_dict=az.rhat(posterior).to_dict()["data_vars"]
    l=[]
    for k,v in rhat_dict.items():
        t=np.array(v["data"])
        t=t.reshape(-1)
        l.append(np.max(t[np.logical_not(np.isnan(t))]))
    return np.max(l)
    







NUM_SAMPLES=600



for name in ["bayes_logreg_zero","bayesian_nn_zero","bayes_nb_zero","bayes_nbpp_zero","bayes_gam_zero","bayes_lda_zero","bayes_qda_zero"]:
    sys.stdout=open("./model_checking/"+name+".txt","w+")
    data=np.load("npy_files/data.npy",allow_pickle=True).item()
    train=data["train"]
    test=data["test"]
    values=np.load("./npy_files/"+name+".npy",allow_pickle=True).item()
    fit=values["fit"]
    inference_data=az.from_pystan(fit,log_likelihood="like_pred")
    inference_dict=inference_data.to_dict()
    posterior_dict=inference_dict["posterior"]
    posterior_dict.pop("y_pred",None)
    posterior_dict.pop("y_new",None)
    posterior_dict.pop("p_pred",None)
    posterior_dict.pop("p_new",None)
    posterior_dict.pop("logit_new",None)
    posterior_dict.pop("logit_pred",None)
    y_fitted_dist=values["y_fitted_dist"]
    y_predictive_dist=values["y_predictive_dist"]
    target_train=values["target_train"]
    target_test=values["target_test"]
    divergences=values["divergences"]
    NUM_CHAINS=values["num_chains"]
    NUM_BAYES_SAMPLES=values["num_bayes_samples"]
    y_fitted_prob=np.mean(y_fitted_dist,axis=1)
    y_fitted=np.round(y_fitted_prob)
    y_predictive_prob=np.mean(y_predictive_dist,axis=1)
    y_predictive=np.round(y_predictive_prob)
    print("number of divergences is", divergences)
    print("train accuracy score is", accuracy_score(target_train,y_fitted,normalize=True))
    print("test accuracy score is", accuracy_score(target_test,y_predictive,normalize=True))
    compute_bayesian_p_value(y_fitted_dist,target_train)
    print("rse is", compute_rse(posterior_dict))
    print("waic is", az.waic(inference_data).elpd_waic)
    print("rhat is", compute_rhat(posterior_dict))
    sys.stdout.close()
    sys.stdout=sys.__stdout__


    
