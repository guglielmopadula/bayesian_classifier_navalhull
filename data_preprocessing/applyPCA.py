import numpy as np
true=np.load("npy_files/positive_data.npy")
false=np.load("npy_files/negative_data.npy")
print(true.shape)

from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca.fit(true)
true_red=pca.transform(true)
false_red=pca.transform(false)
np.save("npy_files/positive_data_red.npy",true_red)
np.save("npy_files/negative_data_red.npy",false_red)