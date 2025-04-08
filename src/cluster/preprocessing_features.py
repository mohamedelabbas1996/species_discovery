from sklearn import preprocessing
from sklearn.decomposition import PCA


def standardize(features):
    scaler = preprocessing.StandardScaler().fit(features)
    features = scaler.transform(features)
    print("standardized features")
    return features


def dimension_reduction(features):
    pca = PCA(n_components=384)
    features = pca.fit_transform(features)
    print("PCA performed")
    return features
