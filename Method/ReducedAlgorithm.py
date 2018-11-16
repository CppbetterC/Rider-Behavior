import numpy as np
import matplotlib.pyplot as plt

# All sklearn plugin
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
from sklearn.decomposition import FactorAnalysis, FastICA
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


"""
'lle', 'pca', 'mds', 'ProbPCA', 'FactorAnalysis', 'Sammon',
'Isomap', 'LandmarkIsomap', 'Laplacian', 'HessianLLE', 'LTSA',
'DiffusionMaps', 'KernelPCA', 'KernelLDA', 'SNE', 'SymSNE', 'tSNE',
'NPE', 'LPP', 'SPE', 'LLTSA','CCA', 'MVU', 'LandmarkMVU', 'FastMVU', 'LLC',
'ManifoldChart', 'CFA', 'GPLVM', 'Autoencoder', 'NCA', 'MCML', 'LMNN'
"""


class ReducedAlgorithm:

    @staticmethod
    def lle(data, dim=3):
        embedding = LocallyLinearEmbedding(n_components=dim, eigen_solver='dense')
        result = embedding.fit_transform(data)
        return result

    @staticmethod
    def modified_lle(data, dim=3):
        embedding = LocallyLinearEmbedding(method='modified', n_components=dim, eigen_solver='dense')
        result = embedding.fit_transform(data)
        return result

    @staticmethod
    def hessian_lle(data, dim=3):
        embedding = LocallyLinearEmbedding(
            method='hessian', n_components=dim, eigen_solver='dense')
        result = embedding.fit_transform(data)
        return result

    @staticmethod
    def ltsa_lle(data, dim=3):
        embedding = LocallyLinearEmbedding(
            method='ltsa', n_components=dim, eigen_solver='dense')
        result = embedding.fit_transform(data)
        return result

    @staticmethod
    def pca(data, dim=3):
        pca = PCA(n_components=dim)
        pca.fit(data)
        result = pca.transform(data)
        return result

    @staticmethod
    def prob_pca(data, dim=3):
        pass

    @staticmethod
    def kernel_pca(data, dim=3):
        scikit_kpca = KernelPCA(n_components=dim, kernel='rbf', gamma=15)
        result = scikit_kpca.fit_transform(data)
        return result

    @staticmethod
    def sparse_pca(data, dim=3):
        transformer = SparsePCA(
            n_components=dim, random_state=0)
        transformer.fit(data)
        result = transformer.transform(data)
        return result

    @staticmethod
    def mds(data, dim=3):
        embedding = MDS(n_components=dim)
        result = embedding.fit_transform(data)
        return result

    @staticmethod
    def factor_analysis(data, dim=3):
        transformer = FactorAnalysis(n_components=dim, random_state=0)
        X_transformed = transformer.fit_transform(data)
        return X_transformed

    @staticmethod
    def isomap(data, dim=3):
        embedding = Isomap(n_components=dim)
        result = embedding.fit_transform(data)
        return result

    @staticmethod
    def tsne(data, dim):
        X_embedded = TSNE(n_components=dim).fit_transform(data)
        return X_embedded


