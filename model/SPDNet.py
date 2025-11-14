import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import geoopt


class BiMap(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        W = torch.qr(torch.randn(in_dim, out_dim))[0]
        self.W = geoopt.ManifoldParameter(W, manifold=geoopt.Stiefel())

    def forward(self, X):
        return self.W.T @ X @ self.W


class ReBiMap(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        W = torch.qr(torch.randn(out_dim, in_dim))[0]
        self.W = geoopt.ManifoldParameter(W, manifold=geoopt.Stiefel())

    def forward(self, X):
        return self.W @ X @ self.W.T


class ReEig(nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, X):
        eigvals, eigvecs = torch.linalg.eigh(X)
        eigvals = torch.clamp(eigvals, min=self.eps)
        return eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-2, -1)


class LogEig(nn.Module):
    def forward(self, X):
        eigvals, eigvecs = torch.linalg.eigh(X)
        log_eigvals = torch.log(eigvals)
        return eigvecs @ torch.diag_embed(log_eigvals) @ eigvecs.transpose(-2, -1)


class ExpEig(nn.Module):
    def forward(self, X):
        eigvals, eigvecs = torch.linalg.eigh(X)
        exp_eigvals = torch.exp(eigvals)
        return eigvecs @ torch.diag_embed(exp_eigvals) @ eigvecs.transpose(-2, -1)


class SPDEncoder(nn.Module):
    def __init__(self, input_dim, k=3):
        super().__init__()
        self.SPDNet = nn.Sequential(
            BiMap(9, 7), ReEig(), BiMap(7, 5), ReEig(), BiMap(5, 3), ReEig(), LogEig()
        )
        self.k = k
        self.vech_dim = k * (k + 1) // 2

    def forward(self, X):
        C = self.SPDNet(X)
        idx = torch.triu_indices(self.k, self.k)
        vech = C[:, idx[0], idx[1]]
        return vech


class SPDDecoder(nn.Module):
    def __init__(self, k=3, output_dim=9):
        super().__init__()
        self.SPDNet = nn.Sequential(
            ExpEig(),
            ReBiMap(3, 5),
            ReEig(),
            ReBiMap(5, 7),
            ReEig(),
            ReBiMap(7, 9),
            ReEig(),
        )
        self.k = k

    def forward(self, vech):
        batch_size = vech.shape[0]
        X = torch.zeros((batch_size, self.k, self.k), device=vech.device)
        idx = torch.triu_indices(self.k, self.k)
        X[:, idx[0], idx[1]] = vech
        X = (
            X
            + X.transpose(-2, -1)
            - torch.diag_embed(torch.diagonal(X, dim1=-2, dim2=-1))
        )

        return self.SPDNet(X)
