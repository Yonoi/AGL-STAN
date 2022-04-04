import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg

import torch

class CalculateFilter(object):
    def __init__(self, device, filter_type='laplacian'):
        self._filter_type = filter_type
        self._device = device
    
    def _calculate_laplacian(self, adj):
        adj = sp.csr_matrix(adj)
        d = np.array(adj.sum(1))
        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        L = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).T.dot(d_mat_inv_sqrt)
        return L
    
    def _calculate_laplacian_tensor(self, adj):
        # Consider the adjacent matrix is not very large.
        # I don't implement the matrix multiplication in the way of sparse.
        d = adj.sum(1)
        d_inv_sqrt = torch.pow(d, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        I = torch.eye(adj.size(0), device=self._device)
        L = I - torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return L

    def _calculate_scaled_laplacian(self, adj, lambda_max=2, undirected=True):
        if undirected:
            adj = np.maximum.reduce([adj, adj.T])
        L = self._calculate_laplacian(adj)
        
        # calculate the maximum eigenvalue of adjacent matrix
        if lambda_max is None:
            eigenvalues, _ = linalg.eigsh(L, 1, which='LM')
            lambda_max = eigenvalues[0]
        
        L = sp.csr_matrix(L)
        I = sp.identity(L.shape[0], format='csr').astype(L.dtype)
        L = (2 / lambda_max * L) - I
        return L.astype(np.float32)

    def _calculate_scaled_laplacian_tensor(self, adj, lambda_max=2, undirected=True):
        if undirected:
            adj = torch.maximum(adj, adj.T)
        L = self._calculate_laplacian_tensor(adj)

        # calculate the maximum eigenvalue of adjacent matrix
        if lambda_max is None:
            eigenvalues, _ = torch.linalg.eigh(L)
            lambda_max = eigenvalues.max()
        
        I = torch.eye(L.shape[0], device=self._device)
        L = (2 / lambda_max * L ) - I 
        return L

    def _calculate_random_walk(self, adj):
        adj = sp.csr_matrix(adj)
        d = np.array(adj.sum(1))
        d_inv = np.power(d, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        random_walk_mx = d_mat_inv.dot(adj).tocoo()
        return random_walk_mx
    
    def _calculate_random_walk_tensor(self, adj):
        d = adj.sum(1)
        d_inv = torch.pow(d, -1)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diag(d_inv)
        random_walk_mx = torch.mm(d_mat_inv, adj)
        return random_walk_mx


    def _build_sparse_matrix(self, L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=self._device, dtype=torch.float32) # device
        return L
    
    def _build_sparse_matrix_tensor(self, L):
        L = L.to_sparse()
        return L

    def transform(self, adj):
        conv_filters = []
        if type(adj) != torch.Tensor:

            if self._filter_type == 'laplacian':
                conv_filters.append(self._calculate_scaled_laplacian(adj, lambda_max=None))
            elif self._filter_type == 'random_walk':
                conv_filters.append(self._calculate_random_walk(adj).T)
            elif self._filter_type == 'dual_random_walk':
                conv_filters.append(self._calculate_random_walk(adj).T)
                conv_filters.append(self._calculate_random_walk(adj.T).T)
            else:
                conv_filters.append(self._calculate_scaled_laplacian(adj))

            conv_filters = [
                self._build_sparse_matrix(conv_filter)
                for conv_filter in conv_filters
            ]

        else:
            if self._filter_type == 'laplacian':
                conv_filters.append(self._calculate_scaled_laplacian_tensor(adj, lambda_max=None))
            elif self._filter_type == 'random_walk':
                conv_filters.append(self._calculate_random_walk_tensor(adj).T)
            elif self._filter_type == 'dual_random_walk':
                conv_filters.append(self._calculate_random_walk_tensor(adj).T)
                conv_filters.append(self._calculate_random_walk_tensor(adj.T).T)
            else:
                conv_filters.append(self._calculate_scaled_laplacian_tensor(adj))

            conv_filters = [
                self._build_sparse_matrix_tensor(conv_filter)
                for conv_filter in conv_filters
            ]
            
        return conv_filters

    