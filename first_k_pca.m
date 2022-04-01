function [pcs,eigenvals,X_hat] = first_k_pca(X,k,varargin)
% This script implements Principal Component Analysis on data matrix x
% Input: X = (n-by-d) Data Matrix
%      : k = number of Principal Components to compute
%
% Output: pcs = the first k Principal Components
%       : eigenvals = the corresponding Eigenvalues
%       : X_hat = the projected data matrix onto the span of the first k PCs
%
% Thomas Anzalone, 10/22/2021

[n,~] = size(X);

% We must center X before building the covariance matrix
X_bar = X - mean(X);

% Build the covariance matrix C
C = (1/n)*(X_bar'*X_bar);

% Find the eigenvalues and eigenvectors of C
[V,D] = eig(C);

% Sort the eigenvalues and get indices to sort eigenvector matrix
[eigenvals,ind] = sort(diag(D),'descend');
pcs_all = V(:,ind');

% Select the first k PCs
pcs = pcs_all(:,1:k);

% Build the projected data Matrix with the first k PCs
X_hat = X*pcs;

end