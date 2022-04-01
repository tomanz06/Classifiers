function [y,F] = kmeans_alg(X,K)
% This script implements the kmeans algorithm
% Input: X = D-Dimensional Data Matrix X
%      : K = Number of clusters/labels
% Output: y = labels corresponding to the data points in X
%       : F = the cluster variance as a vector indexed by iteration
%
% Thomas Anzalone, 11/7/2021

[N,D] = size(X);
label = 1:K;
y = zeros(N,1);

% Generate k random D-Dimensional centroids
mu = randn(K,D);

% F is the sum of the cluster variances
f = zeros(K,1);
F_old = 0;
counter = 0;

% Continuous loop to assign and reassign clusters
while true
    counter = counter+1;
    % Assign labels
    for i = 1:N
        % Calculate the 2 norm of the data point away from all the centroids
        d = vecnorm(X(i,:)-mu,2,2);
        [~,idx] = min(d);
        y(i) = label(idx);
    end
    % Recompute centroids
    for j = 1:K
        idx = find(y == label(j));
        mu(j,:) = (1/length(idx))*sum(X(idx,:));
        f(j) = sum(norm(X(idx,:) - mu(j,:))^2);
    end
    F(counter) = sum(f);
    % Check for convergence
    err = norm(F_old - F(counter));
    if err < 0.1
        fprintf('Iterations to convergence: %i\n',counter)
        break;
    else
        % Timeout counter in case convergence isn't reached
        if counter > 1000
            warning('K-means convergence not reached after 1000 iterations.');
            break;
        end
        F_old = F(counter);
    end
end
end
