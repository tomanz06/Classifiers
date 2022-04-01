function varargout = plot_spectral_embedding(x,k,kernel,varargin)
% This script implements the spectral embedding on data matrix x
% Input: x = D-Dimensional Data Matrix x
%      : k = number of non-zero eigenvectors of the embedding
%      : kernel = string that defines the type of Kernel to embed the data
%      : varargin = any set of parameters necessary for the given Kernel
%           : sigma = Variance parameter of Gaussian Kernel (default = 1)
%           : c,d = constant, order of Polynomial Kernel (default = 0,1)
%           : a,r = coeff, constant of Sigmoid Kernel (default = 1,0)
%
% Output: varargout = set of outputs to store if desired
%           : L = the Laplacian Matrix
%           : V = the spectral embedding of the data
%
% Thomas Anzalone, 11/22/2021

% Initialize
n = length(x);
K = zeros(n,n);
D = zeros(n,n);
for j = 1:n
    for i = 1:n
        % Compute the Kernel at each index
        switch kernel
            case 'Gaussian'
                if(isempty(varargin)), sigma = 1;
                else, sigma = varargin{1}; end
                K(i,j) = exp(-(norm(x(i,:)-x(j,:)).^2)/2/sigma^2);
            case 'Polynomial'
                if(isempty(varargin)), c = 0; d = 1;
                else, c = varargin{1}; d = varargin{2}; end
                K(i,j) = (x(i,:)*x(j,:)'+c).^d;
            case 'Sigmoid'
                if(isempty(varargin)), a = 1; r = 0;
                else, a = varargin{1}; r = varargin{2}; end
                K(i,j) = tanh(a*x(i,:)*x(j,:)'+r);
            otherwise
                error('Enter a valid Kernel function.')
        end
    end
    % Compute the diagonal entry of the Degree matrix
    D(j,j) = sum(K(:,j));
end
% Compute the Laplacian
L = D - K;
% Find the Eigenvectors of the Laplacian
[U,~] = eig(L);
% Use the first k non-zero eigenvectors of the Laplacian 
V = U(:,2:(k+1));

labels = kmeans(V,2);

h_fig(1) = figure('Name','K-means of Spectral Embedding');
h_ax(1) = axes(); box on; grid on; hold on;
h_fig(2) = figure('Name','Clustered Data');
h_ax(2) = axes(); box on; grid on; hold on;

% Plot both the spectral embedding and the labeled data
for i = unique(labels)'
    ind = (labels == i);
    figure(h_fig(1));
    scatter(V(ind,1),zeros(length(ind(ind)),1),'DisplayName',sprintf('Cluster %i',i));
    figure(h_fig(2));
    scatter(x(ind,1),x(ind,2),'DisplayName',sprintf('Cluster %i',i));
end
figure(h_fig(1)); legend('show');
title('K-means of Spectral Embedding 2nd Eigenvector');
xlabel('x_1');
figure(h_fig(2)); legend('show');
title('Clustered Data');
xlabel('x_1'); ylabel('x_2');

if(nargout > 0), varargout{1} = L; varargout{2} = V; end

end
