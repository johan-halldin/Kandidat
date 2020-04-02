function model = kalle_make_pca(pos,K);
% 

if nargin<2,
    K = 20; % assume 20 principal components if K is not set. 
end

% Calculate sizes
[m,n,k,N]=size(pos);

% Assuming that k = 1, reshape data into a matrix
% where each column is one image (column-stacked)
M = reshape(pos(:),m*n,N);

% Calculate mean of images
data_mean = mean(M,2);

% Remove mean
M2 = M - data_mean*ones(1,size(M,2));

% Calculate svd,
[u,s,v]=svd(M2,0);

% save relevant data in model
model.m = m;
model.n = n; 
model.k = k;
model.N = N;
model.data_mean = data_mean;
model.u = u(:,1:K);  % The columns of u are the modes
model.s = s(1:K,1:K); % The diagonal elemenst of s are the standard deviation for each mode
model.v = v(:,1:K); % This is not really needed. 


