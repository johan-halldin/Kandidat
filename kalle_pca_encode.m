function code = kalle_pca_encode(model,pos);
% 

% Calculate sizes
[m,n,k,N]=size(pos);

% Assuming that k = 1, reshape data into a matrix
% where each column is one image (column-stacked)
M = reshape(pos(:),m*n,N);

% Remove mean
M2 = M - model.data_mean*ones(1,size(M,2));

% Calculate projection
code = (inv(model.s)*model.u'*M2)';
