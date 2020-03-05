function pos_reconstruct = kalle_pca_decode(model,code);
% 

M2 = model.u*model.s*(code');

% Add mean
M = M2 + model.data_mean*ones(1,size(M2,2));

% reshape data 
pos_reconstruct = reshape(M,model.m,model.n,model.k,size(M2,2));
