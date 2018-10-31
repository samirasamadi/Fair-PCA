function [Mhat] = optApprox(M, d)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

coeff = pca(M, 'NumComponents', d);
P = coeff * transpose(coeff);
Mhat = M*P;

end

