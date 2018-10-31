function [loss] = loss(Y, Z, d)
% loss of matrix Y with respect to matrix Z
% Y and Z are of the same size

Yhat = optApprox(Y, d);
loss =  re(Y, Z) - re (Y, Yhat);

end

