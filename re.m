function [reVal] = re(Y,Z)
% Calculate the reconstruction error of matrix Y with respect to matrix Z
% Matrix Y and Z are of the same size
reVal = norm(Y-Z, 'fro')^2;
end

