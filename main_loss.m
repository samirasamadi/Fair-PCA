clc;

[M, A, B] = creditProcess();

featureNum = 21;

coeff = pca(M);
coeff_A = pca(A);
coeff_B = pca(B);

loss_A = zeros(featureNum,1);
loss_B = zeros(featureNum,1);

z_last = zeros(featureNum, 1);
z = zeros(featureNum, 1);
lossFair_max = zeros(featureNum, 1);

lossFair_A = zeros(featureNum,1);
lossFair_B = zeros(featureNum,1);

% parameters of the mw algorithm
eta = 1;
T = 10; 


for ell=1:featureNum
    
    P = coeff(:,1:ell)*transpose(coeff(:,1:ell));
    
    approx_A = A*P;
    approx_B = B*P;
    
    % vanilla PCA's average loss on popultion A and B
    loss_A(ell) = loss(A, approx_A, ell)/size(A, 1);
    loss_B(ell) = loss(B, approx_B, ell)/size(B, 1);
    
    
    [P_fair,z(ell),P_last,z_last(ell)] = mw(A, B, ell,eta ,T);
    
    if z(ell) < z_last(ell)
        P_smart = P_fair;
    else
        P_smart = P_last;
    end
    
    P_smart = eye(size(P_smart,1)) - sqrtm(eye(size(P_smart,1))-P_smart);
    
    approxFair_A = A*P_smart;
    approxFair_B = B*P_smart;
    
    lossFair_A = loss(A, approxFair_A, ell)/size(A, 1);
    lossFair_B = loss(B, approxFair_B, ell)/size(B, 1);
    lossFair_max(ell) = max([lossFair_A, lossFair_B]);
    
end
    
checkpoints = [1:featureNum];
plot(checkpoints, loss_A,'gx-',checkpoints,loss_B,'ro-',checkpoints, lossFair_max,'b--o');
title('Average loss of A versus B for PCA and fair-PCA')
legend('A','B','fair loss')
xlabel('Number of dimensions')
ylabel('Loss')

writetable(table(loss_A,loss_B,lossFair_max),'loss.txt','Delimiter','\t')
