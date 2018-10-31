clc;

% substitute LFWprocess with any other function that preprocesses your data
% and outputs three centered matrices M, A, B

[M, A, B] = LFWProcess();
featureNum = 5;

coeff = pca(M);
coeff_A = pca(A);
coeff_B = pca(B);

%Vanilla PCA reconstrction error
recons_A = zeros(featureNum,1);
recons_B = zeros(featureNum,1);

% fair reconstruction error
reconsFair_A = zeros(featureNum, 1);
reconsFair_B = zeros(featureNum, 1);

% reconstruction error with respect to the optimal approximation matrix in target rank
reconsAhat = zeros(featureNum, 1);
reconsBhat = zeros(featureNum, 1);

% objective value
z = zeros(featureNum, 1);
z_smart = zeros(featureNum, 1);

% parameters of the mw algorithm
eta = 20;
T = 5; 

for ell=1:featureNum

    % projection matrix given by vanilla PCA on M
    P = coeff(:,1:ell)*transpose(coeff(:,1:ell));
    
    approx_A = A*P;
    approx_B = B*P;
    
    recons_A(ell) = re(A, approx_A)/size(A, 1);
    recons_B(ell) = re(B, approx_B)/size(B, 1);
    
   
    Ahat = optApprox(A, ell);
    reconsAhat(ell) = re(A, Ahat)/size(A, 1);
    
    Bhat = optApprox(B, ell);
    reconsBhat(ell) = re(B, Bhat)/size(B, 1);
    
    %Fair PCA part
    [P_fair, z(ell), P_last, z_last(ell)] = mw(A, B, ell, eta/ell, T);
    if z(ell) < z_last(ell)
        P_smart = P_fair;
    else
        P_smart = P_last;
    end
    P_smart = eye(size(P_smart,1)) - sqrtm(eye(size(P_smart,1))-P_smart);
    %just done with P smart as my fair PCA solution with equal loss
    
    approxFair_A = A * P_smart;
    approxFair_B = B * P_smart;
    
    reconsFair_A(ell) = re(approxFair_A, A)/size(A, 1);
    reconsFair_B(ell) = re(approxFair_B, B)/size(B, 1);
    
end


%z_last
writetable(table(recons_A,recons_B,reconsFair_A,reconsFair_B,reconsAhat,reconsBhat),'rec.txt','Delimiter','\t')

checkpoints = [1:20];
plot(checkpoints, recons_A,'rx-', checkpoints, recons_B, 'bx-', checkpoints, reconsFair_A,'r*-', checkpoints, reconsFair_B,'b*-');

title('Average reconstruction error of A versus B for vanilla PCA and fair-PCA')
legend('ARE A PCA','ARE B PCA', 'ARE A fair PCA', 'ARE B fair PCA')
xlabel('Number of features')
ylabel('Average reconstruction error (ARE)')



    


    