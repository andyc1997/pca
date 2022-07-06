% Remark:
% Cross-check the implementation of SPCABlockl0 in python
% with GPower.m from authors website:
% http://www.inma.ucl.ac.be/~richtarik 
% http://www.montefiore.ulg.ac.be/~journee

%% read iris data
X_tr = readtable('.\iris.csv');
X_tr.Var1 = [];
X_tr.Species = [];

fprintf('Shape of training data: (%d, %d)\n', size(X_tr))
fprintf('Data type: %s\n', class(X_tr))

%% preprocessing
X_tr = table2array(normalize(X_tr));
fprintf('Data type: %s\n', class(X_tr))

K = X_tr'*X_tr / (150-1);
fprintf('Shape of Gram data: (%d, %d)\n', size(K))
fprintf('Data type: %s\n', class(K))

%% debug
[U, S, V] = svd(K);
A = (V*sqrt(S))';
rho = [0.1, 0.5, 0.8];
mu = [1, 1, 1];
Z = GPower(A, rho, 3, 'l0', 1, mu);

    