function [Beta sigma] = FitLinearGaussianParameters(X, U)

% Estimate parameters of the linear Gaussian model:
% X|U ~ N(Beta(1)*U(1) + ... + Beta(n)*U(n) + Beta(n+1), sigma^2);

% Note that Matlab/Octave index from 1, we can't write Beta(0).
% So Beta(n+1) is essentially Beta(0) in the text book.

% X: (M x 1), the child variable, M examples
% U: (M x N), N parent variables, M examples
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

M = size(U,1);
N = size(U,2);

Beta = zeros(N+1,1);
sigma = 1;

% collect expectations and solve the linear system
% A = [ E[U(1)],      E[U(2)],      ... , E[U(n)],      1     ; 
%       E[U(1)*U(1)], E[U(2)*U(1)], ... , E[U(n)*U(1)], E[U(1)];
%       ...         , ...         , ... , ...         , ...   ;
%       E[U(1)*U(n)], E[U(2)*U(n)], ... , E[U(n)*U(n)], E[U(n)] ]

% construct A
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = zeros(N, N);
for i = 1:N
    for j = 1:N
        Ui = U(:, i);
        Uj = U(:, j);
        A(i, j) = A(j, i) = mean(Ui.*Uj);
    end
end
meanU = mean(U);
A = [meanU; A];
A = [A, [1 meanU]'];


% B = [ E[X]; E[X*U(1)]; ... ; E[X*U(n)] ]

% construct B
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
B = mean(repmat(X, 1, N) .* U)';
B = [mean(X); B];

% solve A*Beta = B
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Beta = A \ B;

% then compute sigma according to eq. (11) in PA description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C = zeros(N, N);
K = repmat(Beta(1:N)', M, 1) .* U;
meanK = mean(K);
for i = 1:N
    C(i, :) = mean(repmat(K(:, i), 1, N) .* K) - meanK(i).*meanK;
end
varX = var(X, 1);
sigma = sqrt(varX - sum(sum(C)));







