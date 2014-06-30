function [P loglikelihood] = LearnCPDsGivenGraph(dataset, G, labels)
%
% Inputs:
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% G: graph parameterization as explained in PA description
% labels: N x 2 true class labels for the examples. labels(i,j)=1 if the 
%         the ith example belongs to class j and 0 elsewhere        
%
% Outputs:
% P: struct array parameters (explained in PA description)
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
K = size(labels,2);

loglikelihood = 0;
P.c = zeros(1,K);

% estimate parameters
% fill in P.c, MLE for class probabilities
% fill in P.clg for each body part and each class
% choose the right parameterization based on G(i,1)
% compute the likelihood - you may want to use ComputeLogLikelihood.m
% you just implemented.
%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
priorCnt = sum(labels);
P.c(1, :) = priorCnt/sum(priorCnt);

% with parents
parentIdx = find(G(:, 1) == 1);
[sampleNum, partNum, dimNum] = size(dataset);
P.clg = repmat(struct('theta', [], 'mu_x', [], 'mu_y', [], 'mu_angle', [], 'sigma_x', zeros(1, K), 'sigma_y', zeros(1, K), 'sigma_angle', zeros(1, K)), 1, partNum);
%P.clg = repmat(struct('theta', zeros(K, dimNum*(dimNum+1)), 'mu_x', zeros(K, 1), 'mu_y', zeros(K, 1), 'mu_angle', zeros(K, 1), 'sigma_x', zeros(K, 1), 'sigma_y', zeros(K, 1), 'sigma_angle', zeros(K, 1)), 1, partNum);

for classIdx = 1:K
    sampleIdx = find(labels(:, classIdx) == 1);
    for partIdx = 1:partNum

        hasParent = ismember(partIdx, parentIdx);
        if hasParent
            parentPartIdx = G(partIdx, 2);
            U = squeeze(dataset(sampleIdx, parentPartIdx, :));
            if classIdx == 1
                P.clg(partIdx).theta = zeros(K, dimNum*(dimNum+1));
            end
        else
            if classIdx == 1
                P.clg(partIdx).mu_x = zeros(1, K);
                P.clg(partIdx).mu_y = zeros(1, K);
                P.clg(partIdx).mu_angle = zeros(1, K);
            end
        end

        for dimIdx = 1:dimNum
            if hasParent
                X = squeeze(dataset(sampleIdx, partIdx, dimIdx));
                [beta, sigma] = FitLinearGaussianParameters(X, U);
                if dimIdx == 1
                    beginIdx = 1;
                    endIdx = 4;
                    P.clg(partIdx).sigma_y(classIdx) = sigma;
                elseif dimIdx == 2
                    beginIdx = 5;
                    endIdx = 8;
                    P.clg(partIdx).sigma_x(classIdx) = sigma;
                else
                    beginIdx = 9;
                    endIdx = 12;
                    P.clg(partIdx).sigma_angle(classIdx) = sigma;
                end
                beta = beta';
                P.clg(partIdx).theta(classIdx, beginIdx:endIdx) = [beta(end) beta(1:end-1)];
            else
                [mu, sigma] = FitGaussianParameters(squeeze(dataset(sampleIdx, partIdx, dimIdx)));
                if dimIdx == 1
                    P.clg(partIdx).mu_y(classIdx) = mu;
                    P.clg(partIdx).sigma_y(classIdx) = sigma;
                elseif dimIdx == 2
                    P.clg(partIdx).mu_x(classIdx) = mu;
                    P.clg(partIdx).sigma_x(classIdx) = sigma;
                else
                    P.clg(partIdx).mu_angle(classIdx) = mu;
                    P.clg(partIdx).sigma_angle(classIdx) = sigma;
                end
            end
        end
    end
end

loglikelihood = ComputeLogLikelihood(P, G, dataset);

fprintf('log likelihood: %f\n', loglikelihood);



