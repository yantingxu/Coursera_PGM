function [P G loglikelihood] = LearnGraphAndCPDs(dataset, labels)

% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha) 
% labels: N x 2 true class labels for the examples. labels(i,j)=1 if the 
%         the ith example belongs to class j
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
K = size(labels,2);

G = zeros(10,2,K); % graph structures to learn
% initialization
for k=1:K
    G(2:end,:,k) = ones(9,2);
end

% estimate graph structure for each class
for k=1:K
    % fill in G(:,:,k)
    % use ConvertAtoG to convert a maximum spanning tree to a graph G
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    %%%%%%%%%%%%%%%%%%%%%%%%%
    classIdx = labels(:, k) == 1;
    A = LearnGraphStructure(dataset(classIdx, :, :));
    G(:, :, k) = ConvertAtoG(A);
end

% estimate parameters

P.c = zeros(1,K);
% compute P.c

% the following code can be copied from LearnCPDsGivenGraph.m
% with little or no modification
%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

priorCnt = sum(labels);
P.c(1, :) = priorCnt/sum(priorCnt);

% with parents
[sampleNum, partNum, dimNum] = size(dataset);
P.clg = repmat(struct('theta', [], 'mu_x', [], 'mu_y', [], 'mu_angle', [], 'sigma_x', zeros(1, K), 'sigma_y', zeros(1, K), 'sigma_angle', zeros(1, K)), 1, partNum);

for classIdx = 1:K
    sampleIdx = find(labels(:, classIdx) == 1);
    S = G(:, :, classIdx);
    parentIdx = find(S(:, 1) == 1);
    for partIdx = 1:partNum

        hasParent = ismember(partIdx, parentIdx);
        if hasParent
            parentPartIdx = S(partIdx, 2);
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
