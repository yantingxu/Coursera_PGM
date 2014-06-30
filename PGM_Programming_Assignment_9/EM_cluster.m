% File: EM_cluster.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P loglikelihood ClassProb] = EM_cluster(poseData, G, InitialClassProb, maxIter)

% INPUTS
% poseData: N x 10 x 3 matrix, where N is number of poses;
%   poseData(i,:,:) yields the 10x3 matrix for pose i.
% G: graph parameterization as explained in PA8
% InitialClassProb: N x K, initial allocation of the N poses to the K
%   classes. InitialClassProb(i,j) is the probability that example i belongs
%   to class j
% maxIter: max number of iterations to run EM

% OUTPUTS
% P: structure holding the learned parameters as described in the PA
% loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
%   each iteration
% ClassProb: N x K, conditional class probability of the N examples to the
%   K classes in the final iteration. ClassProb(i,j) is the probability that
%   example i belongs to class j

% Initialize variables
N = size(poseData, 1);
K = size(InitialClassProb, 2);

ClassProb = InitialClassProb;

loglikelihood = zeros(maxIter,1);

P.c = [];
P.clg.sigma_x = [];
P.clg.sigma_y = [];
P.clg.sigma_angle = [];


% EM algorithm
for iter=1:maxIter
  
  % M-STEP to estimate parameters for Gaussians
  %
  % Fill in P.c with the estimates for prior class probabilities
  % Fill in P.clg for each body part and each class
  % Make sure to choose the right parameterization based on G(i,1)
  %
  % Hint: This part should be similar to your work from PA8
  
  P.c = zeros(1,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  P.c = sum(ClassProb) ./ sum(sum(ClassProb));
  %P.c

  for classIdx = 1:K
    for partIdx = 1:10
        if G(partIdx, 1) == 0
            % no parents
            if classIdx == 1
                P.clg(partIdx).mu_x = zeros(1, K);
                P.clg(partIdx).sigma_x = zeros(1, K);
                P.clg(partIdx).mu_y = zeros(1, K);
                P.clg(partIdx).sigma_y = zeros(1, K);
                P.clg(partIdx).mu_angle = zeros(1, K);
                P.clg(partIdx).sigma_angle = zeros(1, K);
            end
            for paramIdx = 1:3
                X = squeeze(poseData(:, partIdx, paramIdx));
                W = ClassProb(:, classIdx);
                [mu, sigma] = FitG(X, W);
                if paramIdx == 1
                    P.clg(partIdx).mu_y(classIdx) = mu;
                    P.clg(partIdx).sigma_y(classIdx) = sigma;
                elseif paramIdx == 2
                    P.clg(partIdx).mu_x(classIdx) = mu;
                    P.clg(partIdx).sigma_x(classIdx) = sigma;
                else
                    P.clg(partIdx).mu_angle(classIdx) = mu;
                    P.clg(partIdx).sigma_angle(classIdx) = sigma;
                end
            end
        else
            % have parents
            if classIdx == 1
                P.clg(partIdx).theta = zeros(K, 12);
                P.clg(partIdx).sigma_x = zeros(1, K);
                P.clg(partIdx).sigma_y = zeros(1, K);
                P.clg(partIdx).sigma_angle = zeros(1, K);
            end
            for paramIdx = 1:3
                parentIdx = G(partIdx, 2);
                X = squeeze(poseData(:, partIdx, paramIdx));
                U = squeeze(poseData(:, parentIdx, :));
                W = ClassProb(:, classIdx);
                [beta, sigma] = FitLG(X, U, W);
                if paramIdx == 1
                    P.clg(partIdx).theta(classIdx, 1:4) = [beta(end) beta(1:end-1)'];
                    P.clg(partIdx).sigma_y(classIdx) = sigma;
                elseif paramIdx == 2
                    P.clg(partIdx).theta(classIdx, 5:8) = [beta(end) beta(1:end-1)'];
                    P.clg(partIdx).sigma_x(classIdx) = sigma;
                else
                    P.clg(partIdx).theta(classIdx, 9:12) = [beta(end) beta(1:end-1)'];
                    P.clg(partIdx).sigma_angle(classIdx) = sigma;
                end
            end
        end
    end
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % E-STEP to re-estimate ClassProb using the new parameters
  %
  % Update ClassProb with the new conditional class probabilities.
  % Recall that ClassProb(i,j) is the probability that example i belongs to
  % class j.
  %
  % You should compute everything in log space, and only convert to
  % probability space at the end.
  %
  % Tip: To make things faster, try to reduce the number of calls to
  % lognormpdf, and inline the function (i.e., copy the lognormpdf code
  % into this file)
  %
  % Hint: You should use the logsumexp() function here to do
  % probability normalization in log space to avoid numerical issues
  
  ClassProb = zeros(N,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  logClassProb = zeros(N,K);
  for sampleIdx = 1:N
    sample = squeeze(poseData(sampleIdx, :, :));
    for classIdx = 1:K
        partProb = zeros(1, 10);
        for partIdx = 1:10
            if G(partIdx, 1) == 0
                mu_y = P.clg(partIdx).mu_y(classIdx);
                sigma_y = P.clg(partIdx).sigma_y(classIdx);
                mu_x = P.clg(partIdx).mu_x(classIdx);
                sigma_x = P.clg(partIdx).sigma_x(classIdx);
                mu_angle = P.clg(partIdx).mu_angle(classIdx);
                sigma_angle = P.clg(partIdx).sigma_angle(classIdx);
            else
                parent = squeeze(poseData(sampleIdx, G(partIdx, 2), :));
                mu_y = sum(P.clg(partIdx).theta(classIdx, 1:4) .* [1 parent']);
                sigma_y = P.clg(partIdx).sigma_y(classIdx);
                mu_x = sum(P.clg(partIdx).theta(classIdx, 5:8) .* [1 parent']);
                sigma_x = P.clg(partIdx).sigma_x(classIdx);
                mu_angle = sum(P.clg(partIdx).theta(classIdx, 9:12) .* [1 parent']);
                sigma_angle = P.clg(partIdx).sigma_angle(classIdx);
            end
            triplet = [classIdx partIdx G(partIdx, 2)];
            logpy = lognormpdf(sample(partIdx, 1), mu_y, sigma_y);
            logpx = lognormpdf(sample(partIdx, 2), mu_x, sigma_x);
            logpangle = lognormpdf(sample(partIdx, 3), mu_angle, sigma_angle);
            logp = logpy + logpx + logpangle;
            partProb(partIdx) = logp;
        end
        logClassProb(sampleIdx, classIdx) = sum(partProb) + log(P.c(classIdx));
    end
  end
  %logClassProb(1, :)
  ClassProb = exp(logClassProb - repmat(logsumexp(logClassProb), 1, K));


  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Compute log likelihood of dataset for this iteration
  % Hint: You should use the logsumexp() function here
  loglikelihood(iter) = 0;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  loglikelihood(iter) = sum(logsumexp(logClassProb));
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Print out loglikelihood
  disp(sprintf('EM iteration %d: log likelihood: %f', ...
    iter, loglikelihood(iter)));
  if exist('OCTAVE_VERSION')
    fflush(stdout);
  end
  
  % Check for overfitting: when loglikelihood decreases
  if iter > 1
    if loglikelihood(iter) < loglikelihood(iter-1)
      break;
    end
  end
  
end

% Remove iterations if we exited early
loglikelihood = loglikelihood(1:iter);
