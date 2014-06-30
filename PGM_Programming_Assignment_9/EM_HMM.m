% File: EM_HMM.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P loglikelihood ClassProb PairProb] = EM_HMM(actionData, poseData, G, InitialClassProb, InitialPairProb, maxIter)

% INPUTS
% actionData: structure holding the actions as described in the PA
% poseData: N x 10 x 3 matrix, where N is number of poses in all actions
% G: graph parameterization as explained in PA description
% InitialClassProb: N x K matrix, initial allocation of the N poses to the K
%   states. InitialClassProb(i,j) is the probability that example i belongs
%   to state j.
%   This is described in more detail in the PA.
% InitialPairProb: V x K^2 matrix, where V is the total number of pose
%   transitions in all HMM action models, and K is the number of states.
%   This is described in more detail in the PA.
% maxIter: max number of iterations to run EM

% OUTPUTS
% P: structure holding the learned parameters as described in the PA
% loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
%   each iteration
% ClassProb: N x K matrix of the conditional class probability of the N examples to the
%   K states in the final iteration. ClassProb(i,j) is the probability that
%   example i belongs to state j. This is described in more detail in the PA.
% PairProb: V x K^2 matrix, where V is the total number of pose transitions
%   in all HMM action models, and K is the number of states. This is
%   described in more detail in the PA.

% Initialize variables
N = size(poseData, 1);
K = size(InitialClassProb, 2);
L = size(actionData, 2); % number of actions
V = size(InitialPairProb, 1);

ClassProb = InitialClassProb;
PairProb = InitialPairProb;

loglikelihood = zeros(maxIter,1);

P.c = [];
P.clg.sigma_x = [];
P.clg.sigma_y = [];
P.clg.sigma_angle = [];

% EM algorithm
for iter=1:maxIter
  
  % M-STEP to estimate parameters for Gaussians
  % Fill in P.c, the initial state prior probability (NOT the class probability as in PA8 and EM_cluster.m)
  % Fill in P.clg for each body part and each class
  % Make sure to choose the right parameterization based on G(i,1)
  % Hint: This part should be similar to your work from PA8 and EM_cluster.m
  
  P.c = zeros(1,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
  firstStates = [];
  for i = 1:L
    firstStates = [firstStates actionData(i).marg_ind(1)];
  end
  P.c = sum(ClassProb(firstStates, :)) ./ sum(sum(ClassProb(firstStates, :)));

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
  
  % M-STEP to estimate parameters for transition matrix
  % Fill in P.transMatrix, the transition matrix for states
  % P.transMatrix(i,j) is the probability of transitioning from state i to state j
  P.transMatrix = zeros(K,K);
  
  % Add Dirichlet prior based on size of poseData to avoid 0 probabilities
  P.transMatrix = P.transMatrix + size(PairProb, 1) * .05;
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  vectorProb = sum(PairProb, 1);
  transMatrix = reshape(vectorProb, K, K) + P.transMatrix;
  P.transMatrix = transMatrix ./ repmat(sum(transMatrix, 2), 1, K);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
    
  % E-STEP preparation: compute the emission model factors (emission probabilities) in log space for each 
  % of the poses in all actions = log( P(Pose | State) )
  % Hint: This part should be similar to (but NOT the same as) your code in EM_cluster.m
  
  logEmissionProb = zeros(N,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
        logEmissionProb(sampleIdx, classIdx) = sum(partProb);
    end
  end

  %logEmissionProb(1:5, :)

  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
    
  % E-STEP to compute expected sufficient statistics
  % ClassProb contains the conditional class probabilities for each pose in all actions
  % PairProb contains the expected sufficient statistics for the transition CPDs (pairwise transition probabilities)
  % Also compute log likelihood of dataset for this iteration
  % You should do inference and compute everything in log space, only converting to probability space at the end
  % Hint: You should use the logsumexp() function here to do probability normalization in log space to avoid numerical issues
  
  ClassProb = zeros(N,K);
  PairProb = zeros(V,K^2);
  loglikelihood(iter) = 0;
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ll = 0.0;
  logTrans = log(reshape(P.transMatrix', 1, K*K));
  for actionIndx = 1:L
    poses = actionData(actionIndx).marg_ind;
    edges = actionData(actionIndx).pair_ind;
    factors = repmat(struct('var', [], 'card', [], 'val', []), 1, 1+length(poses)+length(edges));
    factorIdx = 1;

    for poseIdx = 1:length(poses)
        pose = poses(poseIdx);
        probs = logEmissionProb(pose, :);
        factor = struct('var', [poseIdx], 'card', [K], 'val', probs);
        factors(factorIdx) = factor;
        factorIdx += 1;
        if poseIdx > 1
            factor = struct('var', [poseIdx poseIdx-1], 'card', [K K], 'val', logTrans);
            factors(factorIdx) = factor;
            factorIdx += 1;
        end
    end
    factor = struct('var', [1], 'card', [K], 'val', log(P.c));
    factors(factorIdx) = factor;

    %for i = 1:length(factors)
    %    factors(i)
    %end
    %factorIdx
    %length(poses)+length(edges)

    [M, T] = ComputeExactMarginalsHMM(factors);

    sampleVar = T.cliqueList(2);
    ll = ll + logsumexp(sampleVar.val);

    for i = 1:length(M)
        pIdx = M(i).var(1);
        ClassProb(poses(pIdx), :) = ClassProb(poses(pIdx), :) + exp(M(i).val);
    end

    for i = 1:length(T.cliqueList)
        if length(T.cliqueList(i).var) > 1
            destVar = T.cliqueList(i).var(1);
            vIdx = edges(destVar);
            PairProb(vIdx, :) = exp(T.cliqueList(i).val - logsumexp(logsumexp(T.cliqueList(i).val)'));
        end
    end
  end

  ClassProb = ClassProb ./ repmat(sum(ClassProb, 2), 1, K);
  loglikelihood(iter) = ll;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Print out loglikelihood
  disp(sprintf('EM iteration %d: log likelihood: %f', ...
    iter, loglikelihood(iter)));
  if exist('OCTAVE_VERSION')
    fflush(stdout);
  end
  
  % Check for overfitting by decreasing loglikelihood
  if iter > 1
    if loglikelihood(iter) < loglikelihood(iter-1)
      break;
    end
  end
  
end

% Remove iterations if we exited early
loglikelihood = loglikelihood(1:iter);
end
