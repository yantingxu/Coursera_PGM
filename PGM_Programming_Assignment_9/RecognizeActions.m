% File: RecognizeActions.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [accuracy, predicted_labels] = RecognizeActions(datasetTrain, datasetTest, G, maxIter)

% INPUTS
% datasetTrain: dataset for training models, see PA for details
% datasetTest: dataset for testing models, see PA for details
% G: graph parameterization as explained in PA decription
% maxIter: max number of iterations to run for EM

% OUTPUTS
% accuracy: recognition accuracy, defined as (#correctly classified examples / #total examples)
% predicted_labels: N x 1 vector with the predicted labels for each of the instances in datasetTest, with N being the number of unknown test instances


% Train a model for each action
% Note that all actions share the same graph parameterization and number of max iterations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
actionClassNum = length(datasetTrain);
Ps = [];
for i = 1:actionClassNum
    data = datasetTrain(i);
    [P, ll, cp, pp] = EM_HMM(data.actionData, data.poseData, G, data.InitialClassProb, data.InitialPairProb, maxIter);
    Ps = [Ps P];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Classify each of the instances in datasetTrain
% Compute and return the predicted labels and accuracy
% Accuracy is defined as (#correctly classified examples / #total examples)
% Note that all actions share the same graph parameterization

accuracy = 0;
predicted_labels = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
caseNum = length(datasetTest.labels);
evalProb = zeros(caseNum, actionClassNum);

poseData = datasetTest.poseData;
N = size(poseData, 1);
for j = 1:actionClassNum
    P = Ps(j);
    K = length(P.c);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    logEmissionProb = zeros(N,K);

    % YOUR CODE HERE
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
    logTrans = log(reshape(P.transMatrix', 1, K*K));
    for i = 1:caseNum
        actionData = datasetTest.actionData(i);
        poses = actionData.marg_ind;
        edges = actionData.pair_ind;
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
        [M, T] = ComputeExactMarginalsHMM(factors);
        sampleVar = T.cliqueList(2);
        ll = logsumexp(sampleVar.val);
        evalProb(i, j) = ll;
    end
end
[maxProb, predicted_labels] = max(evalProb, [], 2);
accuracy = sum(datasetTest.labels == predicted_labels) / length(predicted_labels);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







