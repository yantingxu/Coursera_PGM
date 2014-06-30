function accuracy = ClassifyDataset(dataset, labels, P, G)
% returns the accuracy of the model P and graph G on the dataset 
%
% Inputs:
% dataset: N x 10 x 3, N test instances represented by 10 parts
% labels:  N x 2 true class labels for the instances.
%          labels(i,j)=1 if the ith instance belongs to class j 
% P: struct array model parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description) 
%
% Outputs:
% accuracy: fraction of correctly classified instances (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
accuracy = 0.0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K = size(labels, 2);
[sampleNum, partNum, paramNum] = size(dataset);
sampleProb = zeros(sampleNum, K);
S = G;

for i = 1:sampleNum
    for classIdx = 1:K
        for j = 1:partNum
            y = dataset(i, j, 1);
            x = dataset(i, j, 2);
            angle = dataset(i, j, 3);

            % calc mu and sigma for linear Gaussian
            if S(j, 1) == 0
                mu_x = P.clg(j).mu_x(classIdx);
                mu_y = P.clg(j).mu_y(classIdx);
                mu_angle = P.clg(j).mu_angle(classIdx);
                sigma_x = P.clg(j).sigma_x(classIdx);
                sigma_y = P.clg(j).sigma_y(classIdx);
                sigma_angle = P.clg(j).sigma_angle(classIdx);
            else
                py = dataset(i, S(j, 2), 1);
                px = dataset(i, S(j, 2), 2);
                pangle = dataset(i, S(j, 2), 3);
                parent = [py px pangle];

                theta = P.clg(j).theta(classIdx, :);
                mu_y = sum(theta(1:4) .* [1 parent]);
                mu_x = sum(theta(5:8) .* [1 parent]);
                mu_angle = sum(theta(9:12) .* [1 parent]);
                sigma_x = P.clg(j).sigma_x(classIdx);
                sigma_y = P.clg(j).sigma_y(classIdx);
                sigma_angle = P.clg(j).sigma_angle(classIdx);
            end


            % get probability of this sample
            prob_y = lognormpdf(y, mu_y, sigma_y);
            prob_x = lognormpdf(x, mu_x, sigma_x);
            prob_angle = lognormpdf(angle, mu_angle, sigma_angle);
            sampleProb(i, classIdx) += (prob_x + prob_y + prob_angle);

        end
    end
    sampleProb(i, :) +=  log(P.c);
end

predictions = sampleProb(:, 1) > sampleProb(:, 2);
actuals = labels(:, 1);
accuracy = sum(predictions == actuals) / length(predictions)

fprintf('Accuracy: %.2f\n', accuracy);









