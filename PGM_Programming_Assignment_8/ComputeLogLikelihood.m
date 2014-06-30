function loglikelihood = ComputeLogLikelihood(P, G, dataset)
% returns the (natural) log-likelihood of data given the model and graph structure
%
% Inputs:
% P: struct array parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description)
%
%    NOTICE that G could be either 10x2 (same graph shared by all classes)
%    or 10x2x2 (each class has its own graph). your code should compute
%    the log-likelihood using the right graph.
%
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% 
% Output:
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset,1); % number of examples
K = length(P.c); % number of classes

loglikelihood = 0;
% You should compute the log likelihood of data as in eq. (12) and (13)
% in the PA description
% Hint: Use lognormpdf instead of log(normpdf) to prevent underflow.
%       You may use log(sum(exp(logProb))) to do addition in the original
%       space, sum(Prob).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[sampleNum, partNum, paramNum] = size(dataset);

sampleProb = zeros(sampleNum, K);
if length(size(G)) > 2
    multiStructure = 1;
else
    multiStructure = 0;
end

for i = 1:sampleNum
    for classIdx = 1:K
        % multiple structure for each class?
        if multiStructure == 0
            S = G;
        else
            S = G(:, :, classIdx)
        end

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

loglikelihood = sum(log(sum(exp(sampleProb), 2)));

end

