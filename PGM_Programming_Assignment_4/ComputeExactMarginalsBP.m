%COMPUTEEXACTMARGINALSBP Runs exact inference and returns the marginals
%over all the variables (if isMax == 0) or the max-marginals (if isMax == 1). 
%
%   M = COMPUTEEXACTMARGINALSBP(F, E, isMax) takes a list of factors F,
%   evidence E, and a flag isMax, runs exact inference and returns the
%   final marginals for the variables in the network. If isMax is 1, then
%   it runs exact MAP inference, otherwise exact inference (sum-prod).
%   It returns an array of size equal to the number of variables in the 
%   network where M(i) represents the ith variable and M(i).val represents 
%   the marginals of the ith variable. 
%
% Copyright (C) Daphne Koller, Stanford University, 2012


function M = ComputeExactMarginalsBP(F, E, isMax)

% initialization
% you should set it to the correct value in your code
M = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% Implement Exact and MAP Inference.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

P = CreateCliqueTree(F, E);
P = CliqueTreeCalibrate(P, isMax);

variables = [];
factorNum = length(F);
for i = 1:factorNum
    variables = union(variables, F(i).var);
end
variables = sort(variables);
totalVarNum = length(variables);

M = repmat(struct('var', [], 'card', [], 'val', []), totalVarNum, 1);

cliqueNum = length(P.cliqueList);
for varIdx = 1:totalVarNum
    var = variables(varIdx);
    for i = 1:cliqueNum
        clique = P.cliqueList(i);
        if ismember(var, clique.var)
            if isMax == 0
                factor = FactorMarginalization(clique, setdiff(clique.var, var), E);
                factor.val = factor.val / sum(factor.val);
            else
                factor = FactorMaxMarginalization(clique, setdiff(clique.var, var), E);
            end
            M(varIdx) = factor;
        end
    end
end

end



