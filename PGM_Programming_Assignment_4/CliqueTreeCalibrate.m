%CLIQUETREECALIBRATE Performs sum-product or max-product algorithm for 
%clique tree calibration.

%   P = CLIQUETREECALIBRATE(P, isMax) calibrates a given clique tree, P 
%   according to the value of isMax flag. If isMax is 1, it uses max-sum
%   message passing, otherwise uses sum-product. This function 
%   returns the clique tree where the .val for each clique in .cliqueList
%   is set to the final calibrated potentials.
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function P = CliqueTreeCalibrate(P, isMax)


% Number of cliques in the tree.
N = length(P.cliqueList);

% Setting up the messages that will be passed.
% MESSAGES(i,j) represents the message going from clique i to clique j. 
MESSAGES = repmat(struct('var', [], 'card', [], 'val', []), N, N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% We have split the coding part for this function in two chunks with
% specific comments. This will make implementation much easier.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% YOUR CODE HERE
% While there are ready cliques to pass messages between, keep passing
% messages. Use GetNextCliques to find cliques to pass messages between.
% Once you have clique i that is ready to send message to clique
% j, compute the message and put it in MESSAGES(i,j).
% Remember that you only need an upward pass and a downward pass.
%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isMax == 1
    cliqueNum = length(P.cliqueList)
    for i = 1:cliqueNum
        P.cliqueList(i).val = log(P.cliqueList(i).val);
    end
end

passTimes = sum(sum(P.edges));
for k = 1:passTimes
    [i, j] = GetNextCliques(P, MESSAGES);
    sentClique = P.cliqueList(i);
    recvClique = P.cliqueList(j);
    nonMsgVar = setdiff(sentClique.var, intersect(sentClique.var, recvClique.var));

    recvMsgs = MESSAGES(:, i);
    recvEdges = P.edges(:, i);
    result = sentClique;
    for l = 1:N
        if recvEdges(l) == 1 && l ~= j && length(recvMsgs(l).var) > 0
            if isMax == 0
                result = FactorProduct(result, recvMsgs(l));
            else
                result = FactorSum(result, recvMsgs(l));
            end
        end
    end
    if isMax == 0
        msg = FactorMarginalization(result, nonMsgVar);
        msg.val = msg.val / sum(msg.val);
    else
        msg = FactorMaxMarginalization(result, nonMsgVar);
    end
    MESSAGES(i, j) = msg;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% Now the clique tree has been calibrated. 
% Compute the final potentials for the cliques and place them in P.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:N
    neighbours = P.edges(i, :);
    result = P.cliqueList(i);
    for j = 1:N
        if neighbours(j) == 1
            if isMax == 0
                result = FactorProduct(result, MESSAGES(j, i));
            else
                result = FactorSum(result, MESSAGES(j, i));
            end
        end
    end
    P.cliqueList(i) = result;
end

return
