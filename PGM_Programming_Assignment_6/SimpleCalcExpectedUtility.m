% Copyright (C) Daphne Koller, Stanford University, 2012

function EU = SimpleCalcExpectedUtility(I)

  % Inputs: An influence diagram, I (as described in the writeup).
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return Value: the expected utility of I
  % Given a fully instantiated influence diagram with a single utility node and decision node,
  % calculate and return the expected utility.  Note - assumes that the decision rule for the 
  % decision node is fully assigned.

  % In this function, we assume there is only one utility node.
  F = [I.RandomFactors I.DecisionFactors];
  U = I.UtilityFactors(1);
  EU = [];
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE
  %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    factor = F(1);
    for i = 2:length(F)
        factor = FactorProduct(factor, F(i));
    end
    %factor.val = factor.val / sum(factor.val);

    varToReduce = setdiff(factor.var, U.var);
    factor = VariableElimination(factor, varToReduce);

    posMap = zeros(1, length(factor.var));
    for i = 1:length(factor.var)
        posMap(i) = find(U.var == factor.var(i));
    end

    assignments = IndexToAssignment(1:prod(factor.card), factor.card);
    assignments = assignments(:, posMap);
    idx = AssignmentToIndex(assignments, U.card);

    EU = 0.0;
    for i = 1:length(U.val)
        EU += U.val(i)*factor.val(find(idx == i));
    end

    EU = [EU];
end
