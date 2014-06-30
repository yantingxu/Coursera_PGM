% Copyright (C) Daphne Koller, Stanford University, 2012

function [MEU OptimalDecisionRule] = OptimizeLinearExpectations( I )
  % Inputs: An influence diagram I with a single decision node and one or more utility nodes.
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return value: the maximum expected utility of I and an optimal decision rule 
  % (represented again as a factor) that yields that expected utility.
  % You may assume that there is a unique optimal decision.
  %
  % This is similar to OptimizeMEU except that we will have to account for
  % multiple utility factors.  We will do this by calculating the expected
  % utility factors and combining them, then optimizing with respect to that
  % combined expected utility factor.  
  MEU = [];
  OptimalDecisionRule = [];
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE
  %
  % A decision rule for D assigns, for each joint assignment to D's parents, 
  % probability 1 to the best option from the EUF for that joint assignment 
  % to D's parents, and 0 otherwise.  Note that when D has no parents, it is
  % a degenerate case we can handle separately for convenience.
  %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  OD = I.DecisionFactors(1);
  Us = I.UtilityFactors;
  I.UtilityFactors = Us(1);
  eEUF = CalculateExpectedUtilityFactor(I);
  for i = 2:length(Us)
    I.UtilityFactors = Us(i);
    EUF = CalculateExpectedUtilityFactor(I);
    eEUF = FactorSum(eEUF, EUF);
  end
  EUF = ReorderFactorVars(eEUF, OD.var);

  D = I.DecisionFactors(1);
  D.card = EUF.card;
  D.val = zeros(1, prod(D.card));
  if length(EUF.var) == 1
    [v, p] = max(EUF.val, [], 2);
    D.val(p) = 1;
  else
    assignments = IndexToAssignment(1:prod(EUF.card), EUF.card);
    indices = AssignmentToIndex(assignments(:, 2:end), D.card(2:end));
    d = assignments(:, 1);
    %[d indices EUF.val']
    values = zeros(prod(D.card(2:end)), max(d));
    for i = 1:length(d)
        di = d(i);
        indi = indices(i);
        val = EUF.val(i);
        values(indi, di) = val;
    end
    [v, p] = max(values, [], 2);
    for i = 1:length(p)
        as = [p(i) IndexToAssignment(i, D.card(2:end))];
        pos = AssignmentToIndex(as, D.card);
        D.val(pos) = 1;
    end
  end
  OptimalDecisionRule = ReorderFactorVars(D, sort(OD.var));
  %OptimalDecisionRule = D;

  factor = FactorProduct(OptimalDecisionRule, EUF);
  MEU = sum(factor.val);

end
