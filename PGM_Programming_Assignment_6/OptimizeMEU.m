% Copyright (C) Daphne Koller, Stanford University, 2012

function [MEU OptimalDecisionRule] = OptimizeMEU( I )

  % Inputs: An influence diagram I with a single decision node and a single utility node.
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return value: the maximum expected utility of I and an optimal decision rule 
  % (represented again as a factor) that yields that expected utility.
  
  % We assume I has a single decision node.
  % You may assume that there is a unique optimal decision.
  D = I.DecisionFactors(1);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE...
  % 
  % Some other information that might be useful for some implementations
  % (note that there are multiple ways to implement this):
  % 1.  It is probably easiest to think of two cases - D has parents and D 
  %     has no parents.
  % 2.  You may find the Matlab/Octave function setdiff useful.
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  OD = I.DecisionFactors(1);
  EUF = CalculateExpectedUtilityFactor(I);
  EUF = ReorderFactorVars(EUF, OD.var);

  D.card = EUF.card;
  D.val = zeros(1, prod(D.card));
  if length(EUF.var) == 1
    [v, p] = max(EUF.val, [], 2);
    D.val(p) = 1;
  else
    assignments = IndexToAssignment(1:prod(EUF.card), EUF.card);
    indices = AssignmentToIndex(assignments(:, 2:end), D.card(2:end));
    d = assignments(:, 1);
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
