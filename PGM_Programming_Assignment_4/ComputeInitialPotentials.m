%COMPUTEINITIALPOTENTIALS Sets up the cliques in the clique tree that is
%passed in as a parameter.
%
%   P = COMPUTEINITIALPOTENTIALS(C) Takes the clique tree skeleton C which is a
%   struct with three fields:
%   - nodes: cell array representing the cliques in the tree.
%   - edges: represents the adjacency matrix of the tree.
%   - factorList: represents the list of factors that were used to build
%   the tree. 
%   
%   It returns the standard form of a clique tree P that we will use through 
%   the rest of the assigment. P is struct with two fields:
%   - cliqueList: represents an array of cliques with appropriate factors 
%   from factorList assigned to each clique. Where the .val of each clique
%   is initialized to the initial potential of that clique.
%   - edges: represents the adjacency matrix of the tree. 
%
% Copyright (C) Daphne Koller, Stanford University, 2012


function P = ComputeInitialPotentials(C)

% number of cliques
N = length(C.nodes);

% initialize cluster potentials 
P.cliqueList = repmat(struct('var', [], 'card', [], 'val', []), N, 1);
P.edges = zeros(N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% First, compute an assignment of factors from factorList to cliques. 
% Then use that assignment to initialize the cliques in cliqueList to 
% their initial potentials. 

% C.nodes is a list of cliques.
% So in your code, you should start with: P.cliqueList(i).var = C.nodes{i};
% Print out C to get a better understanding of its structure.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
factorNum = length(C.factorList);
assignments = zeros(1, factorNum);
for j = 1:factorNum
    for i = 1:N
        if sum(ismember(C.factorList(j).var, C.nodes{i})) == length(C.factorList(j).var)
            assignments(j) = i;
            break;
        end
    end
end

for i = 1:N
    idx = (assignments == i);
    factors = C.factorList(idx);
    if length(factors) == 0
        result = struct('var', [], 'card', [], 'val', []);
        result.var = C.nodes{i};
        result.card = zeros(1, length(result.var));
        for varIdx = 1:length(result.var)
            var = result.var(varIdx);
            for factorIdx = 1:factorNum
                idx = (C.factorList(factorIdx).var == var);
                if sum(idx) > 0
                    result.card(varIdx) = C.factorList(factorIdx).card(idx);
                    break;
                end
            end
        end
        result.val = ones(1, prod(result.card));
    else
        for j = 1:length(factors)
            if j == 1
                result = factors(1);
            else
                result = FactorProduct(result, factors(j));
            end
        end
    end
    P.cliqueList(i) = ReorderFactorVariables(result);
end
P.edges = C.edges;

end



function out = ReorderFactorVariables(in)   
% Function accepts a factor and reorders the factor variables  
% such that they are in ascending order  

[S, I] = sort(in.var);  

out.var = S;  
out.card = in.card(I);  

allAssignmentsIn = IndexToAssignment(1:prod(in.card), in.card);  
allAssignmentsOut = allAssignmentsIn(:,I); % Map from in assgn to out assgn  
out.val(AssignmentToIndex(allAssignmentsOut, out.card)) = in.val;  

end



