%CREATECLUSTERGRAPH Takes in a list of factors and returns a Bethe cluster
%   graph. It also returns an assignment of factors to cliques.
%
%   C = CREATECLUSTERGRAPH(F) Takes a list of factors and creates a Bethe
%   cluster graph with nodes representing single variable clusters and
%   pairwise clusters. The value of the clusters should be initialized to 
%   the initial potential. 
%   It returns a cluster graph that has the following fields:
%   - .clusterList: a list of the cluster beliefs in this graph. These entries
%                   have the following subfields:
%     - .var:  indices of variables in the specified cluster
%     - .card: cardinality of variables in the specified cluster
%     - .val:  the cluster's beliefs about these variables
%   - .edges: A cluster adjacency matrix where edges(i,j)=1 implies clusters i
%             and j share an edge.
%  
%   NOTE: The index of the cluster for each factor should be the same within the
%   clusterList as it is within the initial list of factors.  Thus, the cluster
%   for factor F(i) should be found in P.clusterList(i) 
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function P = CreateClusterGraph(F, Evidence)
P.clusterList = [];
P.edges = [];
for j = 1:length(Evidence),
    if (Evidence(j) > 0),
        F = ObserveEvidence(F, [j, Evidence(j)]);
    end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
inodes = [];
jnodes = [];
position = [];
factorNum = length(F);
P.clusterList = repmat(struct('var', [], 'card', [], 'val', []), 1, factorNum);
for i = 1:factorNum
    factor = F(i);
    P.clusterList(i) = factor;
    %P.clusterList = [P.clusterList factor];
    if length(factor.var) == 2
        inodes = [inodes factor.var(1)];
        jnodes = [jnodes factor.var(2)];
        position = [position i];
    end
end

edges = zeros(factorNum, factorNum);
edgeNum = length(inodes);
for i = 1:edgeNum
    edge = [inodes(i) jnodes(i)];
    pos = position(i);
    for j = 1:factorNum
        if length(F(j).var) == 1 && ismember(F(j).var, edge)
            edges(j, pos) = 1;
            edges(pos, j) = 1;
        end
    end
end
P.edges = edges;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

