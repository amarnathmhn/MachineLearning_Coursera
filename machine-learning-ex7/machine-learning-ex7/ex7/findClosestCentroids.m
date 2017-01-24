function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% loop over all examples
for ex=1:size(X,1)
    xvec = X(ex,:); % row vector
    dist = zeros(K,1); % distance vector to each centroid
    % for each centroid calculate distance to this vector xvec
    for centr=1:K
        dist(centr,1) = norm(xvec - centroids(centr,:)); %centroids(centr) is also a row vector    
    end
    [~, I] = min(dist); % find the index of minimum dist
    idx(ex)= I;
    
end







% =============================================================

end

