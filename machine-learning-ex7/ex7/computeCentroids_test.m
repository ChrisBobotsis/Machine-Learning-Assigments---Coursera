function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

C_matrix = zeros(K,n+1);	% Matrix that has K rows (for each centroid classifier) and each column holds the cumulative sum of the data points belonging to said centroid except the last which holds the number of points as part of said centroid

for i = 1:n

	for j = 1:m
		
		C_matrix(idx(j),i) += X(j,i);		%idx(j) will give back the centroid classifier (1 to K)
		
		C_matrix(idx(j),n+1) += 1;			%increment the counter (0.5 because I am running through the columns twice)
		
	end
	
end

% Now we process the data in the matrix


centroids = 2*C_matrix(:,1:n)./C_matrix(:,n+1);




% =============================================================


end

