function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

C_sig_vec = [0.01,0.03,0.1,0.3,1,3,10,30];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

error_matrix = size(length(C_sig_vec),length(C_sig_vec));

for i = 1:length(C_sig_vec)

	for j = 1:length(C_sig_vec)

		% 1st Step is to get the model using X,y

			model= svmTrain(X,y,C_sig_vec(i), @(x1, x2) gaussianKernel(x1, x2, C_sig_vec(j))); 

		% 2nd Step is to compute the prediction on the cross validation set

			predictions = svmPredict(model,Xval);
			
		% 3rd Step is to compute the prediction error and record the lowest (store the i,j of C_sig_vec)
			
			error = mean(double(predictions ~=yval));
			
			%error_matrix(i,j) = error;
			
			if(i == 1 && j == 1)
				lowest_error = error +1;
			end

			if (error < lowest_error)
				Index_i = i;
				Index_j = j;
				
				lowest_error = error;
			end

			
			
			
	
	end

end

C = C_sig_vec(Index_i);
sigma = C_sig_vec(Index_j);



% =========================================================================

end
