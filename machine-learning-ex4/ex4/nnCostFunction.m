function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

K = size(Theta2,1);		% Get size of theta2 to determine number of classifiers

log_h_theta = zeros(length(y),1); % this will have in each row a classifier number to check with y

temp = zeros(K,1);

temp_y = zeros(K,1);


%%%%%%%%%%%%% This section calculates the cost function J(theta) %%%%%%%%%%%%%%%%%		WORKS!
for j = 1:m
	
	temp = sigmoid(Theta2*[1;(sigmoid(Theta1*([1,X(j,:)]')))]);		% Kx1 column vector with all probabilities
	
	temp_y(y(j)) = 1;			% Kx1 vector with all zeros except 1 at index of classifier for his example
	
	J +=  -temp_y'*log(temp)-(1-temp_y)'*log(1-temp);		% Calculates cumlative cost
	
	temp_y = zeros(K,1);		% resets temp_y to be zero vector
	
		
end

J /= m;

%%%%%%%%%%%% End of Cost function calculation %%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%% Adding in regularization to cost function %%%%%%%%%%%%				WORKS!

J += (lambda/(2*m))*(sum(sum((Theta1(:,2:end).^2))) + sum(sum((Theta2(:,2:end).^2))));

%%%%%%%%%%%% End of regularization of cost function %%%%%%%%%%%%%%%%


%%%%%%%%%%%% Beginning of Gradient Computation using backpropagation %%%%%%%%%%%%%%%%			HAVEN'T TESTED THIS YET!!!!

% First do a forward pass to compute activations

d_2 = zeros(size(Theta1,1)+1,1);	%	26x1	Lower case delta vector for second (i.e hidden layer)

d_3 = zeros(size(Theta2,1),1);		%	10x1	Lower case delta vector for third (i.e. output layer)

z_2 =  zeros(size(Theta1,1),1);		% Z vector for layer 2

z_3 = zeros(size(Theta2,1),1);		% Z vector for layer 3

a_1 = zeros(size(X,2)+1,1);			% Adding one to row to account for bias unit

a_2 = zeros(size(Theta1,1)+1,1);	% Adding one to row to account for bias unit

a_3 = zeros(size(Theta2,1),1);		% Output layer


D_2 = zeros(size(d_3,1),size(a_2,1));	%	10x26	

D_1 = zeros(size(d_2,1)-1,size(a_1,1));	%	25x401	subtract one from d_2 due to bias unit


for i = 1:m

y_temp = zeros(size(Theta2,1),1);

	% Compute forward propogation
	
	a_1 = [1;X(i,:)'];		%	401x1		 Only use ith example for a_1 and add the bias unit
	
	z_2 = Theta1*a_1;		%	25x1		 vector of z values for 2nd layer
	
	a_2 = [1;sigmoid(z_2)];	%	26x1
	
	z_3 = Theta2*a_2;		%	10x1
	
	a_3 = sigmoid(z_3);		%	10x1
	
	% Compute back propogation
	
	y_temp(y(i)) = 1;		%	10x1
	
	
	d_3 = a_3 - y_temp;		%	10x1
	
	%size(Theta2'*d_3)
	%pause;
	
	d_2 = (Theta2'*d_3).*(a_2.*(1-a_2));	%	26x1
	
	
	D_1 += d_2(2:end)*a_1';
	
	D_2 += d_3*a_2';
	

end	
	

Dij_1 = (1/m)*D_1 + (lambda/m)*[zeros(size(Theta1,1),1),Theta1(:,2:end)];
	
Dij_2 = (1/m)*D_2 + (lambda/m)*[zeros(size(Theta2,1),1),Theta2(:,2:end)];


Theta1_grad = Dij_1;

Theta2_grad = Dij_2;






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
