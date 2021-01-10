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

%adding X0 feature
X=[ones(m,1),X];

%recoding y
y_new=zeros(m,num_labels);
for i=1:m,
	y_new(i,y(i))=1;
end;

%second layer calculation
z2=X*Theta1';
a2=sigmoid(z2);

%adding a(2)0 element in hidden layer
a2=[ones(m,1),a2];

%output layer calculation
z3=a2*Theta2';
a3=sigmoid(z3);

%cost function calculation by first adding all columns and then adding all rows
J=(1/m)*sum(sum(-y_new.*log(a3)-(1-y_new).*log(1-a3),2),1);

%calculating first regularisation term
reg1=lambda/(2*m)*sum(sum(Theta1.^2,2)-Theta1(:,1).^2,1);

%calculating second regularisation term
reg2=lambda/(2*m)*sum(sum(Theta2.^2,2)-Theta2(:,1).^2,1);

%Calculating regularized cost function
J=J+reg1+reg2;

%finding sigmoid gradient of z2
%a2sg=sigmoidGradient(z2);
%size(a2sg)

%adding a(2)0 value
%a2sg=[ones(m,1),a2sg];

%implementing back propogation

delta2=zeros(num_labels,size(a2,2));
delta1=zeros(size(a2,2)-1, size(X,2));
%size(delta1)

for i=1:m,
	%forward pass
	z2=X(i,:)*Theta1';	
	a2=sigmoid(z2);
	a2=[1,a2];
	z3=a2*Theta2';
	a3=sigmoid(z3);
	
	%calculating delta3 (lower case)
	d3=a3-y_new(i,:);

	%calculating delta2 (lower case)
	d2=d3*Theta2(:,2:end).*sigmoidGradient(z2);

	%calculation delta(upper case)
	delta2=delta2+d3'*a2;
	delta1=delta1+d2'*X(i,:);
end;

%calculating average error in gradient
Theta2_grad=delta2/m;
Theta1_grad=delta1/m;

%Theta2_grad(:,1)

%Regularizing the gradient
for i =1:hidden_layer_size,
	for j=1:input_layer_size+1,
		if (j~=1),
			Theta1_grad(i,j)=Theta1_grad(i,j)+lambda/m*Theta1(i,j);
		end;
	end;
end;
for i =1:num_labels,
	for j=1:hidden_layer_size+1,
		if (j~=1),
			Theta2_grad(i,j)=Theta2_grad(i,j)+lambda/m*Theta2(i,j);
		end;
	end;
end;








% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
