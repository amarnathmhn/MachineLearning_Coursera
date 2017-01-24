function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% add 1's to the X matrix
X = [ones(m,1) X];

% calculate A1
A1 = Theta1*(X'); % In a column of A1, there are activations of all hidden layer neurons. Each column is for a testing sample
A1 = [ones(1,m); sigmoid(A1)];
A2 = Theta2*(A1);% In a column of A2, there are activations of all output layer neurons. Each column is for a testing sample

[~,p] = max(sigmoid(A2));














% =========================================================================


end
