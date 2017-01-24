function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% non regularized term
T1 = 0;
for i=1:m
     h = sigmoid((theta')*(X(i,:)'));
     T1 = T1 + (-y(i)*log(h) - (1 - y(i))*log(1-h) );
end

T1 = T1/m;

% regularized term
T2 = 0;
for i=2:length(theta)
    T2 = T2 + (theta(i))^2
end
T2 = T2*lambda/(2*m)

J = T1 + T2;


% gradients
for j = 1:length(theta)
    grad(j) = 0;
    for k = 1:m
        h = sigmoid((theta')*(X(k,:)'));
        grad(j) = grad(j) + (h - y(k))*X(k,j);
    end
    if j == 1
        grad(j) = grad(j)/m;
    else
        grad(j) = grad(j)/m + lambda*theta(j)/m;
    end
end

% =============================================================

end
