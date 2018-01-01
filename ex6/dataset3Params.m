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
nums = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
m = size(nums, 2);
params = zeros(m ^ 2, 2);
errors = zeros(m ^ 2, 1);

% initialize
idx = 1;
for i = 1 : m
    for j = 1 : m
        params(idx, :) = [nums(i), nums(j)];
        idx++;
    end
end

% calc prediction error on cv data
for i = 1 : size(params, 1)
    tmp_C = params(i, 1);
    tmp_sigma = params(i, 2);
    tmp_model = svmTrain(X, y, tmp_C, @(x1, x2) gaussianKernel(x1, x2, tmp_sigma));
    tmp_y = svmPredict(tmp_model, Xval);
    errors(i) = mean(double(tmp_y ~= yval));
end

% get the param
[val, ind] = min(errors)
C = params(ind, 1);
sigma = params(ind ,2);







% =========================================================================

end
