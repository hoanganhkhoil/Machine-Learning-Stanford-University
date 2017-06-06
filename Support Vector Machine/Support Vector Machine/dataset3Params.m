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


C_array = [0.01,0.03,0.1,0.3,1,3,10,30];
Sigma_array = [0.01,0.03,0.1,0.3,1,3,10,30];


box = zeros(64,3);
count = 0;

for i = 1:8
    C = C_array(i);

    for j = 1:8

        count = count + 1;

        sigma = Sigma_array(j);

        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));

        box(count,1) = C;
        box(count,2) = sigma;
        box(count,3) = error;
    end
end

column_error = box(:,3);

[Value,index] = min(column_error);
            
C = box(index,1);
sigma = box(index,2);



% =========================================================================

end
