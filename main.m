% Artificial Neural Networks final project.
% Random Fourier extreme learning machine with ?2;1-norm regularization.

clc;
clear;
close all;

%% Load Dataset

load iris_dataset;

X = irisInputs';
Y = irisTargets';

%% Initialization

nEpoch = 50;

nSamples = size(X, 1);
nFeatures = size(X, 2);

maxL = 200;      % Maximum number of neurons in the hidden layer.
stepL = 5;      % Hidden layer growth step size.

delta = 1.5;    % Weights matrix variance whitch could be in range (2^{-24}, 2^{8}).
C = 1.5;        % Regularization term(2^{-8}, 2^{24}).

nO = size(Y, 2);        % Number of neurons in the output layer.

nBetaUpdate = 10;     % The number of times that the output matrix is updated. 1000

trainAcc = zeros(maxL / stepL, 1);  % Matrix to hold training accuracy results.

% K-Fold Validation Parameters

K=5;    % Number of folds



%% ELM Outer Loop

for L = 5 : stepL : maxL
    
    D = eye(L);
    
    W = delta^2 * randn(nFeatures, L); % The weights of each neuron is stored in each column.
    
    % Intialize output weights matrix
    beta = zeros(L, nO);
    
    for epoch = 1 : nEpoch
        
        % Perform K-Fold cross validation
        Indices = crossvalind('Kfold', nSamples, K);
        
        k = randi(K);
        
        % Specify train and test data based on the indices resulted from
        % k-fold partioning.
        trX = X(Indices ~= k, :);
        trY = Y(Indices ~= k, :);
        
        tsX = X(Indices == k, :);
        tsY = Y(Indices == k, :);
        
        % Initialize and Create Train Phase H
        H_tr = zeros(size(trX, 1), L);
        
        for i = 1 : size(trX, 1)
            
            for j = 1 : L
                
                tmp = trX(i, :) * W(:, j);
                
                % Perform Fourier Transform
                H_tr(i, j) = (1 / sqrt(L)) * exp(1i * tmp);
                
                % Perform Morlet Transform
%                 H_tr(i, j) =...
%                     (1 / sqrt(L)) * exp(-.5 * (1i * tmp)^2) * cos(5 * 1i * tmp);
                
            end
            
        end
        
        
        % Initialize and Create Test Phase H
        H_ts = zeros(size(tsX, 1), L);
        
        for i = 1 : size(tsX, 1)
            
            for j = 1 : L
                
                tmp = tsX(i, :) * W(:, j);
                
                % Perform Fourier Transform
                H_ts(i, j) = (1 / sqrt(L)) * exp(1i * tmp);
                
            end
            
        end
        
        
        % Caculate Beta (the output weight matrix)
        for t = 1 : nBetaUpdate
            
            % Train
            
% %             if L < size(trX, 1)
                
                beta = inv(D/C + H_tr' * H_tr) * H_tr' * trY;
                
%             else
                
%                 beta = inv(D) * H_tr' * inv(eye(L)/C + H_tr * inv(D) * H_tr') * trY;
       
%                 
%             end
            
            
            for i = 1 : L
                
                D(i, :) = D(i, :) * norm(beta(i, :), 2);
                
            end
            
        end
        
    end
    
    % Calculate Accuracy for different number of hidden neurons
    
    trOut = real(round(H_tr * beta));
    
    tsOut = real(round(H_ts * beta));
    
    trainAcc(L / stepL) = L2_1_Norm(trOut & trY) / size(trY, 1);
    
    testAcc(L / stepL) = L2_1_Norm(tsOut & tsY) / size(tsY, 1);
    
    % Display Results
    
    disp(['Number of Hidden Neurons: ', num2str(L),...
        ', Train Acc: ', num2str(trainAcc(L / stepL)),...
        ', Test Acc: ',  num2str(testAcc(L / stepL))]);
    
end


%% Show Results

figure(1);

nH = stepL : stepL : maxL;

plot(nH, trainAcc);

hold on

plot(nH, testAcc);

xlabel('Number of Hidden Neurons');
ylabel('Accuracy');

legend({'Train', 'Test'});