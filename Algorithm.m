clc
close all
clearvars

%% Step 1: Dataset Preparation
% The data were prepared so that they could be fed to the SVM accordingly. The data is first loaded from the CSV file and 
% the necessary columns extracted.The data is also shuffled up to avoid any variance or bias problems.
%%
%Auto detect import options
optionSet = detectImportOptions('BCCD.csv');

%Read the csv file and store
file = readtable('BCCD.csv',optionSet); %The readtable function automatically removes any column headers from the data if present.

data = table2cell(file); %Convert the table to a cell array for ease of access

sampleNumber = size(data,1) %Get the number of all diagnosis by getting the number of rows

%Create X and Y for SVM Training
randomNumber = randperm(sampleNumber); %Permutate numbers upto the number of samples randomly, This works as shuffling our rows

%Load our X with features
X_temp = cell2mat(data(:,1:9)); %Convert it to a matrix and selecting the collumn ranges containing features
X = X_temp(randomNumber(1:end),:); %Store accordingly with the random number permutations

numberFeatures = size(X,2); %Store the number of features of the dataset

%Load Y with the labels.
Y_temp = cell2mat(data(:,10));
%Convert the categorical characters as classnames to indexes
Y = Y_temp(randomNumber(1:end),:);
%% Step 2: Perform Cross Validation using K-fold 10 times
% To reduce any underfitting or overfitting that may occur during testing, the data is cross 
% validated using K-Fold 10 times. The folds are stratified ensuring a uniform distribution 
% of each class within each fold further lowering any data bias that would have been present.
%%
%Validate now the training set using K-fold 10 times. Stratifying is ensured an equal class distribution in all folds
CV = cvpartition(Y,'KFold',10,'Stratify',true);
%% Step 3: Feature Ranking
% These steps ranks our feature and creates a feature sets consisting  a given number of all the features 
% from 1 feature to all features of the dataset
%%
optionSet = statset('display','iter','UseParallel',true); %Sets the display option
rng(5); %This sets our random state.

fun = @(train_data, train_labels, test_data, test_labels)...
    sum(predict(fitcsvm(train_data, train_labels,'Standardize',true,'KernelFunction','gaussian'), test_data) ~= test_labels);

%Rank the features using forward sequential forward selection
%The ranking of features stored inside history
[fs, history] = sequentialfs(fun, X, Y, 'cv', CV, ...
    'options', optionSet,'nfeatures',numberFeatures);
%% Step 4: Kernel and Feature Selection
% This step analyzes 3 kernel functions performance in regard to a given 
% feature set
%%
rng(3);
Accuracy(numberFeatures,6) = 0; %Initializes where to store the performance for each feature set and kernel
for count=1:numberFeatures
    %Store our best features
    Accuracy(count,1) = count;
    
    %Linear
    Model1= fitcsvm(X(:,history.In(count,:)),Y,'BoxConstraint',1,'CVPartition',CV,'KernelFunction',...
        'linear','Standardize',true,'KernelScale','auto');
   
    % Compute validation accuracy
    Accuracy(count,2) = (1 - kfoldLoss(Model1, ...
        'LossFun', 'ClassifError'))*100;
    
    %polynomial training
    Model2= fitcsvm(X(:,history.In(count,:)),Y,'BoxConstraint',1,'CVPartition',CV,'KernelFunction',...
        'polynomial','Standardize',true,'KernelScale','auto');
   
    % Compute validation accuracy
    Accuracy(count,3) = (1 - kfoldLoss(Model2, ...
        'LossFun', 'ClassifError'))*100;
    
    %Gaussian
    Model3= fitcsvm(X(:,history.In(count,:)),Y,'BoxConstraint',1,'CVPartition',CV,'KernelFunction',...
        'gaussian','Standardize',true,'KernelScale','auto');
   
    % Compute validation accuracy
    Accuracy(count,4) = (1 - kfoldLoss(Model3, ...
        'LossFun', 'ClassifError'))*100;
end
%% Visualize Results
%%
figure
plot(Accuracy(:,2:6))
title('Breast Cancer Coimbra Model Perfomance')
xlabel('Number of Ranked Features')
ylabel('Model Perfomance(%)')
legend('Linear','Polynomial','Gaussian')
grid on;
%% Step 5: Select the best kernel function
% Select the best hyperparameters for the highest accuracy perfomance for the dataset.
%%
%The best observed Kernel is a gaussian
rng(3);
%Increase its Maximum Objective evaluations to 80
Model3 = fitcsvm(X(:,history.In(5,:)),Y,'KernelFunction', ...
    'gaussian','Standardize',true,'OptimizeHyperparameters','auto', ...
    'HyperparameterOptimizationOptions',struct('UseParallel',true,...
    'ShowPlots',false,'MaxObjectiveEvaluations',80,'Repartition',true));
%% Step 6: Train the SVM model and find the Accuracy and the evaluate the validation loss
%%
%Pass on the X and Y to the SVM classifier.
rng(3); %seeds our random generator allowing for reproducibility of results

const = Model3.BoxConstraints(1,:);
kernel = Model3.KernelParameters.Scale;

%using the gaussian kernel
bestModel = fitcsvm(X(:,history.In(4,:)),Y,'CVPartition',CV,'KernelFunction',...
    'gaussian','Standardize',true,'BoxConstraint',const,'KernelScale',kernel);

% Compute validation accuracy
Accuracy = (1 - kfoldLoss(bestModel, 'LossFun', 'ClassifError'))*100
%Compute validation loss. The lower the better the prediction
Error = kfoldLoss(bestModel)*100
%% Step 7: Evaluate the model's perfomance using a confusion Matrix
figure
[Y_prediction, validationScores] = kfoldPredict(bestModel);
confusionMatrix=confusionmat(Y,Y_prediction);
%Create a Confusion Matrix
%The confusion matrix gives a view of the rate of true positives to false negatives.
%It enables to have proper view of how effecive the model is in prediction of illness minimizing of having false negatives
conMatHeat = heatmap(confusionMatrix,'Title','Confusion Matrix of BCCD dataset','YLabel','True Diagnosis','XLabel','Predicted Diagnosis',...
    'XDisplayLabels',{'Healthy(1)','Patients(2)'},'YDisplayLabels',{'Healthy(1)','Patients(2)'},'ColorbarVisible','off');
%The following functions calculate the recall, specificity, precision and F1 Score
recallFunction = @(confusionMat) diag(confusionMat)./sum(confusionMat,2);
specificityFuncion = @(confusionMat) diag(confusionMat)./sum(confusionMat,2);
precisionFuncion = @(confusionMat) diag(confusionMat)./sum(confusionMat,1);
f1ScoresFuncion = @(confusionMat) 2*(precisionFuncion(confusionMat).*recallFunction(confusionMat))./(precisionFuncion(confusionMat)+recallFunction(confusionMat));

Recall = recallFunction(confusionMatrix)*100
Specificity = specificityFuncion(confusionMatrix)*100
Precision = precisionFuncion(confusionMatrix)*100
F1Score = f1ScoresFuncion(confusionMatrix)*100