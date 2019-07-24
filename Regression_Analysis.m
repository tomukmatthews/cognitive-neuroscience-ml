% Author: Part III Physics student
% INPUT: Feature matrix (must have 68 columns, with the first 67 being the
% features and the last being the target output). This code will then train
% an SVM regression model on 'training Data' and the model be used to
% predict reaction times by passing the feature matrix (ONLY) to
% trainedModel.predictFcn. E.g. trainedModel.predictFcn(X) will produce of
% vector of predicted reaction times. Finally this function will plot some
% of the results and give the RMSE so the performance can be assessed. Also
% the code applies PCA feature reduction before training.


pathappend = '/Users/tommatthews/part3project/';

load([pathappend 'scripts/class_data5.mat']);
load([pathappend 'scripts/class_data6.mat']);
load([pathappend 'scripts/class_data15.mat']);
load([pathappend 'scripts/class_data18.mat']);
data5  = cd5;
data6  = cd6;
data15 = cd15;
data18 = cd18;

trainingData = cat(1,data18,data5,data15);

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
% Convert input to table
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21', 'column_22', 'column_23', 'column_24', 'column_25', 'column_26', 'column_27', 'column_28', 'column_29', 'column_30', 'column_31', 'column_32', 'column_33', 'column_34', 'column_35', 'column_36', 'column_37', 'column_38', 'column_39', 'column_40', 'column_41', 'column_42', 'column_43', 'column_44', 'column_45', 'column_46', 'column_47', 'column_48', 'column_49', 'column_50', 'column_51', 'column_52', 'column_53', 'column_54', 'column_55', 'column_56', 'column_57', 'column_58', 'column_59', 'column_60', 'column_61', 'column_62', 'column_63', 'column_64', 'column_65', 'column_66','column_67','column_68'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21', 'column_22', 'column_23', 'column_24', 'column_25', 'column_26', 'column_27', 'column_28', 'column_29', 'column_30', 'column_31', 'column_32', 'column_33', 'column_34', 'column_35', 'column_36', 'column_37', 'column_38', 'column_39', 'column_40', 'column_41', 'column_42', 'column_43', 'column_44', 'column_45', 'column_46', 'column_47', 'column_48', 'column_49', 'column_50', 'column_51', 'column_52', 'column_53', 'column_54', 'column_55', 'column_56', 'column_57', 'column_58', 'column_59', 'column_60', 'column_61', 'column_62', 'column_63', 'column_64', 'column_65','column_66','column_67'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_68;

% Apply a PCA to the predictor matrix.
predictors = table2array(varfun(@double, predictors));

% 'inf' values have to be treated as missing data for PCA.
predictors(isinf(predictors)) = NaN;
numComponentsToKeep = 5;
[pcaCoefficients, pcaScores, ~, ~, explained, pcaCenters] = pca(...
     predictors, ...
    'NumComponents', numComponentsToKeep);
predictors = array2table(pcaScores(:,:));

%% Hyper-parameter optimisation via 5 - fold cross validation

% To use the optimal variables, leave these both as false. 
optimize2var = false;
optimize3var = false;

epsilon = 0.0173;
c = cvpartition(size(response, 1),'KFold',5);
sigma = optimizableVariable('sigma',[1e0,1e3],'Transform','log');
box = optimizableVariable('box',[1e-1,1e2],'Transform','log');

if optimize3var
    epsilon = optimizableVariable('epsilon',[0.001,0.1],'Transform','log');
    minfn = @(z)kfoldLoss(fitrsvm(predictors,response,'CVPartition',c,...
        'KernelFunction','gaussian','BoxConstraint',z.box,...
        'KernelScale',z.sigma,'Epsilon',z.epsilon));

    results = bayesopt(minfn,[sigma,box,epsilon],'IsObjectiveDeterministic',true,...
        'AcquisitionFunctionName','expected-improvement-plus')
    

    epsilon = results.XAtMinObjective.epsilon;
    kernelscale = results.XAtMinObjective.sigma;
    boxConstraint = results.XAtMinObjective.box;
    
elseif optimize2var

    minfn = @(z)kfoldLoss(fitrsvm(predictors,response,'CVPartition',c,...
        'KernelFunction','gaussian','BoxConstraint',z.box,...
        'KernelScale',z.sigma));

    results = bayesopt(minfn,[sigma,box],'IsObjectiveDeterministic',true,...
        'AcquisitionFunctionName','expected-improvement-plus')
    grid on

    Loss = results.MinObjective

    kernelscale = results.XAtMinObjective.sigma;
    boxConstraint = results.XAtMinObjective.box;

else
    
  kernelscale = 40.9;
  boxConstraint = 97.3;
   
end

%% Train the model with optimised hyper-parameters

% Train a regression model
% This code specifies all the model options and trains the model.
responseScale = iqr(response);
if ~isfinite(responseScale) || responseScale == 0.0
    responseScale = 1.0;
end

% Default parameters commented out
% boxConstraint = responseScale/1.349;
% epsilon = responseScale/13.49;
% kernelscale = 2.2;
regressionSVM = fitrsvm(...
    predictors, ...
    response, ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', kernelscale, ...
    'BoxConstraint', boxConstraint, ...
    'Epsilon', epsilon, ...
    'Standardize', true);

% Create the result struct with predict function
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
pcaTransformationFcn = @(x) [ array2table((table2array(varfun(@double, x)) - pcaCenters) * pcaCoefficients), x ];
svmPredictFcn = @(x) predict(regressionSVM, x);
trainedModel.predictFcn = @(x) svmPredictFcn(pcaTransformationFcn(predictorExtractionFcn(x)));

% Add additional fields to the result struct
trainedModel.PCACenters = pcaCenters;
trainedModel.PCACoefficients = pcaCoefficients;
trainedModel.RegressionSVM = regressionSVM;
trainedModel.About = 'This struct is a trained model exported from Regression Learner R2018b.';
trainedModel.HowToPredict = sprintf('To make predictions on a new predictor column matrix, X, use: \n  yfit = c.predictFcn(X) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nX must contain exactly 65 columns because this model was trained using 65 predictors. \nX must contain only predictor columns in exactly the same order and format as your training \ndata. Do not include the response column or any columns you did not import into the app. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appregression_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

%% Perform 5-fold cross validation to validate the model

% Perform cross-validation
KFolds = 5;
% NB: CROSS VALIDATION DOESNT USE STRATIFIED SAMPLING, COULD CHANGE THIS
cvp = cvpartition(size(response, 1), 'KFold', KFolds);
% Initialize the predictions to the proper sizes
validationPredictions = response;

for fold = 1:KFolds
    trainingPredictors = predictors(cvp.training(fold), :);
    trainingResponse = response(cvp.training(fold), :);
    
    % Apply a PCA to the predictor matrix.

    trainingPredictors = table2array(varfun(@double, trainingPredictors));
    % 'inf' values have to be treated as missing data for PCA.
    trainingPredictors(isinf(trainingPredictors)) = NaN;
    numComponentsToKeep = 5;
    [pcaCoefficients, pcaScores, ~, ~, explained, pcaCenters] = pca(...
        trainingPredictors, ...
        'NumComponents', numComponentsToKeep);
    trainingPredictors = array2table(pcaScores(:,:));
    
    % Train a regression model
    % This code specifies all the model options and trains the model.
    responseScale = iqr(trainingResponse);
    if ~isfinite(responseScale) || responseScale == 0.0
        responseScale = 1.0;
    end

%     Use optimised paramters instead of default.
%     boxConstraint = responseScale/1.349;
%     epsilon = responseScale/13.49;
%     kernelscale = results.XAtMinObjective.sigma;
%     boxConstraint = results.XAtMinObjective.box;
%     epsilon = results.XAtMinObjective.epsilon;
    
    regressionSVM = fitrsvm(...
        trainingPredictors, ...
        trainingResponse, ...
        'KernelFunction', 'gaussian', ...
        'PolynomialOrder', [], ...
        'KernelScale', kernelscale, ...
        'BoxConstraint', boxConstraint, ...
        'Epsilon', epsilon, ...
        'Standardize', true);
    
    % Create the result struct with predict function
    pcaTransformationFcn = @(x) array2table((table2array(varfun(@double, x)) - pcaCenters) * pcaCoefficients);
    svmPredictFcn = @(x) predict(regressionSVM, x);
    validationPredictFcn = @(x) svmPredictFcn(pcaTransformationFcn(x));
    
    
    % Compute validation predictions
    validationPredictors = predictors(cvp.test(fold), :);
    foldPredictions = validationPredictFcn(validationPredictors);
    
    % Store predictions in the original order
    validationPredictions(cvp.test(fold), :) = foldPredictions;
end

%% Evaluate performance of the model after k-fold CV and plot results.

vp = validationPredictions;
% Compute validation RMSE
isNotMissing = ~isnan(vp) & ~isnan(response);
validationRMSE = sqrt(nansum(( vp - response ).^2) / numel(response(isNotMissing) ))
[~,~,~,~,stats]  = regress(response,vp);
R_sqrd = stats(1)
[Corr ~] = corr(response,vp,'type','Pearson')

% yfit = trainedModel.predictFcn(X); where X is the matrix of predictors
% only

Np = 1:length(response);
figure
hold on
scatter(Np,response,'blue','filled')
scatter(Np,vp,'red','filled')
hold off
xlabel('Record number')
ylabel('Reaction Time / s')
legend({'Data', 'Model Prediction'})

figure
scatter(response,vp,'filled')
hline = refline(1,0);
hline.Color = 'k';
hline.LineStyle = '-';
ylabel('Model Predicted Reaction Time / s');
xlabel('Actual Reaction Time /s');
title('Model predtion vs Actual');
xlim([0.5 2])
ylim([0.5 2])

% responsefit = trainingData(:,end);
% figure
% subplot(1,2,1)
% ylim([0.5 2])
% hold on
% scatter(time,responsefit,'blue','filled')
% scatter(time,vp,'red','filled')
% plot(time_dscore,subj_dscore,'color','k','LineWidth',1.5)
% title('Predicted and observed RT vs time');
% grid on
% hold off
% xlabel('Time / s')
% ylabel('RT / s')
% legend({'Observed RT', 'Predicted RT', 'Subjective Drowsiness'})
% subplot(1,2,2)
% scatter(responsefit,vp,'filled')
% hline = refline(1,0);
% hline.Color = 'k';
% hline.LineStyle = '-';
% ylabel('Predicted RT / s');
% xlabel('Observed RT / s');
% title('Model prediction vs observations');
% xlim([0.5 2])
% ylim([0.5 2])
% grid on

%% Evaluate model generalisability to an unseen dataset (dataset 6, after the model was trained on datasets 5,15,18)...

yfit = trainedModel.predictFcn(cd6(:,1:end-1));
responsefit = cd6(:,end);
Np = 1:length(responsefit);
figure
hold on
scatter(Np,responsefit,'blue','filled')
scatter(Np,yfit,'red','filled')
hold off
xlabel('Record number')
ylabel('Reaction Time / s')
legend({'Data', 'Model Prediction'})
figure
scatter(responsefit,yfit,'filled')
hline = refline(1,0);
hline.Color = 'k';
hline.LineStyle = '-';
ylabel('Model Predicted Reaction Time / s');
xlabel('Actual Reaction Time /s');
title('Model predtion vs Actual');

RMSE = sqrt(nansum(( yfit - responsefit ).^2) / numel(responsefit(isNotMissing) ))
[~,~,~,~,stats]  = regress(responsefit,yfit);
R_sqrd = stats(1)
[Corr ~] = corr(responsefit,yfit,'type','Pearson')

