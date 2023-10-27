%% 学习使用SVM的文档，对比vl工具包和MATLAB工具包
clear
clc
close all
addpath(genpath("./vlfeat"))
y=[];X=[];
% 数据约定：一列表示一个样本

% Load training data X and their labels y
% 一列表示一个样本
load('vl_demo_svm_data.mat')

Xp = X(:,y==1); 
Xn = X(:,y==-1);


%创建样本
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(X(1,:)):d:max(X(1,:)),...
    min(X(2,:)):d:max(X(2,:)));
xGrid = [x1Grid(:),x2Grid(:)]';%整个坐标平面的点坐标
N = size(xGrid,2);


% Parameters
lambda = 0.01 ; % Regularization parameter
maxIter = 1000 ; % Maximum number of iterations

%% 使用VL工具包
figure(1)
plot(Xn(1,:),Xn(2,:),'*r')
hold on
plot(Xp(1,:),Xp(2,:),'*b')
axis equal ;

% 训练分类器
[w b info] = vl_svmtrain(X, y, lambda,...
                           'MaxNumIterations',maxIter,...
                           'DiagnosticFrequency',1)
% Visualisation
eq = [num2str(w(1)) '*x+' num2str(w(2)) '*y+' num2str(b)];

line = ezplot(eq, [-0.9 0.9 -0.9 0.9]);
set(line, 'Color', [0 0.8 0],'linewidth', 2);

% 正样本的得分
Scores_vl = w'*xGrid+b;%每个点都预测一下

labels_vl = ones(size(Scores_vl));
labels_vl(Scores_vl<0) = -1;

figure(3)
h_vl(1:2) = gscatter(xGrid(1,:),xGrid(2,:),labels_vl,...
         [0.1 0.5 0.5; 0.5 0.1 0.5]);%预测分类结果用不同的颜色展示
hold on
h_vl(3:4) = gscatter(X(1,:),X(2,:),y);%画出原始的训练样本
legend(h_vl,{'setosa region','versicolor region',...
    'observed setosa','observed versicolor'},...
    'Location','Northwest');
axis tight
hold off


%% 使用MATLAB工具包
figure(2)

plot(Xn(1,:),Xn(2,:),'*r')
hold on
plot(Xp(1,:),Xp(2,:),'*b')
axis equal ;

% 训练分类器
% MATLAB默认的数据和VL的不一样，一行表示一个样本，刚好是转置关系
matlab_svm = fitcsvm(X',y','Standardize',false,...
        'KernelFunction','rbf','BoxConstraint',1,'IterationLimit',maxIter);
% [mpr_lables,mpr_scores] = predict(matlab_svm,X');


if ~isempty(matlab_svm.Beta)
    eq = [num2str(matlab_svm.Beta(1)) '*x+' num2str(matlab_svm.Beta(2)) '*y+' num2str(matlab_svm.Bias)];

    line = ezplot(eq, [-0.9 0.9 -0.9 0.9]);
    set(line, 'Color', [0 0.8 0],'linewidth', 2);
end


% Scores_m = xGrid * matlab_svm.Beta + matlab_svm.Bias;
[~,score] = predict(matlab_svm,xGrid');
Scores_m = score(:,2);

labels_m = ones(size(Scores_m));
labels_m(Scores_m<0) = -1;


figure(4)
h_m(1:2) = gscatter(xGrid(1,:),xGrid(2,:),labels_m,...
         [0.1 0.5 0.5; 0.5 0.1 0.5]);%预测分类结果用不同的颜色展示
hold on
h_m(3:4) = gscatter(X(1,:),X(2,:),y);%画出原始的训练样本
legend(h_m,{'setosa region','versicolor region',...
    'observed setosa','observed versicolor'},...
    'Location','Northwest');
axis tight
hold off
