% 统一采用MATLAB坐标系====[y0, x0, h, w]，先目标中心坐标后目标大小
% 滤波器的大小都是根据第一帧的大小确定的，后面都不发生改变
% 支持的特征类型有（共9种）:
% === fhog   ：31维的FHOG特征，如果是RGB图，选取梯度模值最大的那个
% === gfhog  ：31维的FHOG特征，如果是RGB图，先转为灰度图再计算
% === gray   ：1维灰度特征
% === dsst   ：1维灰度+fhog特征的前28维
% === cn     ：11维度的CN特征，由概率组成
% === fhogcn ：FHOG前27维+11维CN特征
% === cnhist ：11维CN特征+8维局部直方图特征
% anti-occlusion real-time correlation tracking(ORCT)
%% 初始化参数
clear
close all
% 把必要的包添加到路劲
% addpath(genpath("./Data-set"));%下载数据所需的.
addpath(genpath("./temp"));
addpath(genpath("./show_result"));
addpath(genpath("./features"));
addpath(genpath("./onlineSVM"));
addpath(genpath("./scale"));
addpath(genpath("./translation"));
addpath(genpath("./unity"));

% 数据集的位置
run("../data_set.m");%读取上一级的文件,设置测试数据路径base_path

%定义结构体，存储数据
kernel_x1.type = 'gaussian'; %'linear' | 'gaussian' | 'polynomial'
kernel_x2.type = 'gaussian';
kernel_app.type = 'gaussian';

% 选取合适的特征
features_x1.feature_type = 'fhog'; %'fhog' | 'gfhog' | 'gray' | 'dsst' | 'cn' | 'fhogcn' | 'cnhist' | 'fhoghist'
features_x2.feature_type = 'cn';
features_s.feature_type = 'fhog';
features_app.feature_type = 'cn';%dsst

% 可视化debug控件
params.show_visualization = 1;%是否展示实时跟踪结果 |0 |1 
params.show_plots = 1;%是否绘制精度图
params.show_param = 0;%是否打印出参数
params.show_TempPlots = 1;%是否绘制置信度信息
params.debug = 1;
%是否开启SVM
params.is_support_SVM = true;


%%%%%%%%%%%%%========设置位置滤波器核相关的参数，不同核具有不同的学习率======%%%%%%%%%%%%%%%%%%%
[params.kernel_x1, params.features_x1] = set_kernel_and_feature(kernel_x1,features_x1);
[params.kernel_x2, params.features_x2] = set_kernel_and_feature(kernel_x2,features_x2);
[params.kernel_app, params.features_app] = set_kernel_and_feature(kernel_app,features_app);
%设置尺度滤波器的参数
params.features_s = set_scale_feature(features_s);

%定义一个结构体，传输参数
params.padding = 1.95;%三个滤波器都共用这个参数
params.IMG_RATIO = 0.75;%图片缩放比例
params.rho = 0.8; %APCE、PSR比例系数
params.varepsilon=1e-5; %极小值



% 图片最大限制
params.MAX_IMG_SIZE = 360*360;

% 目标模型大小限制
params.MAX_APP_SIZE = 88*88;

% 位置滤波器
params.lambda = 1e-3;
params.output_sigma_factor = 0.1;%位置滤波器采用的标准差，创建高斯标签时所用0.1
params.translation_model_max_area = 1600;%位置滤波器的限制，设为512效果不错 1296


% 尺度滤波器
params.scale_lambda = 1e-3;
params.scale_sigma_factor = 1/16;    %尺度滤波器采用的高斯标签的标准1/16
params.nScales = 17;%真实计算采用的尺度数量，17
params.nScalesInterp = 33;%插值后的尺度数量，33
params.scale_step = 1.02;%尺度变换的底数，尺度呈指数变换
params.scale_model_factor = 1.0;%尺度滤波器的模型大小因子，调整滤波器的大小
params.scale_model_max_area = 1024;%尺度滤波器的最大值，512
params.scale_cell_size = params.features_s.cell_size;%尺度滤波器的cell_size大小
params.learning_rate_s = 0.03;%尺度滤波器的学习率

%% 选择要测试的视频数据
video = choose_video(base_path);%返回选择的图片名字，一个元组
assert(~isempty(video),"没有选择有效的视频，错误！！");


params.video_name = video;%记录选中视频的名字
%% 读取数据集参数，返回MATLAB坐标系的结果
[img_files, pos, target_sz, ground_truth, video_path] = ...
    load_video_info(base_path, video);

%% 打开日志
diary('run_OSCT_allresult.txt');%日志记录
fprintf("\n")
fprintf(datestr(now,31));
fprintf("\n")
%% 调用跟踪算法进行跟踪
%输出MATLAB坐标系下的结果
[positions, time, params] = tracker(video_path, img_files, pos, target_sz, params);

%% 绘图，计算准确率
%calculate and show precision plot, as well as frames-per-second
%precisions是一个数组，在不同阈值下的准确率
%其实已经得到了预测的位置坐标positions，真实的坐标为ground_truth,两者均为MATLAB坐标系，前两者为中心坐标
%对比不同的算法效果的时候，可以用不同的positions，画出不同颜色的框
%约定precisions==[位置准确率，大小准确率]
precisions = precision_plot(positions, ground_truth, video, params.show_plots);

fps = numel(img_files) / time;
[distance_precision, overlap_precision, average_center_location_error,S] = ...
    compute_performance_measures(positions, ground_truth);

% 打印详细的参数设置
if params.show_param
    print_params(params);
end

fprintf(['OSCT_%s with %s  %s: '...
    '\n## Distance-Precision (20px):% 1.1f '...
    '\n## Overlap_precision  (0.5): % 1.1f '...
    '\n## CLE: %.2f'...
    '\n## S:   %.2f'...
    '\n## FPS: %4.2f\n'],...
    video,params.features_x1.feature_type ,params.features_x2.feature_type , distance_precision*100, overlap_precision*100, ...
    average_center_location_error,S*100,fps)
fprintf('=================================\n');

%return precisions at a 20 pixels threshold
precision = precisions(20);
diary off