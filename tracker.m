% 修改后的positions是：[y0, x0, height, width]共四个维度，MATLAB坐标系
function [positions, time, params] = tracker(video_path, img_files, pos, target_sz, params)

%%%%%%%%%%%%%%%%%%%%  **检测前参数设定**  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 在run_tracker的超参数
params.reset_resize_raito = 0.90;%0.9
params.IMG_RATIO = 0.75;%图片缩放比例 .75
params.MAX_APP_SIZE = 64*64;%88*88
%响应最值阈值
params.G_RATIO_MIN_x  = 0.58;%位置滤波器 0.45 0.6 0.58
params.G_RATIO_MIN_s  = 0.45;%尺度滤波器
%置信度阈值
params.C_RATIO_MIN_x  = 0.31;%位置滤波器0.35
params.C_RATIO_MIN_s  = 0.40;%尺度滤波器

% 位置滤波器的学习率
params.kernel_x1.interp_factor = 0.022;
params.kernel_x2.interp_factor = 0.012;
% 目标外观模型的学习率
learning_rate_app = 0.027;% 0.01 0.03 .025 0.027
params.features_app.cell_size = 4;


% 遮挡重检测后的学习率衰减因子
trans_kernel_x1_learn_factor = 0.4;%0.4
trans_kernel_x2_learn_factor = 0.4;
app_kernel_learn_factor  = 0.4;

%% 重新更新模板模块
% 当前的trans参数与初试的参数相差很大的时候，使用全新的
MIN_FRAME_UPDATE = 20;%模型最小的更新帧数
MIN_CORR2 = 0.4; %最低的相似度


% 记录最初的参数
start_alphaf = [];
is_start_new_filter = false;

corr_list = [];



%% 解析参数
MAX_IMG_SIZE = params.MAX_IMG_SIZE;%整幅图片最大的限制
IMG_RATIO = params.IMG_RATIO;

im = imread([video_path img_files{1}]);%读取一帧图像，用于判断大小、通道数
% 超过阈值，按比例缩放
if prod(size(im,[1,2])) >= MAX_IMG_SIZE
    resize_image = true;
    % 如果图片大并且目标也很大
    TEMP_MAX_TARGET = params.translation_model_max_area/IMG_RATIO;
    if prod(target_sz) >= TEMP_MAX_TARGET
        IMG_RATIO = IMG_RATIO * params.reset_resize_raito;
        params.IMG_RATIO = IMG_RATIO;
    end
    pos = round( pos*IMG_RATIO );
    target_sz = round( target_sz*IMG_RATIO );
    im = imresize(im, IMG_RATIO);%对整幅图片进行缩放
else
    resize_image = false;
end
[im_heigt, im_width, ~] = size(im);% 图片的大小，在生成样本的时候判断是否越界，关键
% 重检测模块参数设置
params.detector = det_config(target_sz, [im_heigt, im_width]);

%% 图片缩放选用的方法
% 创建尺度滤波器模型所使用的缩放方法
params.scale_resize = 'mexResize';% 'imResample' | 'MATLAB' |'mexResize'
% 目标外观模型使用的方法
params.app_resize = 'mexResize';% 'imResample' | 'MATLAB' |'mexResize'
% 位置滤波器使用的方法
params.trans_resize = 'mexResize';% 'imResample' | 'MATLAB' |'mexResize'
% 缩放整幅图使用的方法
params.img_resize = 'mexResize';% 'imResample' | 'MATLAB' |'mexResize'

% 创建SVM检测器所使用的方法
params.detector.use_resize = 'imResample';% 'imResample' | 'MATLAB' |'mexResize'
params.detector.resize_type = 'bilinear';%'bilinear' or 'nearest'


%% 超参数
threshold_detector  = 0.55;%启动重检测的比例阈值0.55
threshold_train_SVM = 0.70;%进行SVM训练的比例阈值0.7 0.72
threshold_updateapp = 0.60;%更新目标模型的最值阈值0.7 0.68
threshold_accept_det = 1.33;%接受重检测结果的最值阈值1.5 1.3 1.26 1.3
%相邻帧双阈值运行下降的幅度
MAX_DOWN_c = 0.14;%0.2
MAX_DOWN_max = 0.14;%0.2

detector_window_sz_factor = 1.0;%重检测区域比创建样本区域大的倍速
MAX_INTERVAL_FPS_NOT_UPDATE_SVM = 120;%如果svm连续n帧不更新，将禁用svm模块，不认为重检测可信
MIN_INTERVAL_FPS_UPDATE_SVM = 3;%最大容忍间隔中需要更新的最低帧数
MAX_SVM_UPDATE_FPS = 30;% SVM更新中，如果连续稳定MAX_SVM_UPDATE_FPS，跳帧更新
STEP_SVM_UPDATE = 20;
current_SVM_fps = 0;
update_svm_list = false(1,numel(img_files));

MAX_APP_UPDATE_FPS = 60;% app更新中，如果连续稳定MAX_APP_UPDATE_FPS，跳帧更新
STEP_APP_UPDATE = 4;
current_APP_fps = 0;
update_app_list = false(1, numel(img_files));

is_accpet_detector = false;%记录是否接受重检测结果
is_start_detector = false;

%位置滤波器的学习率调整因子
LENGTH_TRANS_UPDATE = 120;
trans_learning_rate_factor_list = sin((1:LENGTH_TRANS_UPDATE)./LENGTH_TRANS_UPDATE.*pi/2);
% 目标模型的学习率
% learning_rate_app = kernel_app.interp_factor;
% learning_rate_app = 0.03;%0.01

% %加载映射矩阵
data_11 = load("w2c.mat");
data_10 = load("CNnorm.mat");
w2c_trans = data_11.w2c;
w2c_SVM   = data_10.CNnorm;%svm更适合采用这个
clear data_11;%释放变量
clear data_10;%释放变量

%% 解析参数
kernel_x1 = params.kernel_x1;
kernel_x2 = params.kernel_x2;
features_x1 = params.features_x1;
features_x2 = params.features_x2;
features_s = params.features_s;
padding = params.padding;
rho = params.rho;
varepsilon = params.varepsilon ;

%响应最小值阈值
G_RATIO_MIN_x = params.G_RATIO_MIN_x;
G_RATIO_MIN_s = params.G_RATIO_MIN_s;
%置信度最小值阈值
C_RATIO_MIN_x = params.C_RATIO_MIN_x;
C_RATIO_MIN_s = params.C_RATIO_MIN_s;

%位置滤波器参数
translation_model_max_area = params.translation_model_max_area;%位置滤波器的限制
output_sigma_factor = params.output_sigma_factor;%标签生成所用的标准差
lambda = params.lambda;
learning_rate_x1 = kernel_x1.interp_factor;
learning_rate_x2 = kernel_x2.interp_factor;
%尺度滤波器参数
nScales                 = params.nScales;%真实计算采用的尺度数量，17
nScalesInterp           = params.nScalesInterp;%插值后的尺度数量，33
scale_step              = params.scale_step;%尺度变换的底数，尺度呈指数变换
scale_sigma_factor      = params.scale_sigma_factor;
scale_model_factor      = params.scale_model_factor;%尺度滤波器的模型大小因子，调整滤波器的大小
scale_model_max_area    = params.scale_model_max_area;%尺度滤波器的最大值，512
scale_lambda            = params.scale_lambda;
scale_cell_size = features_s.cell_size;%尺度滤波器的cell_size大小
learning_rate_s = params.learning_rate_s;%尺度滤波器的学习率

% 目标模型参数
kernel_app = params.kernel_app;
features_app = params.features_app;

% 控件
show_visualization = params.show_visualization;

% debug
if params.debug
    update_visualization = show_video(img_files, video_path, resize_image, IMG_RATIO, params.video_name);
    update_visualization_svm = show_video_svm(img_files, video_path, resize_image, IMG_RATIO, params.video_name);
end



%% 如果目标大小超出了阈值，设置为阈值大小
%如果是init_target_sz超过了大小限制，这里就设置为translation_model_max_area
if prod(target_sz) > translation_model_max_area
    %如果超过了位置滤波器最大值的限制，计算缩放因子
    currentScaleFactor = sqrt(prod(target_sz) / translation_model_max_area);
else
    currentScaleFactor = 1.0;
end

%% 设置初始目标的大小，base_target_sz是缩放基准
base_target_sz = target_sz / currentScaleFactor;

%% 设定位置滤波器搜索区域大小
% 如果目标区域的高太大了，限制为1.5倍
if base_target_sz(1)/base_target_sz(2)>2
    window_sz = floor(base_target_sz.*[1.5, 1+padding] );
    % 如果目标太大了，就限制为2倍
elseif min(base_target_sz)>80 && prod(base_target_sz)/prod(size(im,[1,2]))>0.1
    window_sz=floor(base_target_sz * 2 );
    % 其他情况，同倍率进行放大
else
    window_sz = floor(base_target_sz * (1 + padding) );
end


%% 目标外观模型
app_size = target_sz + 2*features_app.cell_size;
MAX_APP_SIZE = params.MAX_APP_SIZE;
% 限制目标模型的大小
if prod(app_size) > MAX_APP_SIZE
    CurrentAPPScalse = sqrt(prod(app_size) / MAX_IMG_SIZE);
else
    CurrentAPPScalse = 1.0;
end
app_size = round( app_size/CurrentAPPScalse );



cell_size1 = features_x1.cell_size;
cell_size2 = features_x2.cell_size;
assert(cell_size1==cell_size2,'两个滤波器大小不一样，请查看Cell参数！');%当条件错误时，错发错误

%这个值不能乱改，需要和FHOG特征的cell_size保持一致
featureRatio = cell_size1; %特征比例，用于减少参数
%对目标区域与Cell_size为大小提取特征，缩小后的特征图大小，即实际滤波器大小
use_sz = floor(window_sz/featureRatio);
%%%对滤波器响应进行Cell_size插值，插回原图大小，提高精度
interp_sz = use_sz * featureRatio;%插值后的大小

%% 位置滤波器1
%由目标大小下采样featureRatio后的大小计算σ %创建高斯分布的标签，经过featureRatio降采样
output_sigma1 = sqrt(prod(floor(base_target_sz/featureRatio))) * output_sigma_factor;
%%创建标签，与fDSST一样的
y1f = fft2(gaussian_shaped_labels(output_sigma1, use_sz));
y1f = single(y1f);
cos_window1 = hann(size(y1f,1)) * hann(size(y1f,2))';	%最高点在中心
cos_window1 = single(cos_window1);

% y1f(abs(y1f) < 0.03) = 0;%把很小的值设为0，可以加快运算速度，而结果不受影响
y1f(abs(y1f) < max(abs(y1f))*0.008) = 0;%把很小的值设为0，可以加快运算速度，而结果不受影响
% cos_window1( cos_window1 < 0.02 ) = 0;
cos_window1( cos_window1 < max(cos_window1)*0.008 ) = 0;


%% 位置滤波器2
y2f = y1f;
cos_window2 = cos_window1;


%% 目标外观模型
app_sigma = sqrt(prod(floor(app_size/features_app.cell_size))) * output_sigma_factor;
app_yf = fft2(gaussian_shaped_labels(app_sigma, floor(app_size/features_app.cell_size)));
app_yf = single(app_yf);
% cos_window_app = hann(size(app_yf,1)) * hann(size(app_yf,2))';
% cos_window_app = single(cos_window_app);

app_yf(abs(app_yf) < max(abs(app_yf))*0.008) = 0;
% cos_window_app( cos_window_app < max(cos_window_app)*0.008 ) = 0;

%目标外观模型不需要添加窗函数，都已经全部都是目标本身了
cos_window_app = [];


%% 尺度滤波器：
if nScales > 0
    scale_sigma = nScalesInterp * scale_sigma_factor;
    %真正创建的尺度，共17个
    scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2)) * nScalesInterp/nScales;
    scale_exp_shift = circshift(scale_exp, [0 -floor((nScales-1)/2)]);%circshift(A,K) 循环将 A 中的元素平移 K 个位置
    %需要插值到的尺度，共33个
    interp_scale_exp = -floor((nScalesInterp-1)/2):ceil((nScalesInterp-1)/2);
    interp_scale_exp_shift = circshift(interp_scale_exp, [0 -floor((nScalesInterp-1)/2)]);

    scaleSizeFactors = scale_step .^ scale_exp;%17个尺度
    interpScaleFactors = scale_step .^ interp_scale_exp_shift;%要插值的尺度

    ys = exp(-0.5 * (scale_exp_shift.^2) /scale_sigma^2);%尺度标签，17个
    ysf = single(fft(ys));%行向量
    scale_window = single(hann(size(ysf,2)))';%行向量

    %make sure the scale model is not to large, to save computation time
    if scale_model_factor^2 * prod(target_sz) > scale_model_max_area
        %如果尺度滤波器大小太大了，缩放一下，计算缩放因子
        scale_model_factor = sqrt(scale_model_max_area/prod(target_sz));%尺度滤波器变换因子
    end

    %set the scale model size
    %得到尺度滤波器的大小，如果超过了限制，就设置为scale_model_max_area
    scale_model_sz = floor(target_sz * scale_model_factor);%设置尺度滤波器的大小

    %force reasonable scale changes
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ window_sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

%记录位置中心，目标尺度，共四个信息
positions = zeros(numel(img_files), 4);  %to calculate precision

%第一帧没有权重
lambda_fit_ratio = zeros(numel(img_files)-1,2);%存储位置滤波器的学习权重参数
G_RATIO_ALL = zeros(numel(img_files)-1,2);%存放占比数据
C_RATIO_ALL = zeros(numel(img_files)-1,2);%存放占比数据

sum_Cx = 0;%计算位置滤波器的置信度和
sum_Cs = 0;%计算尺度滤波器的置信度和
sum_Gx = 0;%计算位置滤波器的最大响应和
sum_Gs = 0;%计算尺度滤波器的最大响应和

MAX_APP_RESPONSE = [];

app_response_list = zeros(1,numel(img_files)-1);
app_response_mean = zeros(1,numel(img_files)-1);

detector_sign = false(1,numel(img_files));%是否重检测的标志
trainSVM_sign = false(1,numel(img_files));%是否更新SVM的标志
update_app_model = false(1,numel(img_files));

update_trans_sign = false(1,numel(img_files));
add_update_sign_list = false(1,numel(img_files));
%重检测SVM中间变量，用于debug
detector_max_response = [];
detector_re_max_response=[];
detector_xlable = [];
pred_score_list = [];


trans_c_list = zeros(numel(img_files)-1,2);

trans_x1_learning_list = [];
trans_x2_learning_list = [];
app_learning_list = [];

time = 0;  %to calculate FPS
%%%%%%%%%%%%%%%%%%%% 开始检测  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for frame = 1:numel(img_files)
    %load image
    im = imread([video_path img_files{frame}]);
    if resize_image%如果目标区域太大了，降采样减少图片大小
        switch params.img_resize % 'imResample' | 'MATLAB' |'mexResize'
            case 'mexResize'
                im = mexResize(im, [im_heigt,im_width], 'auto');% 缩放到指定大小
            case 'imResample'
                im = imResample(im,IMG_RATIO);
            case 'MATLAB'
                im = imresize(im,IMG_RATIO);
        end
    end

    tic()%开始计时
    %% ********************************************************************** %%
    %%                 如果不是第一帧，就要开始预测了                            %%
    %% ********************************************************************** %%
    if frame > 1
        responsef_all = cellfun(@get_translation_responsef,...
            {im,im}, {pos,pos}, {window_sz,window_sz}, {currentScaleFactor,currentScaleFactor},...
            {cos_window1,cos_window2}, {features_x1,features_x2}, {w2c_trans,w2c_trans},...
            {model_x1f,model_x2f}, {model_alpha1f,model_alpha2f}, {kernel_x1,kernel_x2},...
            {params.trans_resize,params.trans_resize}, "UniformOutput",false);

        % 对响应值进行插值
        responsef_all = cellfun(@resizeDFT2,...
            responsef_all,{interp_sz,interp_sz},...
            'UniformOutput',false);

        % 进行傅里叶反变换
        response_all = cellfun(@(x) real(ifft2(x,"symmetric")),...
            responsef_all,'UniformOutput',false);

        %在时域上计算APCE、PSR
        PSR_x = cellfun(@get_PSR,        response_all,  "UniformOutput",false);
        [~, APCE_x] = cellfun(@get_APCE, response_all,  "UniformOutput",false);
        %计算置信度
        C_x = cellfun(@(APCE,PSR) rho*APCE+(1-rho)*PSR, APCE_x, PSR_x,"UniformOutput",false);
        C1 = C_x{1};
        C2 = C_x{2};

        % 记录位置滤波器的置信度
        trans_c_list(frame-1,:) = [C1,C2];

        %计算融合系数
        lambda_fit_1 = C1/(C1+C2+varepsilon);
        lambda_fit_2 = C2/(C1+C2+varepsilon);


        lambda_fit_ratio(frame-1,:) = [lambda_fit_1, lambda_fit_2];%融合系数
        % 线性融合在时域、频域上都是一样的
        response = lambda_fit_1*response_all{1} + lambda_fit_2*response_all{2};


        %% 计算融合后的响应置信度，用于更新
        %在时域上计算APCE、PSR
        PSR_x = get_PSR(response);
        [Gmax_x, APCE_x] = get_APCE(response);
        %计算置信度
        C_x = rho * APCE_x + (1-rho) * PSR_x;

        % =============== 计算双阈值 =================================
        % 计算历史总和
        sum_Gx = sum_Gx + Gmax_x;
        sum_Cx = sum_Cx + C_x;
        %计算历史平均值
        mean_Gx = sum_Gx / (frame-1);
        mean_Cx = sum_Cx / (frame-1);
        % 计算当前值和历史均值的比例
        G_RATIO_ALL(frame-1,1) = Gmax_x/mean_Gx;
        C_RATIO_ALL(frame-1,1) = C_x/mean_Cx;


       % 使用ECO算法的优化部分，效果比纯最大值的好
       responsef = lambda_fit_1*responsef_all{1} + lambda_fit_2*responsef_all{2};
       [vert_delta, horiz_delta, ~] = optimize_scores(responsef, 3);
       pos = pos + [vert_delta , horiz_delta ] * currentScaleFactor;







if any(isnan(pos),"all") || any(isempty(pos),"all")
    pos = positions(frame-1,[1,2]);
end





        %% SVM模块部分
        if params.is_support_SVM
            % 加入目标外观模型
            responsef_app = get_translation_responsef(im,pos, app_size, CurrentAPPScalse, ...
                cos_window_app, features_app, w2c_trans, model_app_xf, model_app_alphaf, kernel_app, params.app_resize);
            response_app = real(ifft2(responsef_app,"symmetric"));
            max_app_response = max(response_app(:));% 得到相应最大值
            
            % 存储最大的相关数值
            if frame==2
                MAX_APP_RESPONSE = max_app_response;
            end

            % debug调参变量
            app_response_list(frame-1) = max_app_response;
            app_response_mean(frame-1) = sum(app_response_list)/(frame-1);

            % 如果双阈值方法断定发生了遮挡，并达到重检测阈值
            % 2022-08-26-16:13修改，把双阈值换成&&
            if max_app_response/MAX_APP_RESPONSE < threshold_detector ...
                    && ((G_RATIO_ALL(frame-1,1) < G_RATIO_MIN_x) && (C_RATIO_ALL(frame-1,1) < C_RATIO_MIN_x))...%判断当前发生了遮挡
                % 如果发生了遮挡进行重检测了，重新开始计数
                current_SVM_fps = 0;
                current_APP_fps = 0;

                %% 判断当前的SVM模型是否可靠
                if frame > MAX_INTERVAL_FPS_NOT_UPDATE_SVM
                    %如果在容忍区间中更新了svm
                    if sum(trainSVM_sign(frame-MAX_INTERVAL_FPS_NOT_UPDATE_SVM: frame-1)) >= MIN_INTERVAL_FPS_UPDATE_SVM
                        is_start_detector = true;
                    else
                        is_start_detector = false;
                    end
                else% 如果帧数还没超过MAX_INTERVAL_FPS_NOT_UPDATE_SVM，都判定为可靠的
                    is_start_detector = true;
                end
                %% 如果当前的SVM模型可靠，进行重检测
                if is_start_detector
                    params.detector.num_detector = params.detector.num_detector + 1;
                    detector_sign(frame) = true;
                    
                    % 对需要重检测的区域进行特征提取
                    detector_window_sz = window_sz*currentScaleFactor;
                    [feat, pos_samples, ~, weights]=get_det_samples_fhog(im, pos, detector_window_sz*detector_window_sz_factor, ...
                        params.detector, w2c_SVM, params.detector.SVM_detect_win_ratio, false);
                    % 重新检测目标
                    pred_score = params.detector.svm_struct.w' * feat + ...
                        params.detector.svm_struct.b;
                    % 加上高斯滤波器
                    pred_score = pred_score .* reshape(weights,1,[]);%乘上一个滤波器模板

                    % 找到得分最高的位置
                    max_pred_score = max(pred_score);
                    tpos=round(pos_samples(:, find(pred_score==max_pred_score, 1)));
                    tpos=reshape(tpos,1,[]);

                    %% debug
                    if isempty(tpos)
                        warning("tpos is empty!");
                    else
                        % debug显示的中间变量
                        t_rect = [tpos([2,1])-target_sz([2,1])/2, target_sz([2,1])];% svm检测的目标位置框
                        yy_min = min(pos_samples(1,:));
                        yy_max = max(pos_samples(1,:));
                        xx_min = min(pos_samples(2,:));
                        xx_max = max(pos_samples(2,:));
                        wear_rect = [xx_min, yy_min, xx_max-xx_min, yy_max-yy_min];% svm重检测区域
                        % 滤波器检测的目标位置
                        rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];


                        %% 计算重检测结果
                        responsef_app_re = get_translation_responsef(im,tpos,app_size, CurrentAPPScalse, ...
                            cos_window_app,features_app,w2c_trans,model_app_xf,model_app_alphaf,kernel_app,params.app_resize);
                        response_app_re = real(ifft2(responsef_app_re,"symmetric"));
                        % 得到相应最大值
                        max_app_response_re = max(response_app_re(:));

                        % debug显示的中间变量
                        detector_xlable(end+1) = frame;
                        detector_max_response(end+1) = max_app_response;
                        detector_re_max_response(end+1) = max_app_response_re;
                        pred_score_list(end+1)  = max_pred_score;


                        % 如果超过重检测前的相应值，接受当前结果
                        if max_app_response_re > threshold_accept_det * max_app_response && ...
                                max_app_response_re > 0 && max_pred_score >= 0%确保分类是正样本，是正确的

                            pos = tpos;
                            max_app_response = max_app_response_re;
                            is_accpet_detector = true;
                        else
                            is_accpet_detector = false;
                        end

                        %% 可视化显示
                        if params.debug
                            stop_svm = update_visualization_svm(frame, rect_position_vis, t_rect, wear_rect, is_accpet_detector);
                            if stop_svm, break; end  %user pressed Esc, stop early
                            hold off
                            drawnow
                        end
                    end
                else
                    % 2022-08-26-17:12修改
                    is_accpet_detector = false;%如果没有触发重检测，介绍结果这个赋值为false
                end
            else
                % 2022-08-26-17:12修改
                is_accpet_detector = false;
            end
        end

        %% 尺度自适应模块
        if nScales > 0
            %%%%%%%%%%%%%%%%%%%%%%%%%% 使用降维 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            xs_pca = get_scale_subwindow(im, pos, base_target_sz, currentScaleFactor*scaleSizeFactors, scale_model_sz, scale_cell_size, params.scale_resize);
            xs = bsxfun(@times, scale_window, scale_basis * xs_pca);%17*17
            xsf = fft(xs,[],2);%计算每一行的傅里叶变换
            %%%%================采用MOSSE算法===================
            scale_responsef = sum(sf_num .* xsf, 1) ./ (sf_den + scale_lambda);%一维行向量
            %把17个响应值插值到33个尺度上
            interp_scale_response = ifft( resizeDFT(scale_responsef, nScalesInterp), 'symmetric');
            recovered_scale_index = find(interp_scale_response == max(interp_scale_response(:)), 1);%找到最佳尺度响应

            %set the scale
            currentScaleFactor = currentScaleFactor * interpScaleFactors(recovered_scale_index);%计算好缩放因子，以初始大小作为基准
            %adjust to make sure we are not to large or to small
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end
            
            % 在时域上计算APCE、PSR
            PSR_s = get_PSR(interp_scale_response);
            [Gmax_s, APCE_s] = get_APCE(interp_scale_response);
            %计算置信度
            C_s = rho*APCE_s + (1-rho)*PSR_s;
            % 计算双阈值
            sum_Gs = sum_Gs + Gmax_s;
            sum_Cs = sum_Cs + C_s;

            mean_Gs = sum_Gs / (frame-1);
            mean_Cs = sum_Cs / (frame-1);

            G_RATIO_ALL(frame-1,2) = Gmax_s/mean_Gs;
            C_RATIO_ALL(frame-1,2) = C_s/mean_Cs;
        end
    end


    %% ********************************************************************** %%
    %%                               更新参数部分                                 %%
    %% ********************************************************************** %%
    if frame == 1  %第一帧，初始化参数
        if params.is_support_SVM
            [xf_all,alphaf_all] = cellfun(@get_translation_param,...
                {im,im,im},{pos,pos,pos},{window_sz,window_sz,app_size},{currentScaleFactor,currentScaleFactor,CurrentAPPScalse},...
                {cos_window1,cos_window2,cos_window_app},{features_x1,features_x2,features_app},{w2c_trans,w2c_trans,w2c_trans},...
                {kernel_x1,kernel_x2,kernel_app},{y1f,y2f,app_yf},{lambda,lambda,lambda}, ...
                {params.trans_resize,params.trans_resize,params.app_resize}, "UniformOutput",false);

            % 外观模型
            model_app_xf = xf_all{3};
            model_app_alphaf =alphaf_all{3};
            % 初始化SVM参数
            detector_window_sz = window_sz*currentScaleFactor;
            [feat, ~, labels, ~]=get_det_samples_fhog(im, pos, detector_window_sz, ...
                params.detector, w2c_SVM, params.detector.SVM_sample_win_ratio, true);%创建样本
            [params.detector.svm_struct.w, params.detector.svm_struct.b]= vl_svmtrain(feat, labels, ...
                params.detector.lambda, ...
                'MaxNumIterations',params.detector.MAX_ITER);

        else
            [xf_all,alphaf_all] = cellfun(@get_translation_param,...
                {im,im},{pos,pos},{window_sz,window_sz},{currentScaleFactor,currentScaleFactor},...
                {cos_window1,cos_window2},{features_x1,features_x2},{w2c_trans,w2c_trans},...
                {kernel_x1,kernel_x2},{y1f,y2f},{lambda,lambda}, ...
                {params.trans_resize,params.trans_resize}, "UniformOutput",false);
        end
        update_trans_sign(frame) = true;
        add_update_sign_list(frame) = true;

        % 位置滤波器1：
        model_alpha1f = alphaf_all{1};
        model_x1f = xf_all{1};
        % 位置滤波器2：
        model_alpha2f = alphaf_all{2};
        model_x2f = xf_all{2};

        start_alphaf = {model_alpha1f,model_alpha2f};

    else
        %% 判断双阈值是否剧烈下降
        if frame >= 3
            % 如果发生了快速下滑，也判断为样本不可靠
            if (G_RATIO_ALL(frame-1,1) - G_RATIO_ALL(frame-2,1) <= -MAX_DOWN_max  ) ||...
                    (C_RATIO_ALL(frame-1,1) - C_RATIO_ALL(frame-2,1) <= -MAX_DOWN_c )
                add_update_trans_sign = false;
            else
                add_update_trans_sign = true;
            end
        else
            add_update_trans_sign = true;% 一开始的状态
        end

        





        %% 双阈值判断是否可靠
        %         if (G_RATIO_ALL(frame-1,1) >= G_RATIO_MIN_x) && (C_RATIO_ALL(frame-1,1) >= C_RATIO_MIN_x) && add_update_sign
        if ((G_RATIO_ALL(frame-1,1) >= G_RATIO_MIN_x) && (C_RATIO_ALL(frame-1,1) >= C_RATIO_MIN_x)) ...%当判断没有发生遮挡
                || is_accpet_detector%或者发生了遮挡，但是重检测接受了新的结果
        
            %% 位置滤波器
%              update_trans_sign(frame) = true;%***** 只影响学习率，而不参与决断是否更新
            if add_update_trans_sign % 是否快速下降影响是否更新，参与决策
                update_trans_sign(frame) = true;%******











                % 更新学习率影响因子
                if frame >= LENGTH_TRANS_UPDATE
                    learning_idx = sum(update_trans_sign(frame-LENGTH_TRANS_UPDATE+1:frame));
                    current_learn_factor = trans_learning_rate_factor_list(learning_idx);
                else
                    current_learn_factor = 1.0;
                end


            [xf_all,alphaf_all] = cellfun(@get_translation_param,...
                {im,im},{pos,pos},{window_sz,window_sz},{currentScaleFactor,currentScaleFactor},...
                {cos_window1,cos_window2},{features_x1,features_x2},{w2c_trans,w2c_trans},...
                {kernel_x1,kernel_x2},{y1f,y2f},{lambda,lambda}, ...
                {params.trans_resize,params.trans_resize}, "UniformOutput",false);




                start_alphaf = cellfun(@abs,start_alphaf, "UniformOutput",false);
                temp_alphaf_all = cellfun(@abs,{alphaf_all{1},alphaf_all{2}}, "UniformOutput",false);
                filter_corr = cellfun(@corr2,start_alphaf, temp_alphaf_all);

                corr_list(end+1,:) = [frame,filter_corr];

                if frame>=MIN_FRAME_UPDATE && all(filter_corr<MIN_CORR2)
                    is_start_new_filter = true;
                else
                    is_start_new_filter = false;
                end







            % 根据置信度调C整学习率
            if is_accpet_detector
                learnRatio_x1 = learning_rate_x1*(1-1/C1) * current_learn_factor * trans_kernel_x1_learn_factor;
                learnRatio_x2 = learning_rate_x2*(1-1/C2) * current_learn_factor * trans_kernel_x2_learn_factor;
            else
                if is_start_new_filter
                    learnRatio_x1 = 1;
                    learnRatio_x2 = 1;
                else
                    learnRatio_x1 = learning_rate_x1*(1-1/C1) * current_learn_factor;
                    learnRatio_x2 = learning_rate_x2*(1-1/C2) * current_learn_factor;
                end
            end



            %用更新后的学习率更新模型
            model_alpha1f = (1 - learnRatio_x1) * model_alpha1f + learnRatio_x1 * alphaf_all{1};
            model_x1f     = (1 - learnRatio_x1) * model_x1f     + learnRatio_x1 * xf_all{1};

            model_alpha2f = (1 - learnRatio_x2) * model_alpha2f + learnRatio_x2 * alphaf_all{2};
            model_x2f     = (1 - learnRatio_x2) * model_x2f     + learnRatio_x2 * xf_all{2};

            trans_x1_learning_list(end+1,:) = [frame,learnRatio_x1];
            trans_x2_learning_list(end+1,:) = [frame,learnRatio_x2];

            if is_start_new_filter
                start_alphaf = alphaf_all;
            end
              

            end%**********


            %% 重检测部分
            if params.is_support_SVM
                % 更新目标外观模型的参数
                if max_app_response/MAX_APP_RESPONSE > threshold_updateapp

                    %% 外观模型
                    update_app_list(frame) = true;
                    current_APP_fps = current_APP_fps + 1;
                    % 判断是否连续更新，决断是否跳帧
                    if frame>MAX_APP_UPDATE_FPS ...
                            && sum(update_app_list(frame-MAX_APP_UPDATE_FPS:frame-1))==MAX_APP_UPDATE_FPS
                        if mod(current_APP_fps,STEP_APP_UPDATE) == 0%跳帧执行
                            update_app_sign = true;
                        else
                            update_app_sign = false;
                        end
                    else
                        update_app_sign = true;
                    end

                    if update_app_sign
                        [app_xf, app_alphaf] = get_translation_param(im,pos,app_size,CurrentAPPScalse, ...
                            cos_window_app,features_app,w2c_trans,kernel_app,app_yf,lambda,params.app_resize);
                       
                        % 如果开启了重检测并接受了，调整一下学习率，乘上一个衰减因子
                        if is_accpet_detector
                            learning_app = learning_rate_app * app_kernel_learn_factor;
                        else
                            if is_start_new_filter
                                learning_app = 1;
                            else
                                learning_app = learning_rate_app;
                            end
                        end

                        % 更新目标外观模型
                        model_app_xf = (1-learning_app)*model_app_xf + learning_app * app_xf;
                        model_app_alphaf = (1-learning_app)*model_app_alphaf + learning_app * app_alphaf;
                        update_app_model(frame) = true;



                        app_learning_list(end+1,:)=[frame, learning_app];
                    end
                end
                %% SVM模型
                if max_app_response/MAX_APP_RESPONSE > threshold_train_SVM
                    update_svm_list(frame) = true;
                    current_SVM_fps = current_SVM_fps + 1;
                    % 如果连续更新了MAX_UPDATE_FPS帧，认为稳定，开启跳帧更新模式
                    if frame>MAX_SVM_UPDATE_FPS ...
                            && sum(update_svm_list(frame-MAX_SVM_UPDATE_FPS:frame-1))==MAX_SVM_UPDATE_FPS
                        if mod(current_SVM_fps,STEP_SVM_UPDATE) == 0%跳帧执行
                            update_svm_sign = true;
                        else
                            update_svm_sign = false;
                        end
                    else
                        update_svm_sign = true;
                    end

                    % 开始更新SVM模型
                    if update_svm_sign
                        params.detector.num_getsample = params.detector.num_getsample+1;
                        detector_window_sz = window_sz*currentScaleFactor;% 同步放大采样区域，防止出错
                        trainSVM_sign(frame) = true;

                        [feat, ~, labels, ~]=get_det_samples_fhog(im, pos, detector_window_sz, ...
                            params.detector, w2c_SVM, params.detector.SVM_sample_win_ratio, true);

                        if is_start_new_filter
                            [params.detector.svm_struct.w, params.detector.svm_struct.b]= vl_svmtrain(feat, labels, ...
                params.detector.lambda, ...
                'MaxNumIterations',params.detector.MAX_ITER);
                        else
                            [params.detector.svm_struct.w, params.detector.svm_struct.b] = onlineSVMTrain(feat', labels, params.detector.C, ...
                            params.detector.svm_struct.w, params.detector.svm_struct.b);
                        end

%                         [params.detector.svm_struct.w, params.detector.svm_struct.b] = onlineSVMTrain(feat', labels, params.detector.C, ...
%                             params.detector.svm_struct.w, params.detector.svm_struct.b);
                    end
                end
            end
        end
    end


    %%%%%%%%%%%%%%%%%%%%%%%% 尺度滤波器相关  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Compute coefficents for the scale filter
    if nScales > 0
        %create a new feature projection matrix
        %%%=====================  采用降维方法  =====================
        %计算N*17的特征图
        xs_pca = get_scale_subwindow(im, pos, base_target_sz, currentScaleFactor*scaleSizeFactors, scale_model_sz, scale_cell_size, params.scale_resize);
        %更新尺度滤波器参数
        %这个是尺度滤波器的特征参数，类似KCF中的α参数，目标的外观模型
        if frame == 1
            s_num = xs_pca;%N*17
        else
            %更新特征图，作为一个学习参数
%           % 加入一个更新控制，使其保持和位置滤波器同步更新
            if (G_RATIO_ALL(frame-1,2) >= G_RATIO_MIN_s) && (C_RATIO_ALL(frame-1,2) >= C_RATIO_MIN_s)...
                    && add_update_trans_sign
                

                if is_start_new_filter
                    learning_rate_s = 1;
                else
                    learnRatio_s = learning_rate_s*(1-1/C_s);
                end


                
                s_num = (1 - learnRatio_s) * s_num + learnRatio_s * xs_pca;
            end
        end

        %这个是降维参数
        bigY = s_num;%N*17，一直学习的特征参数
        bigY_den = xs_pca;%N*17，当前帧的特征

        %%%%%%%% ===========采用QR分解================
        %bigY的维度大小为：N*17，所以scale_basis的大小为：N*17
        [scale_basis, ~] = qr(bigY, 0);
        [scale_basis_den, ~] = qr(bigY_den, 0);
        %重新计算得到的尺度滤波器降维矩阵，用学习到的特征参数计算，和位置滤波器的降维矩阵类似
        scale_basis = scale_basis';%转置，大小变为17*N
        %create the filter update coefficients
        %对特征进行降维，然后求解响应
        sf_proj = fft(bsxfun(@times, scale_window, scale_basis * s_num),[],2);%17*17

        %尺度滤波器的分子参数，学习参数，从s_num间接学习
        sf_num = bsxfun(@times,ysf,conj(sf_proj));%根据一直学习的特征，降维求解出的，17*17

        %得到的是提取的尺度特征，加窗后，得到的是当前帧的
        xs = bsxfun(@times, scale_window, scale_basis_den' * xs_pca);%17*17
        xsf = fft(xs,[],2);
        new_sf_den = sum(xsf .* conj(xsf),1);%对每一列求和，返回一个行向量

        %更新尺度滤波器的分母
        if frame == 1
            sf_den = new_sf_den;
        else
            %如果响应的最值和置信度的和前一帧的比值大于阈值，则进行更新
%             if (Gmax_s >= mean_Gs * G_RATIO_MIN_s) && (C_s >= mean_Cs*C_RATIO_MIN_s)
%             if (G_RATIO_ALL(frame-1,2) >= G_RATIO_MIN_s) && (C_RATIO_ALL(frame-1,2) >= C_RATIO_MIN_s)

            % 加入控制信号，使尺度滤波器保持和位置滤波器同步更新
            if (G_RATIO_ALL(frame-1,2) >= G_RATIO_MIN_s) && (C_RATIO_ALL(frame-1,2) >= C_RATIO_MIN_s)...
                    && add_update_trans_sign
                sf_den = (1 - learnRatio_s) * sf_den + learnRatio_s * new_sf_den;
            end
        end
    end


    % calculate the new target size
    target_sz = floor(base_target_sz * currentScaleFactor);%更新尺度大小
    CurrentAPPScalse = sqrt(prod(target_sz+ 2*features_app.cell_size)/prod(app_size));%更新目标的比例大小

    time = time + toc();


    %save position and timing
    %保存结果，MATLAB坐标系
    positions(frame,:) = [pos target_sz];%MATLAB坐标系


    %% 结果可视化
    %visualization
    if show_visualization ==1
        %在figure中画方框，参数是笛卡尔坐标系的形式
        rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        if params.debug
            stop = update_visualization(frame, rect_position_vis);
            if stop, break; end  %user pressed Esc, stop early
            hold off
        else
            %这样写能够快很多，不用每次都创建窗口
            if frame == 1
                figure("Name","ORCT-Tracking");
                %'Border'设置为'tight'，不留空隙
                % 'InitialMagnification'设置图像显示的初始放大倍率
                im_handle = imshow(im, 'Border','tight', 'InitialMag', 100 + 100 * (length(im) < 500));

                %rectangle('Position',pos) 在二维坐标中创建一个矩形。将 pos 指定为 [x y w h] 形式的四元素向量（以数据单位表示）
                rect_handle = rectangle('Position',rect_position_vis, 'EdgeColor','g','LineWidth',2);%画图像框
                text_handle = text(10, 10, int2str(frame));%显示帧数
                set(text_handle, 'color', [0 1 1]);
            else
                try
                    set(im_handle, 'CData', im)%更新图片
                    set(rect_handle, 'Position', rect_position_vis)%更新图像框
                    set(text_handle, 'string', int2str(frame));%更新显示的帧数
                catch
                    return
                end
            end
        end
        drawnow
    end
end

if resize_image
    positions = positions/IMG_RATIO;
end

run('myplot.m');

end
