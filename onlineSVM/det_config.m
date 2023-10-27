function det_config = det_config(target_sz, image_sz)
%% 设置SVM的参数
det_config.svm_feature_type = 'fhog';% 'fhog' | 'cn'
%提取的特征参数设置，选取fHOG特征

switch det_config.svm_feature_type
    case 'fhog'
        % 创建正负样本的阈值
        det_config.thresh_p = 0.75; % IOU threshold for positive training samples 0.75
        det_config.thresh_n = 0.28; % IOU threshold for negative ones 0.30
        det_config.fhog_orientations = 9;
        det_config.cell_size = 4;

        target_max_win = 10*10 * 16;%目标最大的大小
    case 'cn'
        % 创建正负样本的阈值
        det_config.thresh_p = 0.75; % IOU threshold for positive training samples 0.75
        det_config.thresh_n = 0.28; % IOU threshold for negative ones 0.30
        target_max_win = 144;%目标最大的大小
        det_config.nbin = 32;% 提取特征所需的参数32
        det_config.cell_size = 1;
end

% 针对cell_size对目标大小进行修正，不能打只能小
target_sz = floor(target_sz / det_config.cell_size) * det_config.cell_size;

det_config.ratio = sqrt(target_max_win/prod(target_sz));%缩放比例
det_config.t_sz = round(target_sz*det_config.ratio);%所使用的区域大小
% 针对cell_size对目标大小进行修正，不能打只能小
det_config.t_sz = floor(det_config.t_sz / det_config.cell_size) * det_config.cell_size;
det_config.ratio = sqrt(prod(det_config.t_sz) / prod(target_sz));


det_config.lambda = 0.5;
det_config.MAX_ITER = 20;%最大迭代次数
det_config.C = 0.5;%pa算法中的学习率参数0.5

det_config.MAX_FPS_SAMPLE = 10;%样本容器最大容纳的帧数

det_config.use_resize = 'imResample';% 'MATLAB' | 'mexResize' |'imResample'
det_config.resize_type = 'bilinear';%'bilinear' or 'nearest'


det_config.target_sz = target_sz;
det_config.image_sz = image_sz;
det_config.detector_win_ratio = 1.5;%相对于核相关区域窗口的比例，更大的窗口重检测1.5
det_config.SVM_sample_win_ratio = 1.5;%svm创建训练样本的区域大小
det_config.SVM_detect_win_ratio = 1.9;%svm重新检测的区域大小
det_config.step = 1;% 采样步长

det_config.filter_sigma = 25;

det_config.current_fps = 0;

det_config.num_getsample = 0;
det_config.num_detector  = 0;

end