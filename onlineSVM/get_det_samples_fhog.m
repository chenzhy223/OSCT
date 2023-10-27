% det using the  bb as [x y w d]
function [feat, pos_samples, labels, weights]=get_det_samples_fhog(im, pos, window_sz, det_config, w2c, win_ratio, isgetsamples)
% feat[D,N]============二维的样本特征，一列表示一个样本
% pos_samples[2,N]=====样本的位置信息，先y后x
% labels[N,1]==========样本与目标的重叠比例而产生的SVM标签
% weights=========高斯滤波器模板

t_sz = det_config.t_sz;%缩放后使用的区域大小
%% 检查检测区域大小是否合法
detector_wear_size = floor(window_sz*win_ratio);
if any(t_sz > detector_wear_size*det_config.ratio)%如果任一维度超过了检测区域，需要扩大检测区域
    if t_sz(1) > detector_wear_size(1)*det_config.ratio
        detector_wear_size(1) = t_sz(1)/det_config.ratio * 1.5;
    else
        detector_wear_size(2) = t_sz(2)/det_config.ratio * 1.5;
    end
    warning('svm样本创建区域大小非法，进行修正');
    
end

%% 获取当前帧的SVM检测窗口的图片区域
% 重检测训练样本的窗口更大
% w_area的中心是目标中心坐标，因为超出边界的元素使用了边界元素进行填充，保证了中心坐标重合
w_area = get_trans_subwindow(im, pos, detector_wear_size, 1);%取到的区域以pos为中心

% 提取特征
switch det_config.svm_feature_type
    case 'fhog'
        feat = single(fhog(single(w_area), ...
            det_config.cell_size, det_config.fhog_orientations));
        feat(:,:,28:end) = [];
    case 'cn'
        if size(w_area,3) == 1% 如果是灰白图片，采用统计特征
            feat = get_detector_feature(w_area, det_config.nbin);
        else
            feat = im2cn(single(w_area), w2c, -2);%一定要转一下图片的数据类型
        end
    otherwise
        error('非法参数 det_config.svm_feature_type ')
end

% 按照比例进行缩放
switch det_config.use_resize
    case 'MATLAB'
        feat = imresize(feat, det_config.ratio, det_config.resize_type);
    case 'mexResize'
        feat = mexResize(feat, round(size(feat,[1,2])*det_config.ratio));
    case 'imResample' %either 'bilinear' or 'nearest'，默认是'bilinear'
        feat = imResample(feat, det_config.ratio, det_config.resize_type);
    otherwise
        error("非法参数 det_config.use_resize !");
end


sz = size(feat);
step = det_config.step;%取样步长
feature_t_size = t_sz ./ det_config.cell_size;%特征图中目标对应的大小

%% 拉伸特征的形状，维度：D*N
% 变换feat的形状，一列是一个样本，长度为：t_sz*size(feat,3)
% feat=im2colstep(double(feat), [t_sz(1:2), size(feat,3)], [step, step, size(feat,3)]);

% 如果fhog特征图拉伸到了原图大小，则按原图大小取块
feat = im2colstep(double(feat), ...
    [feature_t_size(1:2), size(feat,3)], ...%取块的大小
    [step, step, size(feat,3)]);  %取块的步长


% 转换数据类型
feat = single(feat);

%% 根据目标大小分割特征图，创建样本

%偏移量,在w_area中，作用是遍历完特征块
[xx, yy] = meshgrid(1:step:sz(2)-feature_t_size(2)+1, 1:step:sz(1)-feature_t_size(1)+1);

weights = fspecial('gaussian',size(xx), det_config.filter_sigma);%高斯滤波器

% 创建的样本集位置信息：[x,y,w,h]，笛卡尔坐标系下,以w_area的起点为坐标原点(0,0)
bb_samples = [xx(:), yy(:), ...
    ones(numel(xx),1)*feature_t_size(2), ones(numel(xx),1)*feature_t_size(1)];%二维矩阵

% 目标的位置，用笛卡尔坐标系表:[x,y,w,h]
% 传入的坐标是MATLAB坐标：[y0,x0,h,w]，需要转化
% 目标所在的位置[x,y,w,h]，以w_area的起点为坐标原点(0,0)
bb_target = [(sz(2)-feature_t_size(2))/2, (sz(1)-feature_t_size(1))/2,...
    feature_t_size(2), feature_t_size(1)];

labels = get_iou(bb_samples, bb_target);%得到重叠区域的比例

% 把样本坐标转到 ==中心坐标== 表示
% xy -sz/2           ====>>>> 把样本偏移量转为在w_area中以目标中心为原点的坐标
% xy -sz/2 + t_sz/2  ====>>>> 把上面的转为中心坐标表示，参考坐标原点为目标中心，在w_area中
% ====>>>> 把样本偏移量转为在w_area中以目标中心为原点的坐标
% yy=(yy+t_sz(1)/2-sz(1)/2)/det_config.ratio;%坐标原点在目标中心
% yy=yy(:)+pos(1);%坐标原点在起始点

yy = det_config.cell_size * yy - (det_config.cell_size-1)/2 ;%转到了w_area同大小的偏移量上了
yy = (yy + t_sz(1)/2 - sz(1)*det_config.cell_size/2) / det_config.ratio;
yy=yy(:)+pos(1);%坐标原点在起始点

% xx=(xx+t_sz(2)/2-sz(2)/2)/det_config.ratio;
% xx=xx(:)+pos(2);

xx = det_config.cell_size * xx - (det_config.cell_size-1)/2 ;%转到了w_area同大小的偏移量上了
xx = (xx + t_sz(2)/2 - sz(2)*det_config.cell_size/2) / det_config.ratio;
xx=xx(:)+pos(2);

pos_samples = [yy' ; xx'];%这个是样本的位置信息，这个是****中心坐标*****信息

im_sz = det_config.image_sz;%整幅图的大小

%% 删除超出边界的样本坐标
idx=yy>im_sz(1) | yy<0 | ...
    xx>im_sz(2) | xx<0;

% %不要超出图像的样本，比上面的条件更严格
% target_sz = det_config.target_sz;
% idx=yy>(im_sz(1)-target_sz(1)) | yy<(0+target_sz(1)) | ...
%     xx>(im_sz(2)-target_sz(2)) | xx<(0+target_sz(2));

% 删除非法区域
feat(:, idx)=[];
pos_samples(:, idx)=[];
labels(idx)=[];
weights(idx)=[];


%% 根据标签labels产生SVM分类器标签
if isgetsamples
    % 产生正负样本，还有部分是舍弃的，并非所有的样本都使用了
    idx_p = labels > det_config.thresh_p;%正样本
    idx_n = labels < det_config.thresh_n;%负样本

    %选取符合要求的样本作为数据集，部分会被舍弃
    feat=feat(:, idx_p|idx_n);

    % 设置样本位置信息
    pos_samples = pos_samples(:, idx_p|idx_n);

    %设置标签，正样本设为1，负样本设为-1
    labels(idx_p)=1;
    labels(idx_n)=-1;
    %只取正负样本的集合
    labels=labels(idx_p|idx_n);

    % 设置权重
    weights = weights(idx_p|idx_n);
end

end



function iou = get_iou(r1,r2)
%计算样本与目标的重叠比[x,y,w,h]
% r1为区域信息
% r2为目标所在的位置
if size(r2,1)==1
    r2=r2(ones(1, size(r1,1)),:);
end

left = max((r1(:,1)),(r2(:,1)));%重叠区域的左边的垂直于x轴的边界线
top = max((r1(:,2)),(r2(:,2)));%重叠区域的上边的垂直于y轴的边界线
right = min((r1(:,1)+r1(:,3)),(r2(:,1)+r2(:,3)));%重叠区域的右边的垂直于x轴的边界线
bottom = min((r1(:,2)+r1(:,4)),(r2(:,2)+r2(:,4)));%重叠区域的下边的垂直于y轴的边界线
ovlp = max(right - left,0).*max(bottom - top, 0);
iou = ovlp./(r1(:,3).*r1(:,4)+r2(:,3).*r2(:,4)-ovlp);

end