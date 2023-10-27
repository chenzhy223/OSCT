function [x1f,alpha1f]=get_translation_param(im, pos, window_sz, currentScaleFactor, cos_window, features_x, w2c,kernel_x,yf,lambda ,img_resize)
%% 本模块针对不使用降维方法的位置滤波器编写
%% 得到特征图，就是get_translation_sample的内容
%     x1 = get_translation_sample(im, pos, window_sz, currentScaleFactor, cos_window1, features_x1, w2c);
%如果是标量，改成向量格式
if isscalar(window_sz)  %square sub-window
    window_sz = [window_sz, window_sz];
end


% 得到含背景信息的感兴趣区域
im_patch = get_trans_subwindow(im, pos, window_sz, currentScaleFactor);


% 对感兴趣区域进行特征提取
x1 = get_trans_feature(im_patch, features_x, w2c, img_resize);


%% 统一数据格式为single
x1 = single(x1);

%process with cosine window if needed
if ~isempty(cos_window)
    %用bsxfun函数进行指定的操作,可以防止内存超出
    x1 = bsxfun(@times, x1, cos_window);
end

%% 计算剩下的
x1f = single(fft2(x1));%转为单精度

%Kernel Ridge Regression, calculate alphas (in Fourier domain)
%计算核矩阵Kxx，并FFT（即计算核函数，采用三种核）
switch kernel_x.type
    case 'gaussian'
        k1f = gaussian_correlation(x1f, x1f, kernel_x.sigma);
    case 'polynomial'
        k1f = polynomial_correlation(x1f, x1f, kernel_x.poly_a, kernel_x.poly_b);
    case 'linear'
        k1f = linear_correlation(x1f, x1f);
end
alpha1f = yf ./ (k1f + lambda);   %equation for fast training
end