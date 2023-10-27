function responsef = get_translation_responsef(im, pos, window_sz, currentScaleFactor, ...
    cos_window, features_x, w2c,model_xf, model_alphaf, kernel_x, img_resize)
% 本模块针对不使用降维的位置滤波器编写，计算位置滤波器的相应输出

if isscalar(window_sz)  %square sub-window
    window_sz = [window_sz, window_sz];
end

% 得到含背景信息的感兴趣区域
im_patch = get_trans_subwindow(im, pos, window_sz, currentScaleFactor);

% 对感兴趣区域进行特征提取
z1 = get_trans_feature(im_patch, features_x, w2c, img_resize);

% 统一数据格式为single
z1 = single(z1);

% 加窗处理
if ~isempty(cos_window)
    %用bsxfun函数进行指定的操作,可以防止内存超出
    z1 = bsxfun(@times, z1, cos_window);
end


%% 计算相关
z1f = single( fft2(z1) );%转为单精度


%calculate response of the classifier at all shifts
switch kernel_x.type
    case 'gaussian'
        kz1f = gaussian_correlation(z1f, model_xf, kernel_x.sigma);
    case 'polynomial'
        kz1f = polynomial_correlation(z1f, model_xf, kernel_x.poly_a, kernel_x.poly_b);
    case 'linear'
        kz1f = linear_correlation(z1f, model_xf);
end


%这个响应最大值是分布在四个角落的，没有用shift经行移位
responsef = bsxfun(@times, model_alphaf, kz1f);

end