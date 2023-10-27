function im_patch = get_trans_subwindow(im, pos, window_sz, currentScaleFactor)
% 得到以pos[y0,x0]为中心，上下左右拓展选取的区域
%   此处提供详细说明
%% 根据尺度滤波器计算得到的缩放因子currentScaleFactor，得到当前目标的大小
patch_sz = floor(window_sz * currentScaleFactor);

%make sure the size is not to small
if patch_sz(1) < 1
    patch_sz(1) = 2;
end
if patch_sz(2) < 1
    patch_sz(2) = 2;
end

xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);
ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);

%如果超出了边界，复制边界值
% check for z1-of-bounds coordinates, and set them to the values at
% the borders
xs(xs < 1) = 1;
ys(ys < 1) = 1;
xs(xs > size(im,2)) = size(im,2);
ys(ys > size(im,1)) = size(im,1);

% extract image
im_patch = im(ys, xs, :);

%% 调整图片的大小，虽然在图片中采样的大小是根据缩放因子来的，
% 但是因为位置滤波器的大小不变，所以要把采样得到的图片修改成于初始目标大小一致
%
% resize image to model size，如果缩放因子为1，不需要缩放了
if currentScaleFactor ~= 1
    im_patch = imResample(im_patch, window_sz);%默认使用['bilinear'] 'bilinear' or 'nearest'
    % im_patch = mexResize(im_patch, window_sz, 'auto');
    % im_patch = imresize(im_patch, window_sz);%调用MATLAB自带的速度会慢很多，准确率高一点点
end
end