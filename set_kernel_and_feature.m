function [kernel,features] = set_kernel_and_feature(kernel,features)
% 根据选取的特征，设置kernel、feature参数。
% 根据输入的feature_type来确定所选的特征，然后对应设置好参数
switch features.feature_type
    case 'gray'
        %自适应线性插值因子，就是模型学习率
        kernel.interp_factor = 0.075;  %linear interpolation factor for adaptation
        %计算高斯核相关矩阵时的标准差
        kernel.sigma = 0.2;  %gaussian kernel bandwidth
        %设置多项式核参数，加法项核乘法项
        kernel.poly_a = 1;  %polynomial kernel additive term
        kernel.poly_b = 7;  %polynomial kernel exponent
        
        features.cell_size = 1;
        
    case 'fhog'
        kernel.interp_factor = 0.04;%学习率
        kernel.sigma = 0.5;
        
        kernel.poly_a = 1;
        kernel.poly_b = 9;
        
        features.fhog_orientations = 9;%方向个数参数
        %这个参数不能设的太大，8不行，6一般般
        features.cell_size = 4;%一个Cell的大小，设大了速度会提高，但是精确度会下降，默认是4
        
    case 'gfhog'
        kernel.interp_factor = 0.04;%学习率
        kernel.sigma = 0.5;
        
        kernel.poly_a = 1;
        kernel.poly_b = 9;
        
        features.fhog_orientations = 9;%方向个数参数
        %这个参数不能设的太大，8不行，6一般般
        features.cell_size = 4;%一个Cell的大小，设大了速度会提高，但是精确度会下降，默认是4
        
    case 'cn'
        kernel.interp_factor = 0.02;%学习率
        kernel.sigma = 0.5;
        
        kernel.poly_a = 1;
        kernel.poly_b = 9;
        
        %这个参数不能设的太大，8不行，6一般般
        features.cell_size = 4;%一个Cell的大小，设大了速度会提高，但是精确度会下降，默认是4
    case 'dsst'
        kernel.interp_factor = 0.04;%学习率
        kernel.sigma = 0.5;
        
        kernel.poly_a = 1;
        kernel.poly_b = 9;
        
        %这个参数不能设的太大，8不行，6一般般
        features.cell_size = 4;%dsst中默认采用1
        features.fhog_orientations = 9;%方向个数参数
        
    case 'fhogcn'
        kernel.interp_factor = 0.02*2;%学习率
        kernel.sigma = 0.5;
        
        kernel.poly_a = 1;
        kernel.poly_b = 9;
        
        features.fhog_orientations = 9;%方向个数参数
        %这个参数不能设的太大，8不行，6一般般
        features.cell_size = 4;

     case 'cnhist'
        kernel.interp_factor = 0.02;%学习率
        kernel.sigma = 0.5;
        
        kernel.poly_a = 1;
        kernel.poly_b = 9;
        
        features.hist_orientations = 8;%方向个数参数
        features.window_size = 6;%局部像素的大小
        %这个参数不能设的太大，8不行，6一般般
        features.cell_size = 4;

     case 'fhoghist'
        kernel.interp_factor = 0.04;%学习率
        kernel.sigma = 0.5;
        
        kernel.poly_a = 1;
        kernel.poly_b = 9;
        
        features.fhog_orientations = 9;%方向个数参数
        %这个参数不能设的太大，8不行，6一般般
        features.cell_size = 4;%一个Cell的大小，设大了速度会提高，但是精确度会下降，默认是4
        features.hist_orientations = 8;%方向个数参数
        features.window_size = 6;%局部像素的大小

    otherwise
        error('Unknown feature.')
end
%异常处理
assert(any(strcmp(kernel.type, {'linear', 'polynomial', 'gaussian'})), 'Unknown kernel.')
end
