function z1 = get_trans_feature(im_patch, features_x, w2c, img_resize)
%get_trans_feature 提取图像的特征
%   此处提供详细说明

cell_size = features_x.cell_size;

switch features_x.feature_type
    % gray特征
    case "gray"
        %gray-level (scalar feature)
        if size(im_patch,3)>1
            im_patch = rgb2gray(im_patch);
        end
        z1 = single(im_patch) / 255;
        z1 = single(z1 - mean(z1(:)));

    case "fhog"
        %调用Piotr's Toolbox里边的函数，求解FHOG特征
        z1 = single(fhog(single(im_patch), cell_size, features_x.fhog_orientations));
        z1(:,:,end) = [];  %remove all-zeros channel ("truncation feature")

    case "gfhog"
        %调用Piotr's Toolbox里边的函数，求解FHOG特征
        if size(im_patch,3)>1
            im_patch = rgb2gray(im_patch);
        end
        z1 = fhog(single(im_patch), cell_size, features_x.fhog_orientations);
        z1(:,:,end) = [];  %remove all-zeros channel ("truncation feature")

    case 'cn'
        %返回11维度的颜色概率，也即cn特征
        %先把图像根据cell_size进行压缩，保证大小和fhog特征的大小一致，方便特征融合
        %     im_patch = imresize(im_patch,floor(window_sz./cell_size));
        %% 方法1：对图像进行缩放：
        %     im_patch = mexResize(im_patch, floor(window_sz./cell_size), 'auto');
        %%
        % 如果是灰度图，直接使用灰度信息也没差多少
        if size(im_patch,3)==1
            z1 = single(im_patch) / 255;
            z1 = single(z1 - mean(z1(:)));
        else
            %如果没有加载进映射矩阵，重新加载
            if isempty(w2c)
                data = load("w2c.mat");
                w2c = data.w2c;
                clear data;%释放变量
            end
            %         z1 = zeros(size(im_patch, 1), size(im_patch, 2), size(w2c,2), 'single');
            z1 = im2cn(single(im_patch), w2c, -2);%一定要转一下图片的数据类型
        end
        %% 方法2：利用积分图计算Cell平均：
        % 需要注释掉get_trans_subwindow文件中33行的 
        % im_patch = mexResize(im_patch, floor(window_sz./cell_size), 'auto');
        z1=feature_inter(z1, cell_size);

    
    case 'dsst'
%         temp = fhog(single(im_patch), cell_size, features_x.fhog_orientations);%设置binsize=1，比较小
%         z1 = zeros(size(temp, 1), size(temp, 2), 28, 'single');
%         z1(:,:,2:28) = temp(:,:,1:27);


%         z1 = zeros(size(temp, 1), size(temp, 2), 32, 'single');
%         z1(:,:,1:31) = temp(:,:,1:31);

        z1 = single(fhog(single(im_patch), cell_size, features_x.fhog_orientations));
        z1(:,:,29:end) = [];
        
        % 对图片进行压缩
%         im_gray = mexResize(im_patch, [size(temp,1),size(temp,2)], 'auto');
%         im_gray = imResample(im_patch, [size(temp,1),size(temp,2)]);

        switch img_resize % 'imResample' | 'MATLAB' |'mexResize'
            case 'mexResize'
                im_gray = mexResize(im_patch, [size(z1,1),size(z1,2)], 'auto');
            case 'imResample'
                im_gray = imResample(im_patch, [size(z1,1),size(z1,2)]);
            case 'MATLAB'
                im_gray = imresize(im_patch, [size(z1,1),size(z1,2)]);
        end

        % if grayscale image
        if size(im_patch, 3) == 1
            z1(:,:,end) = single(im_gray)/255 - 0.5;
        else
            z1(:,:,end) = single(rgb2gray(im_gray))/255 - 0.5;%处理成灰度
        end
        
    case 'fhogcn'
        %% 方法1：对图像进行缩放：
        %     im_patch = mexResize(im_patch, floor(window_sz./cell_size), 'auto');
        %%
        temp_fhog = single(fhog(single(im_patch), cell_size, features_x.fhog_orientations));
        % 如果是灰度图
        if size(im_patch,3)==1
            temp_cn = single(im_patch) / 255;
            temp_cn = single(temp_cn - mean(temp_cn(:)));
        else
            %如果没有加载进映射矩阵，重新加载
            if isempty(w2c)
                data = load("w2c.mat");
                w2c = data.w2c;
                clear data;%释放变量
            end
            %         z1 = zeros(size(im_patch, 1), size(im_patch, 2), size(w2c,2), 'single');
            temp_cn = im2cn(single(im_patch), w2c, -2);%一定要转一下图片的数据类型
        end
        %% 方法2：利用积分图计算Cell平均：
        %%     需要注释掉93行的im_patch = mexResize(im_patch, floor(window_sz./cell_size), 'auto');
        
        
        temp_cn=feature_inter(temp_cn, cell_size);
  
        z1 = cat(3,temp_fhog(:,:,1:27),temp_cn);

     case 'cnhist'
        % 如果是灰度图
        if size(im_patch,3)==1
            temp_cn = single(im_patch) / 255;
            temp_cn = single(temp_cn - mean(temp_cn(:)));
        else
            %如果没有加载进映射矩阵，重新加载
            if isempty(w2c)
                data = load("w2c.mat");
                w2c = data.w2c;
                clear data;%释放变量
            end
            temp_cn = im2cn(single(im_patch), w2c, -2);%一定要转一下图片的数据类型
        end

        % pixel intensity histogram, from Piotr's Toolbox
        h1=histcImWin(rgb2gray(im_patch),features_x.hist_orientations,...
            ones(features_x.window_size,features_x.window_size),'same'); 

        result = cellfun(@feature_inter,{temp_cn,h1},{cell_size,cell_size},...
            "UniformOutput",false);
        % 拼接特征
        z1 = cat(3,result{2},result{1});

    case 'fhoghist'
        result1 = single(fhog(single(im_patch), cell_size, features_x.fhog_orientations));
        result1(:,:,end) = [];  %remove all-zeros channel ("truncation feature")
        
        % pixel intensity histogram, from Piotr's Toolbox
        h1=histcImWin(rgb2gray(im_patch),features_x.hist_orientations,...
            ones(features_x.window_size,features_x.window_size),'same'); 

        result2 = feature_inter(h1,cell_size);
        % 拼接特征
        z1 = cat(3,result1,result2);

    otherwise
        error('Unknown feature.')

end
end

%% 内部函数，实现积分图像块求平均
function temp=feature_inter(temp, cell_size)
    %compute the integral image计算积分图像
    iImage = integralVecImage(temp);%在左边和上边有一圈0补充，即大小都+1
    %要+1，是因为积分图会用0补充多一个像素点
    i1 = (cell_size:cell_size:size(temp,1)) + 1;
    i2 = (cell_size:cell_size:size(temp,2)) + 1;
    %利用积分图进行计算平均
    temp_cn_sum = iImage(i1,i2,:) - iImage(i1,i2-cell_size,:) ...
        - iImage(i1-cell_size,i2,:) + iImage(i1-cell_size,i2-cell_size,:);
    temp = temp_cn_sum / (cell_size*cell_size) ;

end