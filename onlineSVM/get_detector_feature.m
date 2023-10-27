function [ feat ] = get_detector_feature( I, nbin )
%计算图像块I的16维特征
if ~ismatrix(I) && ~isequal(I(:,:,1),I(:,:,2),I(:,:,3))
    I = uint8(255*RGB2Lab(I));%Lab色域，还是三通道
%     I = uint8(255*rgb2lab(I));%Lab色域，还是三通道
    nth=4;
else % gray image    
    I=I(:,:,1);
    nth=8;
end
thr = (1/(nth+1):1/(nth+1):1-1/(nth+1))*255;%长度为4

ksize = 4;
f_iif = 255-calcIIF(I(:,:,1),[ksize ksize],nbin);%二维矩阵

f_chn = cat(3,f_iif, I);%拼接成4个通道，一个三维矩阵

feat = zeros(size(f_chn,1), size(f_chn, 2), nth*size(f_chn,3));%16个通道

for ii = 1:size(f_chn,3)%共3个通道
    t_chn = f_chn(:,:,ii);%取出一个通道的二维矩阵
    t_chn = t_chn(:,:,ones(1,nth));%重新构成4个通道，复制拓展而来
    t_chn = bsxfun(@gt, t_chn, reshape(thr, 1, 1, nth));%每个通道分别求出大于thr阈值的索引    
    feat(:,:,(ii-1)*nth+1:ii*nth) = t_chn;%这是4个通道的矩阵，一同赋值过去
end

end

