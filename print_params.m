function print_params(params)
fprintf('\n=================================\n');
fprintf(['参数设置：'...
    '\n 位置滤波器1===： 特征类型： %s 学习率：%f'...
    '\n 位置滤波器2===： 特征类型： %s 学习率：%f'...
    '\n translation_model_max_area：%f  \n  scale_model_max_area: %f\n'...
    'rho： %f \n output_sigma_factor：%f\n scale_sigma_factor: %f\n'],...
    params.features_x1.feature_type,params.kernel_x1.interp_factor,...
    params.features_x2.feature_type,params.kernel_x2.interp_factor,...
    params.translation_model_max_area, params.scale_model_max_area,...
    params.rho, params.output_sigma_factor,params.scale_sigma_factor);
fprintf('采用的核函数：===>>>  位置滤波器1： %s,  位置滤波器2： %s\n',...
    params.kernel_x1.type,params.kernel_x2.type);

fprintf('尺度滤波器===：  特征类型： %s  学习率：%f\n',...
    params.features_s.feature_type,params.learning_rate_s);

fprintf('位置滤波器===：  响应最值阈值： %f  置信度阈值：%f\n',...
    params.G_RATIO_MIN_x, params.C_RATIO_MIN_x);

fprintf('尺度滤波器===：  响应最值阈值： %f  置信度阈值：%f\n',...
    params.G_RATIO_MIN_s,params.C_RATIO_MIN_s);

fprintf('图像大小限制：%f\n',...
    params.MAX_IMG_SIZE);
end