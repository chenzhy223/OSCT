if params.show_TempPlots
    %% 绘制加权系数
    figure("Name","位置滤波器加权系数")
    plot((1:size(lambda_fit_ratio(:,1),1))+1,lambda_fit_ratio(:,1), 'k-', 'LineWidth',2)
    hold on;%%在该图基础上继续画图
    plot((1:size(lambda_fit_ratio(:,1),1))+1,lambda_fit_ratio(:,2), 'g-', 'LineWidth',2)
    legend('位置滤波器1','位置滤波器2');%%用图例标识曲线
    xlabel('帧数'), ylabel('权重')
    title('位置滤波器加权系数')
    hold off
    xticks('auto');
    xlim([2,size(lambda_fit_ratio(:,1),1)+1])



    %% 可视化置信度
    figure('Name','置信度原始大小')
    plot((1:size(trans_c_list(:,1),1))+1,trans_c_list(:,1), 'k-', 'LineWidth',2)
    hold on
    plot((1:size(trans_c_list(:,2),1))+1,trans_c_list(:,2), 'g-', 'LineWidth',2)
    hold off
    legend('位置滤波器1','位置滤波器2');%%用图例标识曲线
    xlim([2,size(trans_c_list(:,2),1)+1])
    xlabel('帧数'), ylabel('置信度')
    title('置信度')
% 
% 
%     %% 绘制滤波器想要最值的比例情况
%     figure("Name","滤波器响应最值比例")
%     plot((1:size(G_RATIO_ALL(:,1),1))+1,G_RATIO_ALL(:,1), 'k-', 'LineWidth',2)
%     hold on;%%在该图基础上继续画图
%     plot((1:size(G_RATIO_ALL(:,1),1))+1,G_RATIO_ALL(:,2), 'g-', 'LineWidth',2)
%     hold on
%     plot((1:size(G_RATIO_ALL(:,1),1))+1, ones(1,size(G_RATIO_ALL(:,1),1))*params.G_RATIO_MIN_x,'r--', 'LineWidth',2.5)
%     legend('位置滤波器','尺度滤波器', '阈值');%%用图例标识曲线
%     xlabel('帧数'), ylabel('响应最值比例')
%     xlim([2,size(G_RATIO_ALL(:,1),1)+1])
%     title('滤波器响应最值比例')
%     xticks('auto');
%     hold off
% 
%     %% 绘制滤波器的置信度比例
%     figure("Name","滤波器置信度比例")
%     plot((1:size(G_RATIO_ALL(:,1),1))+1,C_RATIO_ALL(:,1), 'k-', 'LineWidth',2)
%     hold on;%%在该图基础上继续画图
%     plot((1:size(G_RATIO_ALL(:,1),1))+1,C_RATIO_ALL(:,2), 'g-', 'LineWidth',2)
%     hold on
%     plot((1:size(G_RATIO_ALL(:,1),1))+1, ones(1,size(G_RATIO_ALL(:,1),1))*params.C_RATIO_MIN_x,'r--', 'LineWidth',2.5)
%     legend('位置滤波器','尺度滤波器', '阈值');%%用图例标识曲线
%     xlabel('帧数'), ylabel('置信度比例')
%     xlim([2,size(G_RATIO_ALL(:,1),1)+1])
%     hold off
%     xticks('auto');
%     title('滤波器置信度比例')
% 
% 


    %% 绘制学习率曲线
    figure('Name','滤波器学习率曲线')
    plot(trans_x1_learning_list(:,1),trans_x1_learning_list(:,2),'r-','LineWidth',1.5);
    hold on
    plot(trans_x2_learning_list(:,1),trans_x2_learning_list(:,2),'g-','LineWidth',1.5);
    hold on
    plot(app_learning_list(:,1),app_learning_list(:,2),'b:','LineWidth',1.5);
    legend('位置滤波器1学习率','位置滤波器2学习率','目标外观模型学习率');%%用图例标识曲线
    xlabel('帧数'), ylabel('学习率')
    hold off
    xticks('auto');
    xlim([min([min(trans_x1_learning_list(:,1)),min(trans_x2_learning_list(:,1)),min(app_learning_list(:,1))]),...
        max([max(trans_x1_learning_list(:,1)),max(trans_x2_learning_list(:,1)),max(app_learning_list(:,1))])])
    title('学习率')

    %% 单独绘制位置滤波器的参数
    figure("Name","位置滤波器")
    plot((1:size(G_RATIO_ALL(:,1),1))+1,G_RATIO_ALL(:,1), 'k-', 'LineWidth',2)
    hold on;%%在该图基础上继续画图
    plot((1:size(G_RATIO_ALL(:,1),1))+1,C_RATIO_ALL(:,1), 'g-', 'LineWidth',2)
    hold on
    plot((1:size(G_RATIO_ALL(:,1),1))+1, ones(1,size(G_RATIO_ALL(:,1),1))*G_RATIO_MIN_x,'r--', 'LineWidth',2.5)
    hold on
    plot((1:size(G_RATIO_ALL(:,1),1))+1, ones(1,size(G_RATIO_ALL(:,1),1))*C_RATIO_MIN_x,'r-', 'LineWidth',2.5)

    hold on
    plot((1:size(G_RATIO_ALL(:,1),1))+1, update_trans_sign(2:end),'c', 'LineWidth',1)

    legend('响应最值比例','响应置信度比例','最值阈值','置信度阈值','更新位置滤波器标志');%%用图例标识曲线
    xlabel('帧数'), ylabel('比例')
    xlim([2,size(G_RATIO_ALL(:,1),1)+1])
    hold off
    xticks('auto');
    title('位置滤波器')


    %% 单独绘制尺度滤波器参数
    figure("Name","尺度滤波器")
    plot((1:size(G_RATIO_ALL(:,2),1))+1,G_RATIO_ALL(:,2), 'k-', 'LineWidth',2)
    hold on;%%在该图基础上继续画图
    plot((1:size(G_RATIO_ALL(:,2),1))+1,C_RATIO_ALL(:,2), 'g-', 'LineWidth',2)
    hold on
    plot((1:size(G_RATIO_ALL(:,1),1))+1, ones(1,size(G_RATIO_ALL(:,1),1))*G_RATIO_MIN_x,'r--', 'LineWidth',2.5)
    hold on
    plot((1:size(G_RATIO_ALL(:,1),1))+1, ones(1,size(G_RATIO_ALL(:,1),1))*C_RATIO_MIN_x,'r-', 'LineWidth',2.5)
    legend('响应最值比例','响应置信度比例','最值阈值','置信度阈值');%%用图例标识曲线
    xlabel('帧数'), ylabel('比例')
    xlim([2,size(G_RATIO_ALL(:,2),1)+1])
    hold off
    xticks('auto');
    title('尺度滤波器')


    if params.is_support_SVM
        %% 绘制目标外观模型app的响应参数
        figure("Name","目标外观模型响应")
        plot((1:numel(app_response_list)) +1 ,app_response_list./MAX_APP_RESPONSE, 'b-', 'LineWidth',2)
        hold on
        plot((1:numel(app_response_list)) +1, ones(1,numel(app_response_list))*threshold_detector,'r--', 'LineWidth',2.5)
        hold on
        plot((1:numel(app_response_list)) +1, ones(1,numel(app_response_list))*threshold_train_SVM,'r-.', 'LineWidth',2.5)
        hold on
        plot((1:numel(app_response_list)) +1, ones(1,numel(app_response_list))*threshold_updateapp,'r:', 'LineWidth',2.5)

        hold on
        plot((1:numel(detector_sign)) , detector_sign,'m--', 'LineWidth',2.5)
        hold on
        plot((1:numel(trainSVM_sign)), trainSVM_sign,'c-.', 'LineWidth',2.5)
        hold on
        plot((1:numel(trainSVM_sign)), update_app_model,'y:', 'LineWidth',2.5)
        legend('最值/max','重检测阈值','更新SVM阈值','更新app阈值', ...
            '重检测标识','更新SVM标识','更新app标识');%%用图例标识曲线
        xlabel('帧数'), ylabel('响应指标')
        xlim([2,numel(app_response_list)+1])
        hold off
        xticks('auto');
        title('目标外观模型')


        %% 显示是否接受重检测的部分
        if ~isempty(detector_xlable)
            figure('Name','重检测响应最大值比例')
            color_list = ones(numel(pred_score_list),3) .* [1 0 0];%红色
            color_list(pred_score_list<0,:) = ones(sum(pred_score_list<0),3) .* [0 0 0];%不接受的标志，黑色
            scatter(detector_xlable,detector_re_max_response./detector_max_response,[],color_list,'filled')
            hold on
            plot(detector_xlable,detector_re_max_response,'LineWidth',2,'Color','g');
            hold on
            plot(detector_xlable,detector_max_response,'b:','LineWidth',2)
            hold on
            plot(detector_xlable,ones(size(detector_xlable))*threshold_accept_det,'LineWidth',2,'Color','r')
            legend('re\_max/max','重检测响应最大值','检测前响应最大值','SVM接受结果的阈值');
            xlabel('帧数'), ylabel('re_max/max')
            title('重检测阈值')
        end


        %% 绘制滤波器相似度
        figure("Name","滤波器参数相似度")
        plot(corr_list(:,1) , corr_list(:,2),'m--', 'LineWidth',2.5)
        hold on
        plot(corr_list(:,1), corr_list(:,3),'c-.', 'LineWidth',2.5)
        legend('trans1','trans2');%%用图例标识曲线
        xlabel('帧数'), ylabel('参数相似度')
        xlim([min(corr_list(:,1)),max(corr_list(:,1))])
        hold off
        xticks('auto');
        title('相似度')

    end
end