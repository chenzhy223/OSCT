function update_visualization_func = show_video_svm(img_files, video_path, resize_image,ratio,video_name)
%SHOW_VIDEO
%   Visualizes a tracker in an interactive figure, given a cell array of
%   image file names, their path, and whether to resize the images to
%   half size or not.
%
%   This function returns an UPDATE_VISUALIZATION function handle, that
%   can be called with a frame number and a bounding box [x, y, width,
%   height], as soon as the results for a new frame have been calculated.
%   This way, your results are shown in real-time, but they are also
%   remembered so you can navigate and inspect the video afterwards.
%   Press 'Esc' to send a stop signal (returned by UPDATE_VISUALIZATION).
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/


%store one instance per frame
num_frames = numel(img_files);
boxes = cell(num_frames,1);
boxes_tpos = cell(num_frames,1);
boxes_wear = cell(num_frames,1);


%create window
[fig_h, axes_h, unused, scroll] = videofig(num_frames, @redraw, [], [], @on_key_press);  %#ok, unused outputs
% 	set(fig_h, 'Number','off', 'Name', ['Tracker - ' video_path])
set(fig_h, 'Name', ['SVM-Detector - ' video_name]);
axis off;

%image and rectangle handles start empty, they are initialized later
im_h = [];
rect_h = [];
my_text_handle = [];%需要在这里创建一个变量
is_accpet_detector_list = [];
rect_tpos_box = [];
rect_wear_box = [];

update_visualization_func = @update_visualization;
stop_tracker = false;


    function stop = update_visualization(frame, box, tpos_box, wear_box, is_accpet_detector)
        %store the tracker instance for one frame, and show it. returns
        %true if processing should stop (user pressed 'Esc').
        boxes{frame} = box;%检测得到的位置
        boxes_tpos{frame} = tpos_box;%svm预测得到的位置
        boxes_wear{frame} = wear_box;%svm预测的区域
        is_accpet_detector_list{frame} = is_accpet_detector;%记录是否接受重检测结果
        scroll(frame);
        stop = stop_tracker;
    end

    function redraw(frame)
        %render main image
        im = imread([video_path img_files{frame}]);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %把RGB格式的图片转换为灰度图，如果希望显示的不是灰度的，可以注释掉
        % 		if size(im,3) > 1
        % 			im = rgb2gray(im);
        %         end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if resize_image
            im = imresize(im, ratio);
        end
        %         im=imresize(im,1/resize_image);

        if isempty(im_h)  %create image
            im_h = imshow(im, 'Border','tight', 'InitialMag',200, 'Parent',axes_h);
        else  %just update it
            set(im_h, 'CData', im)
        end

        %render target bounding box for this frame
        if isempty(rect_h)  %create it for the first time
            rect_h = rectangle('Position',[0,0,1,1], 'EdgeColor','b', 'Parent',axes_h,'LineWidth',2);%原目标位置
            rect_tpos_box = rectangle('Position',[0,0,1,1], 'EdgeColor','r', 'Parent',axes_h,'LineWidth',2);%重检测位置

            rect_wear_box = rectangle('Position',[0,0,1,1], 'EdgeColor','g', 'Parent',axes_h,'LineWidth',2);%重检测区域
        end
        if ~isempty(boxes{frame})
            set(rect_h, 'Visible', 'on', 'Position', boxes{frame});

            if is_accpet_detector_list{frame}
                set(rect_tpos_box, 'Visible', 'on', 'Position', boxes_tpos{frame},'LineStyle','-');
            else
                set(rect_tpos_box, 'Visible', 'on', 'Position', boxes_tpos{frame},'LineStyle','--');
            end

            set(rect_wear_box, 'Visible', 'on', 'Position', boxes_wear{frame});
        else
            set(rect_h, 'Visible', 'off');
            set(rect_tpos_box, 'Visible', 'off');
            set(rect_wear_box, 'Visible', 'off');
        end
        % 显示帧数
        %         text(20, 20, int2str(frame), 'color', [0 1 1],'FontSize',20);
        if isempty(my_text_handle)
            my_text_handle = text(20, 20, int2str(frame), 'color', [1 0 0],'FontSize',20, 'Parent',axes_h);%需要设置parent元素
        end

        set(my_text_handle, 'Visible', 'on','String',int2str(frame));

%         if ~isempty(boxes{frame})
%             set(my_text_handle, 'Visible', 'on','String',int2str(frame));
%         else
%             set(my_text_handle, 'Visible', 'off');
%         end

    end

    function on_key_press(key)
        if strcmp(key, 'escape')  %stop on 'Esc'
            stop_tracker = true;
        end
    end

end

