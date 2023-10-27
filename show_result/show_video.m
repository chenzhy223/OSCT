function update_visualization_func = show_video(img_files, video_path, resize_image, ratio, video_name)
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

%create window
[fig_h, axes_h, unused, scroll] = videofig(num_frames, @redraw, [], [], @on_key_press);  %#ok, unused outputs

set(fig_h, 'Name', ['ORCT-Tracker - ' video_name]);
axis off;

%image and rectangle handles start empty, they are initialized later
im_h = [];
rect_h = [];
my_text_handle = [];%需要在这里创建一个变量

update_visualization_func = @update_visualization;
stop_tracker = false;


    function stop = update_visualization(frame, box)
        %store the tracker instance for one frame, and show it. returns
        %true if processing should stop (user pressed 'Esc').
        boxes{frame} = box;
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
        if isempty(rect_h) %create it for the first time
            rect_h = rectangle('Position',[0,0,1,1], 'EdgeColor','g', 'Parent',axes_h,'LineWidth',2);
        end
        if ~isempty(boxes{frame})
            set(rect_h, 'Visible', 'on', 'Position', boxes{frame});
        else
            set(rect_h, 'Visible', 'off');
        end
        % 显示帧数
        %         text(20, 20, int2str(frame), 'color', [0 1 1],'FontSize',20);
        if isempty(my_text_handle)
            my_text_handle = text(20, 20, int2str(frame), 'color', [1 0 0],'FontSize',20, 'Parent',axes_h);%需要设置parent元素
        end
        if ~isempty(boxes{frame})
            set(my_text_handle, 'Visible', 'on','String',int2str(frame));
        else
            set(my_text_handle, 'Visible', 'off');
        end

    end

    function on_key_press(key)
        if strcmp(key, 'escape')  %stop on 'Esc'
            stop_tracker = true;
        end
    end

end

