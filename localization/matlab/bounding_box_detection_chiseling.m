close all
clear
plotting = true;

% read the image
source_file = "1_016_Movie2D_heatmap";
root = "F:\SSL\experiment_3d\output\chiseling\heatmaps\" + source_file + "_extracted_frames\";
files = dir(root);
files = files(~[files.isdir]);
predictions = zeros(length(files)-2, 4);
count = 0;

for k = 1:length(files)

    img = imread(strcat(root, files(k).name));
    % disp(files(k).name)

    % get frame ID
    [path, filename, ext] = fileparts(files(k).name);
    frame_id = filename(end-4:end);
    frame_id = str2double(frame_id);

    for i = 1:1920
        for j = 1:1080
            if j<22
                img(j,i,:) = [255,255,255];
            elseif i<242
                img(j,i,:) = [255,255,255];
            elseif i>1600
                img(j,i,:) = [255,255,255];
            elseif j>1035
                img(j,i,:) = [255,255,255];
            end
        end
    end

    % filter image 
    filtered = zeros(size(img, 1), size(img, 2), 1);
    for i = 1:size(img, 1)
        for j = 1:size(img, 2)
           if (img(i, j, 1) > 200 && img(i, j, 2) < 140 && img(i, j, 3) > 198)
               filtered(i, j) = 1;
           else
               filtered(i, j) = 0;
           end
        end
    end
    
    % compute bounding boxes
    filtered = logical(filtered);
    cc = bwconncomp(filtered);
    props = regionprops(cc, 'BoundingBox');

    % select the bounding box with largest y coordinate
%     y = zeros(length(props), 1);
%     for m = 1:length(y)
%         y(m) = norm(props(m).BoundingBox(2));
%     end
%     [M, I] = max(y);
%     boundingBox = props(I).BoundingBox;

    % select largest bounding box
    vector_norms = zeros(length(props), 1);
    for m = 1:length(vector_norms)
        vector_norms(m) = norm(props(m).BoundingBox(3:4));
    end
    [M, I] = max(vector_norms);

    if ~isempty(props)

        boundingBox = props(I).BoundingBox;
    
        predictions(k, :) = boundingBox;
        
        % bounding box center
        % x = floor(boundingBox(1) + boundingBox(3) / 2);
        % y = floor(boundingBox(2) + boundingBox(4) / 2);
        
        disp([frame_id, boundingBox])

        if k == 1
            result = [frame_id, boundingBox];
        else
            try
                result = [result; frame_id, boundingBox];
            catch
                result = [frame_id, boundingBox];
            end
        end
    
        if plotting == true
            
            h = figure('Position', [50 150 1800 600]);
        
            % show original heatmap
            subplot(1,2,1)
            imshow(img)
            hold on
            rectangle('Position', boundingBox, 'EdgeColor', 'g', 'LineWidth', 4);
        
            % filtered heatmap
            subplot(1,2,2)
            imshow(filtered)
            hold on
            rectangle('Position', boundingBox, 'EdgeColor', 'g', 'LineWidth', 4);       
       
        
            hold off; 
            waitfor(h)
        end
    else
        disp("No bounding box detected")
        count = count + 1;
        if plotting
            h = figure();
            imshow(img)
            waitfor(h)
        end
    end
end

disp(strcat(num2str(length(files)), " files processed"))
disp(strcat(num2str(length(files)-count), " successful detections"))
disp(strcat(num2str(count), " empty detections"))

writematrix(result, "results/" + source_file + ".csv");

