close all
clear
plot = true;

% read the image
root = "F:\datasets\SurgicalSSL\sawing\drilling_cropped_v2\";
files = dir(root);
predictions = zeros(length(files)-2, 4);
overlap_vec = zeros(length(files)-2, 1);

for k = 1:((length(files)-2))

    number = num2str(k);
    if length(number) == 1
        number = "00" + number;
    elseif length(number) == 2
        number = "0" + number;
    end
    
    ssl_heatmap = root + number + ".jpg";
    target_img = "F:\datasets\SurgicalSSL\sawing\drilling_cropped_gt\" + number + ".jpg";
    img = imread(ssl_heatmap);
    
    % filter image 
    filtered = zeros(size(img, 1), size(img, 2), 1);
    for i = 1:size(img, 1)
        for j = 1:size(img, 2)
           if (img(i, j, 2) < 252 && img(i, j, 1) < 252 && img(i, j, 3) < 252 || ...
                   img(i, j, 2) == 255 && img(i, j, 1) < 255 && img(i, j, 3) < 250 || ...
                   img(i, j, 2) < 255 && img(i, j, 1) == 255 && img(i, j, 3) < 250 || ...
                   img(i, j, 2) < 255 && img(i, j, 1) < 255 && img(i, j, 3) == 250)
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

    % select largest bounding box
    vector_norms = zeros(length(props), 1);
    for m = 1:length(vector_norms)
        vector_norms(m) = norm(props(m).BoundingBox(3:4));
    end
    [M, I] = max(vector_norms);
    boundingBox = props(I).BoundingBox;
    predictions(k, :) = boundingBox;

    % GT bounding box
    load('labels_v2_2.mat')
    labels = cell2mat(table2array(gTruth.LabelData));
    overlap = bboxOverlapRatio(labels(k, :), predictions(k,:));
    overlap_vec(k) = overlap;
    disp("Sample " + number + " - Overlap Ratio: " + num2str(overlap));

    if plot == true
        
        h = figure('Position', [50 150 1800 600]);
    
        % show original heatmap
        subplot(1,3,1)
        imshow(img)
%         rectangle('Position', boundingBox, 'EdgeColor', 'g', 'LineWidth', 4);
%         rectangle('Position', labels(k, :), 'EdgeColor', 'r', 'LineWidth', 4); 
    
        % filtered heatmap
        subplot(1,3,2)
        imshow(filtered)
        rectangle('Position', boundingBox, 'EdgeColor', 'g', 'LineWidth', 4);
%         rectangle('Position', labels(k, :), 'EdgeColor', 'r', 'LineWidth', 4); 
    
        % rgb image
        subplot(1,3,3)
        target_img = imread(target_img);
        imshow(target_img)
        hold on; 
        rectangle('Position', boundingBox, 'EdgeColor', 'g', 'LineWidth', 4); 
        rectangle('Position', labels(k, :), 'EdgeColor', 'r', 'LineWidth', 4); 
     
        hold off; 
        waitfor(h)
    end
end

disp("#########################")
disp("PERFORMANCE METRICS")
disp("Mean IOU: " + num2str(mean(overlap_vec)) + ", Std IOU : " + num2str(std(overlap_vec)))
disp("mAP at 0.5 IOU: " + num2str(sum(overlap_vec > 0.49)/length(overlap_vec)))
histogram(overlap_vec, 25)
xlabel('Intersection over Union')
ylabel('Number of samples')

