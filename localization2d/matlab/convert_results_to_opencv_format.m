% process_csvs.m
% This script loads all CSV files in the current folder,
% subtracts 1 from columns 2–5,
% and saves the modified files in "results_export"
% with "_heatmap" removed from their filenames.

clc; clear;

% Folder containing the CSV files
inputFolder = "results";  % current folder
outputFolder = fullfile('results_export');

% Create the output folder if it doesn’t exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Get all CSV files in the folder
files = dir(fullfile(inputFolder, '*.csv'));

% Loop through each CSV file
for k = 1:length(files)
    inputFile = fullfile(inputFolder, files(k).name);
    fprintf('Processing: %s\n', files(k).name);

    % Read CSV file
    data = readmatrix(inputFile);

    % Check for enough columns
    if size(data, 2) >= 5
        % Subtract 1 from columns 2–5
        data(:, 2:3) = data(:, 2:3) - 1;
    else
        warning('File %s has fewer than 5 columns. Skipped.', files(k).name);
        continue;
    end

    % Create output filename (remove "_heatmap")
    cleanName = strrep(files(k).name, '_heatmap', '');
    outputFile = fullfile(outputFolder, cleanName);

    % Write the modified data
    writematrix(data, outputFile);
end

disp('✅ All CSV files processed and saved in "results_export".');
