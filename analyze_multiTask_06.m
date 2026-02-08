
addpath([pwd filesep 'fcn_local'])

cfg.RESULTFOLDER          = '/Users/wenxuan/Documents/MATLAB/Multitasking_analysis';
cfg.RESULTFILE            = 'WM_AS_CR_FL__MultiTask';
iResultFileMetrics = [cfg.RESULTFILE '_Metrics'];
%RESULTFOLDER = [HOME_FOLDER filesep iResultFolder '_MAT']; if ~exist(RESULTFOLDER), mkdir(RESULTFOLDER),end
METRICSFOLDER  = [cfg.RESULTFOLDER '_METRICS']; if ~exist(METRICSFOLDER), mkdir(METRICSFOLDER),end
FIGURE_Folder = [cfg.RESULTFOLDER '_FIG'];    if ~exist(FIGURE_Folder), mkdir(FIGURE_Folder), end

% --- --- --- --- --- --- --- --- --- --- ---
% --- load metrics results if they are not yet loaded
% --- --- --- --- --- --- --- --- --- --- ---
if ~exist('metrics_mt')
    load([METRICSFOLDER filesep iResultFileMetrics])
	output_table = generate_behavioral_summary_table(metrics_mt);
end

% new_output_table = addSessionDuration(output_table);
% analyzeAndExportSessionTimes(new_output_table);







function analyzeAndExportSessionTimes(new_output_table)
% analyzeAndExportSessionTimes Analyzes session data, prints a summary, and exports to CSV.
%
%   This function takes a table containing session data, filters for valid entries,
%   calculates the mean and standard error (SE) of session durations for each
%   subject and overall, prints a summary to the console, and saves a
%   detailed report to a CSV file.
%
%   A session is considered INVALID if ANY of its columns contain a missing
%   value (NaN, NaT, etc.).
%
%   Syntax:
%       analyzeAndExportSessionTimes(new_output_table)
%
%   Input:
%       new_output_table - A MATLAB table that MUST contain the following columns:
%                          'SubjectID'       (string or cell array of strings)
%                          'SessionDateTime' (datetime)
%                          'SessionDuration' (numeric)

%% --- 1. Setup and Input Validation ---
fprintf('Starting session time analysis and export...\n');

% Validate that the input is a table with the required columns
requiredCols = {'SubjectID', 'SessionDateTime', 'SessionDuration'};
if ~istable(new_output_table) || ~all(ismember(requiredCols, new_output_table.Properties.VariableNames))
    error('Input must be a MATLAB table containing "SubjectID", "SessionDateTime", and "SessionDuration" columns.');
end

% Define the output directory and create it if it doesn't exist
outputDirName = 'All_Session_Time';
if ~exist(outputDirName, 'dir')
    mkdir(outputDirName);
    fprintf('Created output directory: %s\n', outputDirName);
end

%% --- 2. Filter for Valid Session Data (UPDATED LOGIC) ---
% A valid session has NO missing values in ANY column.
% We will work with a copy and remove invalid rows.
valid_sessions_table = new_output_table;

% 'ismissing' checks for NaN, NaT, <missing>, etc.
% any(..., 2) creates a logical index for rows with at least one missing value.
rows_with_missing_values = any(ismissing(valid_sessions_table), 2);

% Remove all identified invalid rows from the table.
valid_sessions_table(rows_with_missing_values, :) = [];


if isempty(valid_sessions_table)
    fprintf('No valid sessions found after filtering for missing values. Nothing to process or export.\n');
    return;
end

fprintf('Found %d valid sessions to analyze (rows with no missing values).\n\n', height(valid_sessions_table));

%% --- 3. Calculate Per-Subject Statistics ---
uniqueSubjects = unique(valid_sessions_table.SubjectID);
numSubjects = length(uniqueSubjects);

% Pre-allocate a table to store subject-level results
subjectStats = table('Size', [numSubjects, 4], ...
    'VariableTypes', {'string', 'double', 'double', 'double'}, ...
    'VariableNames', {'Subject', 'Mean', 'SE', 'Count'});

fprintf('--- Session Duration Summary ---\n');

for i = 1:numSubjects
    subjectName = uniqueSubjects{i};
    
    % Get all sessions for the current subject
    subject_sessions = valid_sessions_table(strcmp(valid_sessions_table.SubjectID, subjectName), :);
    durations = subject_sessions.SessionDuration;
    n = length(durations);
    
    % Calculate mean and standard error
    subjectMean = mean(durations);
    
    % Standard error is std / sqrt(n). Handle n=1 case to avoid NaN.
    if n > 1
        subjectSE = std(durations) / sqrt(n);
    else
        subjectSE = 0; % SE is 0 if there is only one data point
    end
    
    % Store results
    subjectStats(i, :) = {subjectName, subjectMean, subjectSE, n};
    
    % Print summary to console
    subjectInitial = subjectName(1);
    fprintf('%s: %.2f +- %.2f s (n=%d)\n', subjectInitial, subjectMean, subjectSE, n);
end

%% --- 4. Calculate Overall Statistics ---
all_durations = valid_sessions_table.SessionDuration;
overall_n = length(all_durations);
overall_mean = mean(all_durations);
overall_se = std(all_durations) / sqrt(overall_n);

fprintf('--------------------------------\n');
fprintf('Overall: %.2f +- %.2f s (n=%d)\n', overall_mean, overall_se, overall_n);

%% --- 5. Prepare and Export Data to CSV ---
% Define the full path for the output CSV file
outputFilePath = fullfile(outputDirName, 'session_time_summary.csv');

% --- Part A: Write the list of all valid sessions ---
% Select and rename columns for clarity in the CSV
data_to_export = valid_sessions_table(:, {'SubjectID', 'SessionDateTime', 'SessionDuration'});
data_to_export.Properties.VariableNames = {'Subject', 'DateTime', 'Duration_s'};

try
    writetable(data_to_export, outputFilePath);
    fprintf('\nSuccessfully saved list of valid sessions to:\n%s\n', outputFilePath);
    
    % --- Part B: Append the summary statistics to the same file ---
    
    % Create a cell array for the summary report
    summary_report = {'\n'; '--- Summary Statistics ---'}; % Start with blank lines
    summary_report{end+1, 1} = 'Subject,Statistic,Value';
    
    % Add each subject's stats
    for i = 1:numSubjects
        stats = subjectStats(i,:);
        summary_report{end+1, 1} = sprintf('%s,Mean (s),%.2f', stats.Subject, stats.Mean);
        summary_report{end+1, 1} = sprintf('%s,SE (s),%.2f', stats.Subject, stats.SE);
        summary_report{end+1, 1} = sprintf('%s,Session Count,%d', stats.Subject, stats.Count);
    end
    
    % Add overall stats
    summary_report{end+1, 1} = '---,---,---'; % Separator
    summary_report{end+1, 1} = sprintf('Overall,Mean (s),%.2f', overall_mean);
    summary_report{end+1, 1} = sprintf('Overall,SE (s),%.2f', overall_se);
    summary_report{end+1, 1} = sprintf('Overall,Total Session Count,%d', overall_n);

    % Write the summary report using low-level file I/O for appending
    fileID = fopen(outputFilePath, 'a'); % 'a' for append
    if fileID == -1
        error('Could not open file for appending: %s', outputFilePath);
    end
    
    fprintf(fileID, '%s\n', summary_report{:});
    fclose(fileID);
    
    fprintf('Successfully appended summary statistics to the file.\n');
    
catch ME
    error('Failed to write to CSV file. Error: %s', ME.message);
end

fprintf('\nAnalysis complete.\n');

end




function output_table = addSessionDuration(output_table)
%addSessionDuration Adds the session duration from .mat files to a table.
%
%   This function scans subject directories to find session files, extracts the
%   'RunBlock_EndTimeAbsolute' from each, and maps it to the session's
%   datetime (parsed from the filename). It then uses the 'SessionDateTime'
%   column in the input table to look up this duration and add it to a new
%   'SessionDuration' column.
%
%   Syntax:
%       modified_table = addSessionDuration(output_table, baseDir)
%
%   Inputs:
%       output_table - A MATLAB table that MUST contain a 'SessionDateTime'
%                      column of datetime objects.
%       baseDir      - A string specifying the path to the base directory
%                      containing the subject folders.
%
%   Output:
%       output_table - The modified table with the new 'SessionDuration' column.
%                      Rows that could not be matched will have NaN in this column.

%% --- 1. Input Validation ---
if ~istable(output_table) || ~ismember('SessionDateTime', output_table.Properties.VariableNames)
    error('Input must be a MATLAB table containing a "SessionDateTime" column.');
end
baseDir = '/Users/wenxuan/Documents/MATLAB/Multitasking_analysis/';
fprintf('Starting session duration processing...\n');

%% --- 2. Build a Map from DateTime to Session Duration ---

% The map will use a standardized string version of a datetime as its key
% and the session duration (in seconds) as its value.
dateTimeToDurationMap = containers.Map('KeyType', 'char', 'ValueType', 'double');

% Find all subject directories matching the pattern
subjectPattern = 'MUSEMAT01_WM_AS_CR_FL_083_*';
subjectDirs = dir(fullfile(baseDir, subjectPattern));
subjectDirs = subjectDirs([subjectDirs.isdir]);

if isempty(subjectDirs)
    error('No subject directories found matching the pattern in: %s', baseDir);
end

fprintf('Found %d subject folders. Reading .mat files to extract durations...\n', length(subjectDirs));

% Loop through all subjects and all their session files
for i = 1:length(subjectDirs)
    subjectFolderPath = fullfile(baseDir, subjectDirs(i).name);
    sessionFiles = dir(fullfile(subjectFolderPath, 'DAT*.mat'));

    for j = 1:length(sessionFiles)
        fileName = sessionFiles(j).name;
        sessionFilePath = fullfile(subjectFolderPath, fileName);

        % a) Parse the datetime from the filename
        token = regexp(fileName, '_(\d{2}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2})_', 'tokens', 'once');
        if isempty(token)
            fprintf('  -> Warning: No datetime pattern found in filename: %s. Skipping.\n', fileName);
            continue;
        end

        try
            file_dt = datetime(token{1}, 'InputFormat', 'MM_dd_yy__HH_mm_ss');
        catch
            fprintf('  -> Warning: Could not parse datetime from filename: %s. Skipping.\n', fileName);
            continue;
        end

        % b) Load the .mat file and extract the session duration
        try
            loadedStruct = load(sessionFilePath, 'dat');
            if isfield(loadedStruct, 'dat') && isfield(loadedStruct.dat, 'blockData') && ...
               ~isempty(loadedStruct.dat.blockData)

                lastStruct = loadedStruct.dat.blockData{end}(end);
                if isfield(lastStruct, 'RunBlock_EndTimeAbsolute')
                    sessionTime = lastStruct.RunBlock_EndTimeAbsolute;

                    % c) Filter for valid times and populate the map
                    if sessionTime >= 2000
                        % Convert datetime to a standardized string to use as a map key
                        mapKey = datestr(file_dt, 'yyyy-mm-dd HH:MM:SS');
                        dateTimeToDurationMap(mapKey) = sessionTime;
                    else
                        fprintf('  -> Excluding session %s due to invalid time: %.2f s.\n', fileName, sessionTime);
                    end
                end
            end
        catch ME
            fprintf('  -> ERROR processing file %s. Message: %s\n', fileName, ME.message);
        end
    end
end

fprintf('Finished processing files. Found valid durations for %d unique sessions.\n', length(dateTimeToDurationMap));

%% --- 3. Annotate the output_table with the new column ---
fprintf('Matching by SessionDateTime and adding "SessionDuration" column...\n');

% Initialize the new column with NaN
numRows = height(output_table);
output_table.SessionDuration = NaN(numRows, 1);
matchesFound = 0;

% Iterate through the table to find matches
for i = 1:numRows
    table_dt = output_table.SessionDateTime(i);
    
    % Check for a valid datetime and find its corresponding duration in the map
    % Convert table's datetime to the same key format
    lookupKey = datestr(table_dt, 'yyyy-mm-dd HH:MM:SS');
    
    if isKey(dateTimeToDurationMap, lookupKey)
        output_table.SessionDuration(i) = dateTimeToDurationMap(lookupKey);
        matchesFound = matchesFound + 1;
	end
end

fprintf('Successfully added session duration for %d out of %d rows.\n', matchesFound, numRows);
fprintf('Processing complete.\n');

end

