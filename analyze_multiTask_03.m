addpath([pwd filesep 'fcn_local'])
% /Users/wenxuan/Documents/MATLAB/Multitasking_analysis_METRICS/WM_AS_CR_FL__MultiTask_Metrics.mat
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
end



% --- --- --- --- --- --- --- --- --- --- ---
% --- FL
% --- --- --- --- --- --- --- --- --- --- ---
% plot_flexlearning_learningcurves(metrics_mt);
% plot_flexlearning_learning_points_by_condition(metrics_mt);
% plot_plateau_accuracy_by_condition(metrics_mt);
plot_CEn_proportion_by_condition(metrics_mt);
% plot_CEn_proportion_over_sessions(metrics_mt);

% plot_flexlearning_learningcurves_RT(metrics_mt);


% --- --- --- --- --- --- --- --- --- --- ---
% --- AS
% --- --- --- --- --- --- --- --- --- --- ---
% plot_AS_accuracy(metrics_mt);
% plot_AS_accuracy_differences(metrics_mt);
% plot_AS_diffs_over_sessions_time(metrics_mt);
% 
% plot_AS_RT_overall_and_pairs(metrics_mt);
% plot_AS_RT_differences(metrics_mt);
% plot_AS_RT_diffs_over_sessions_time(metrics_mt);


% --- --- --- --- --- --- --- --- --- --- ---
% --- CR
% --- --- --- --- --- --- --- --- --- --- ---
% plot_cr_accuracy_by_subject(metrics_mt)
% plot_cr_accuracy_by_subject_new(metrics_mt)
% plot_cr_avg_trials_to_error(metrics_mt)
% plot_cr_stimuli_at_threshold_accuracy(metrics_mt)
% plot_cr_stimuli_at_threshold_accuracy(metrics_mt, 0.5)
% plot_cr_rt_by_subject(metrics_mt)
% plot_cr_rt_slope_by_subject(metrics_mt)
% plot_cr_rt_intercept_by_subject(metrics_mt)


% plot_cr_nback(metrics_mt);
% plot_cr_nback_heatmap(metrics_mt);
 % plot_cr_nback_unstacked(metrics_mt);
% plot_cr_nback_slope_subject_avg(metrics_mt);


% --- --- --- --- --- --- --- --- --- --- ---
% --- WM
% --- --- --- --- --- --- --- --- --- --- ---

% plot_wm_accuracy_vs_delay(metrics_mt)
% plot_wm_accuracy_by_condition_grouped(metrics_mt)
% plot_wm_RT_vs_delay(metrics_mt)
% plot_wm_RT_by_condition_grouped(metrics_mt)
% plot_wm_accuracy_slope_horizontal(metrics_mt)
% plot_wm_rt_slope_horizontal(metrics_mt)


function plot_cr_accuracy_by_subject_new(metrics_mt)
% plot_cr_accuracy_by_subject_new Plots average accuracy for CR task with error bars.
%   - ... (previous comments) ...
%   - **MODIFIED**: Now saves all plotted data (accuracy curves and thresholds)
%     to CSV files in the same folder as the figures.
%   - **MODIFIED**: Adds an "Overall" average dot for the 75% threshold
%     to both the plot and the threshold CSV file.
%
% INPUT:
%   metrics_mt - Cell array where iS indexes subjects.
%                metrics_mt{iS}(iD) is a structure for one session.

% --- Hardcoded Subject Information and Colors ---
subject_id_list = {
    'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan' 
};
custom_colors = {
    [0.945, 0.345, 0.275]; % #F15846
    [1.0, 0.690, 0.314];   % #FFB050
    [0.984, 0.906, 0.620]; % #FBE79E
    [0.533, 0.843, 0.855]; % #88D7DA
    [0.341, 0.663, 0.902]; % #57A9E6
    [0.420, 0.447, 0.714]  % #6B72B6
};

% --- Plotting Constants ---
x_axis_for_plot = (1:20)';
expected_input_num_rows = 19;
expected_input_num_cols = 4;
base_font_size = 14;

% --- Input Validation ---
if nargin < 1
    error('Usage: plot_cr_accuracy_by_subject_new(metrics_mt)');
end
if ~iscell(metrics_mt)
    error('metrics_mt must be a cell array.');
end

% --- Figure Setup ---
figure_width = 400;
figure_height = 240;
fig = figure('Position', [100, 100, figure_width, figure_height]);
ax = gca;
hold(ax, 'on');

legend_handles = [];
plotted_anything_for_legend = false;
num_subjects_in_data = length(metrics_mt);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
% %%% --- MODIFIED CODE START: Prepare structures for data export --- %%% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
threshold_plot_data = struct('mean', [], 'sem', [], 'color', [], 'label', {});
subject_data_count = 0;
all_subjects_accuracy_data = struct('Subject', {}, 'Data', {}); 

% %%% --- NEW CODE: Array to hold all session data from all subjects --- %%%
all_sessions_all_subjects_thresholds = [];
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
% %%% --- MODIFIED CODE END --- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

% --- Main Loop for Subjects ---
for iS = 1:num_subjects_in_data
    
    session_accuracies_for_subject_1_to_20 = {}; 
    
    if iS > length(metrics_mt) || isempty(metrics_mt{iS})
        continue;
    end
    
    num_sessions_for_subject = length(metrics_mt{iS});
    valid_sessions_count = 0;

    for iD = 1:num_sessions_for_subject
        if ~isstruct(metrics_mt{iS}(iD))
            continue;
        end
        
        session_data = metrics_mt{iS}(iD);
        
        subject_name_for_warning = 'Unknown';
        if iS <= length(subject_id_list)
            subject_name_for_warning = subject_id_list{iS};
        else
            subject_name_for_warning = sprintf('Subject %d', iS);
        end

        if isfield(session_data, 'Acc_NumStim') && ~isempty(session_data.Acc_NumStim)
            acc_stim_data = session_data.Acc_NumStim;
            
            if size(acc_stim_data, 1) == expected_input_num_rows && size(acc_stim_data, 2) == expected_input_num_cols
                mean_acc_2_to_20 = acc_stim_data(:, 2);
                full_session_accuracy_1_to_20 = [1; mean_acc_2_to_20]; 
                session_accuracies_for_subject_1_to_20{end+1} = full_session_accuracy_1_to_20; 
                valid_sessions_count = valid_sessions_count + 1;
            else
                warning('Subject %s (Index %d), Session %d: Acc_NumStim dimensions do not match. Skipping.', ...
                        subject_name_for_warning, iS, iD);
            end
        end
    end

    if valid_sessions_count > 0
        all_sessions_matrix = cat(2, session_accuracies_for_subject_1_to_20{:});
        subject_avg_accuracy = nanmean(all_sessions_matrix, 2); 
        
        if valid_sessions_count > 1
            subject_std_dev = nanstd(all_sessions_matrix, 0, 2);
            subject_sem = subject_std_dev / sqrt(valid_sessions_count);
        else
            subject_sem = zeros(size(subject_avg_accuracy));
        end

        if iS <= length(subject_id_list)
            current_subject_label = subject_id_list{iS};
        else
            current_subject_label = sprintf('Subject %d', iS);
        end
        
        if iS <= length(custom_colors)
            current_color = custom_colors{iS};
        else
            current_color = rand(1,3); 
        end
        
        h = errorbar(ax, x_axis_for_plot, subject_avg_accuracy, subject_sem, ...
                 'LineWidth', 1.5, 'Color', current_color, 'Marker', 'o', ...
                 'MarkerSize', 4, 'DisplayName', current_subject_label, ...
                 'CapSize', 3);
        
        legend_handles(end+1) = h;
        plotted_anything_for_legend = true;
        
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
        % %%% --- NEW CODE START: Store accuracy data for saving --- %%%%%%%%% %
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
        try
            T_acc = table(x_axis_for_plot, subject_avg_accuracy, subject_sem, 'VariableNames', {'StimulusCount', 'MeanAccuracy', 'SEM_Accuracy'});
            all_subjects_accuracy_data(end+1).Subject = current_subject_label;
            all_subjects_accuracy_data(end).Data = T_acc;
        catch ME_store
            warning('Could not store accuracy data for subject %s: %s', current_subject_label, ME_store.message);
        end
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
        % %%% --- NEW CODE END --- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

        
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
        % %%% --- MODIFIED CODE START: Calculate and STORE 75% Acc Threshold --- %%% %
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
        
        accuracy_threshold = 0.75;
        epsilon = 1e-7;
        session_threshold_stimuli = [];
        
        for iSession = 1:length(session_accuracies_for_subject_1_to_20)
            acc_this_session = session_accuracies_for_subject_1_to_20{iSession};
            stim_counts = (1:length(acc_this_session))';
            
            valid_idx = ~isnan(acc_this_session);
            accuracy_valid = acc_this_session(valid_idx);
            stim_counts_for_valid_acc = stim_counts(valid_idx);
            
            estimated_stim_at_thresh = NaN;
            if length(accuracy_valid) >= 2
                accuracy_noisy = accuracy_valid + (rand(size(accuracy_valid)) * epsilon) - (epsilon/2);
                [sorted_acc_noisy, sort_idx] = sort(accuracy_noisy);
                sorted_stim_counts = stim_counts_for_valid_acc(sort_idx);
                [unique_sorted_acc, ia] = unique(sorted_acc_noisy, 'stable');
                unique_corresponding_stim_counts = sorted_stim_counts(ia);
                
                if length(unique_sorted_acc) >= 2
                    estimated_stim_at_thresh = interp1(unique_sorted_acc, unique_corresponding_stim_counts, accuracy_threshold, 'pchip', 'extrap');
                end
            end
            session_threshold_stimuli(end+1) = estimated_stim_at_thresh;
        end
        
        % %%% --- NEW CODE: Accumulate all session thresholds --- %%%
        all_sessions_all_subjects_thresholds = [all_sessions_all_subjects_thresholds; session_threshold_stimuli(:)];
        % %%% --- END NEW CODE --- %%%

        if ~isempty(session_threshold_stimuli)
            mean_threshold_stim = nanmean(session_threshold_stimuli);
            num_valid_thresholds = sum(~isnan(session_threshold_stimuli));
            if num_valid_thresholds > 1
                sem_threshold_stim = nanstd(session_threshold_stimuli) / sqrt(num_valid_thresholds);
            else
                sem_threshold_stim = 0;
            end
            
            if ~isnan(mean_threshold_stim)
                subject_data_count = subject_data_count + 1;
                threshold_plot_data(subject_data_count).mean = mean_threshold_stim;
                threshold_plot_data(subject_data_count).sem = sem_threshold_stim;
                threshold_plot_data(subject_data_count).color = current_color;
                threshold_plot_data(subject_data_count).label = current_subject_label;
            end
        end
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
        % %%% --- MODIFIED CODE END --- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
    end
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
% %%% --- NEW CODE: Calculate Overall Threshold Data --- %%%%%%%%%%%%% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
overall_mean = nanmean(all_sessions_all_subjects_thresholds);
num_valid_overall_sessions = sum(~isnan(all_sessions_all_subjects_thresholds));
overall_sem = 0;
if num_valid_overall_sessions > 1
    overall_sem = nanstd(all_sessions_all_subjects_thresholds) / sqrt(num_valid_overall_sessions);
end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
% %%% --- END NEW CODE --- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
% %%% --- NEW CODE START: Sort and plot the stored threshold data --- %%% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
min_y_for_dots = 1.3; % Keep track of the lowest dot position for ylim
if ~isempty([threshold_plot_data.mean])
    % Sort the data in descending order based on the mean threshold value
    [~, sort_order] = sort([threshold_plot_data.mean], 'descend');
    sorted_threshold_data = threshold_plot_data(sort_order);
    
    % Define vertical plotting positions
    initial_y_pos = 1.4;
    y_step = 0.03;

    % Loop through the SORTED data and plot each point
    for k = 1:length(sorted_threshold_data)
        y_pos = initial_y_pos - (k - 1) * y_step;
        data_to_plot = sorted_threshold_data(k);
        
        errorbar(ax, data_to_plot.mean, y_pos, data_to_plot.sem, 'horizontal', ...
            'o', 'Color', data_to_plot.color, 'MarkerFaceColor', data_to_plot.color, ...
            'MarkerSize', 5, 'LineWidth', 1.5, 'CapSize', 4, ...
            'HandleVisibility', 'off'); % Prevents creating legend entries
    end
    
    % Update the minimum y position for axis scaling
    min_y_for_dots = initial_y_pos - (length(sorted_threshold_data) - 1) * y_step;
end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
% %%% --- NEW CODE END --- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
% %%% --- NEW CODE: Plot the "Overall" dot --- %%%%%%%%%%%%%%%%%%%%%%% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
if ~isnan(overall_mean)
    y_pos_overall = 1.45; % Place it above the others
    errorbar(ax, overall_mean, y_pos_overall, overall_sem, 'horizontal', ...
            's', 'Color', 'k', 'MarkerFaceColor', 'k', ... % 's' is square, 'k' is black
            'MarkerSize', 7, 'LineWidth', 1.5, 'CapSize', 4, ...
            'HandleVisibility', 'off');
    text(ax, overall_mean + 0.1, y_pos_overall, 'Overall', ...
             'FontSize', base_font_size - 2, 'VerticalAlignment', 'middle');
end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
% %%% --- END NEW CODE --- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %


% --- Finalize Plot Styling and Saving ---
set(ax, 'FontSize', base_font_size);

if plotted_anything_for_legend
    lgd = legend(ax, legend_handles, 'Location', 'eastoutside', 'FontSize', base_font_size);
    lgd.Box = 'off'; 
else
    text(ax, 0.5, 0.5, 'No data available to plot.', 'HorizontalAlignment', 'center', 'Units', 'normalized', 'FontSize', base_font_size, 'Color', 'red');
end

xlabel(ax, 'Number of Stimuli', 'FontSize', base_font_size);
ylabel(ax, 'Average Accuracy', 'FontSize', base_font_size);
title(ax, 'CR Task: Average Accuracy vs. Number of Stimuli by Subject', 'FontSize', base_font_size);
grid(ax, 'off'); 
axis(ax, 'tight');
xlim(ax, [min(x_axis_for_plot)-0.5 max(x_axis_for_plot)+0.5]);

% %%% --- MODIFIED YLIM to ensure "Overall" dot is visible --- %%%
ylim(ax, [min(0, min_y_for_dots - 0.1), 1.55]); 

drawnow; pause(0.1); 

if plotted_anything_for_legend && exist('lgd', 'var') && isvalid(lgd)
    try 
        if strcmp(lgd.Location, 'eastoutside')
            original_ax_units = get(ax, 'Units');
            original_lgd_units = get(lgd, 'Units');
            set(ax, 'Units', 'normalized');
            set(lgd, 'Units', 'normalized');
            drawnow; pause(0.1);
            
            ax_pos_norm = get(ax, 'Position'); 
            lgd_outer_pos_norm = get(lgd, 'OuterPosition'); 
            
            max_allowable_ax_width = lgd_outer_pos_norm(1) - ax_pos_norm(1) - 0.03; 
            
            if ax_pos_norm(3) > max_allowable_ax_width && max_allowable_ax_width > 0.1 
                set(ax, 'Position', [ax_pos_norm(1), ax_pos_norm(2), max_allowable_ax_width, ax_pos_norm(4)]);
            end
            
            set(ax, 'Units', original_ax_units);
            set(lgd, 'Units', original_lgd_units);
        end
    catch ME_layout
        fprintf('Warning: Could not auto-adjust layout for legend: %s\n', ME_layout.message);
    end
end

if exist('fig', 'var') && isvalid(fig)
    figure(fig); 
    
    base_filename = 'cr_accuracy_vs_stimuli_by_subject'; 
    date_str = datestr(now, 'yyyymmdd_HHMMSS'); 
    full_base_filename = [base_filename '_' date_str];
    
    save_folder = 'CR_Figures_Accuracy'; 
    if ~exist(save_folder, 'dir')
        try; mkdir(save_folder); catch; save_folder = '.'; end
    end
    
    filepath_base = fullfile(save_folder, full_base_filename);

    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
    % %%% --- NEW CODE START: Save data to CSV files --- %%%%%%%%%%%%%%%%% %
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
    
    % --- 1. Save Average Accuracy Data ---
    if ~isempty(all_subjects_accuracy_data)
        try
            combined_accuracy_table = table();
            for i = 1:length(all_subjects_accuracy_data)
                subject_name = all_subjects_accuracy_data(i).Subject;
                temp_table = all_subjects_accuracy_data(i).Data;
                % Add subject name as a repeated column
                subject_col = repmat(string(subject_name), height(temp_table), 1);
                temp_table.Subject = subject_col;
                % Reorder columns
                temp_table = temp_table(:, {'Subject', 'StimulusCount', 'MeanAccuracy', 'SEM_Accuracy'});
                % Append
                combined_accuracy_table = [combined_accuracy_table; temp_table];
            end
            accuracy_csv_filename = [filepath_base '_accuracy_data.csv'];
            writetable(combined_accuracy_table, accuracy_csv_filename);
            fprintf('Saved accuracy data to: %s\n', accuracy_csv_filename);
        catch ME_csv_acc
            warning('Failed to save accuracy data to CSV: %s', ME_csv_acc.message);
        end
    else
        warning('No accuracy data was stored to save to CSV.');
    end
    
    % %%% --- 2. Save Threshold Data (MODIFIED) --- %%%
    if ~isempty(threshold_plot_data) || ~isnan(overall_mean)
        try
            T_thresh = table();
            if ~isempty(threshold_plot_data)
                 T_thresh = struct2table(threshold_plot_data);
                 % Select and rename columns for clarity
                 T_thresh = T_thresh(:, {'label', 'mean', 'sem'});
                 T_thresh.Properties.VariableNames = {'Subject', 'MeanStimuliToCriterion', 'SEM_StimuliToCriterion'};
            end
            
            % %%% --- NEW CODE: Add "Overall" row to table --- %%%
            if ~isnan(overall_mean)
                overall_row = table(string("Overall"), overall_mean, overall_sem, ...
                    'VariableNames', {'Subject', 'MeanStimuliToCriterion', 'SEM_StimuliToCriterion'});
                T_thresh = [T_thresh; overall_row];
            end
            % %%% --- END NEW CODE --- %%%

            if height(T_thresh) > 0
                threshold_csv_filename = [filepath_base '_threshold_data.csv'];
                writetable(T_thresh, threshold_csv_filename);
                fprintf('Saved threshold data to: %s\n', threshold_csv_filename);
            else
                warning('No threshold data was available to save to CSV.');
            end
        catch ME_csv_thresh
            warning('Failed to save threshold data to CSV: %s', ME_csv_thresh.message);
        end
    else
         warning('No threshold data was calculated to save to CSV.');
    end
    
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
    % %%% --- NEW CODE END --- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

    
    % --- Save Figure ---
    try; print(fig, [filepath_base '.eps'], '-depsc', '-painters'); catch; end
    try; saveas(fig, [filepath_base '.png']); catch; end
end

hold(ax, 'off');

end


function plot_wm_rt_slope_horizontal(metrics_mt)
% plot_wm_rt_slope_horizontal Calculates and plots the RT slope vs. delay.
%
%   - For each session, calculates the slope of [Reaction Time vs. Delay].
%   - For each subject, averages these per-session slopes.
%   - Error bars represent the SEM of the slopes across sessions.
%   - Uses a try-catch block to ignore sessions where model fitting fails.
%   - An "Overall" slope is calculated by averaging the subject mean slopes.
%   - Figure size is 450x300.
%   - Saves figure as PNG, EPS, and data as CSV.

% --- Hardcoded Subject Information and Colors ---
subject_id_list_original = {
    'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan'
};
custom_colors = {
    [245, 124, 110]/255; [242, 181, 110]/255; [251, 231, 158]/255;
    [132, 195, 183]/255; [136, 215, 218]/255; [113, 184, 237]/255;
    [184, 174, 234]/255; [242, 168, 218]/255
};
overall_color = [0 0 0]; % Black for overall

% --- Font Size & Plotting Constants ---
base_font_size = 14;
y_pos_overall = 1; % 'Overall' is fixed at the top position
gap_after_overall = 1.2; % Gap between 'Overall' and the first subject
inter_subject_gap = 0.8; % Gap between subsequent subjects

% --- Input Validation ---
if nargin < 1
    error('Usage: %s(metrics_mt)', mfilename);
end
if ~iscell(metrics_mt)
    error('metrics_mt must be a cell array.');
end

% --- Figure Setup ---
figure_width = 450;  % MODIFIED
figure_height = 300; % MODIFIED
fig = figure('Position', [100, 100, figure_width, figure_height]);
ax = gca;
hold(ax, 'on');

num_subjects_in_data = length(metrics_mt);
subject_slopes = NaN(1, num_subjects_in_data);
subject_slope_errors = NaN(1, num_subjects_in_data);
subject_initial_labels_all = cell(1, num_subjects_in_data);
valid_subject_mask = false(1, num_subjects_in_data);

% --- Data Extraction and Slope Calculation ---
for iS = 1:num_subjects_in_data
    if iS > length(metrics_mt) || isempty(metrics_mt{iS})
        continue;
    end
    
    % This will store one slope value from each session for the current subject
    all_session_slopes = [];
    
    % Loop through each session for the current subject
    for iD = 1:length(metrics_mt{iS})
        if ~isstruct(metrics_mt{iS}(iD)); continue; end
        session_data = metrics_mt{iS}(iD);

        % Define field names for clarity
        rt_field = 'RT_MeanSE_wm';
        delay_cond_field = 'conditions_nDistr_delays_wm';

        has_rt = isfield(session_data, rt_field) && ~isempty(session_data.(rt_field));
        has_delays = isfield(session_data, delay_cond_field) && ~isempty(session_data.(delay_cond_field));

        if has_rt && has_delays
            reaction_times = session_data.(rt_field)(:, 1);
            delay_conditions = session_data.(delay_cond_field);

            if size(delay_conditions, 2) >= 2 && size(delay_conditions, 1) == size(reaction_times, 1)
                delays = delay_conditions(:, 2);
                
                % A slope requires at least 2 unique data points
                if sum(~isnan(reaction_times)) >= 2 && length(unique(delays)) >= 2
                    % --- MODIFIED: Use try-catch for robust model fitting ---
                    try
                        mdl = fitlm(delays, reaction_times);
                        session_slope = mdl.Coefficients.Estimate(2);
                        all_session_slopes = [all_session_slopes; session_slope];
                    catch ME % Catch any error during model fitting
                        fprintf('Warning: Could not fit model for Subject %d, Session %d. Skipping session.\n', iS, iD);
                        fprintf('  Error ID: %s\n', ME.identifier);
                        fprintf('  Error Message: %s\n', ME.message);
                    end
                end
            end
        end
    end

    % Now, average the slopes collected from all sessions for this subject
    if ~isempty(all_session_slopes)
        subject_slopes(iS) = mean(all_session_slopes, 'omitnan');
        
        % Calculate Standard Error of the Mean for the slopes
        if length(all_session_slopes) > 1
            subject_slope_errors(iS) = std(all_session_slopes, 'omitnan') / sqrt(length(all_session_slopes));
        else
            subject_slope_errors(iS) = 0; % No error if only one session
        end
        
        valid_subject_mask(iS) = true;
        
        if iS <= length(subject_id_list_original) && ~isempty(subject_id_list_original{iS})
            original_id = subject_id_list_original{iS};
            subject_initial_labels_all{iS} = sprintf('%c', upper(original_id(1)));
        else
            subject_initial_labels_all{iS} = sprintf('S%d', iS); 
        end
    end
end

% --- Filter for valid subjects and prepare for plotting ---
actual_subject_slopes = subject_slopes(valid_subject_mask);
actual_subject_errors = subject_slope_errors(valid_subject_mask);
final_subject_initials = subject_initial_labels_all(valid_subject_mask);
num_valid_subjects = length(actual_subject_slopes);

legend_handles = [];
plotted_anything_for_legend = false;
ytick_positions = [];
ytick_labels_list = {};
min_x_extent = Inf; 
max_x_extent = -Inf;

% --- Overall Slope Calculation and Plotting ---
overall_mean_slope = NaN;
overall_sem_slope = NaN;
if num_valid_subjects > 0
    overall_mean_slope = mean(actual_subject_slopes, 'omitnan');
    if num_valid_subjects > 1
        overall_sem_slope = std(actual_subject_slopes, 'omitnan') / sqrt(num_valid_subjects);
    else
        overall_sem_slope = 0;
    end
    
    h_overall = errorbar(ax, overall_mean_slope, y_pos_overall, overall_sem_slope, 'horizontal', 'o', ...
        'MarkerSize', 3, 'MarkerEdgeColor', overall_color, 'MarkerFaceColor', overall_color, ...
        'Color', overall_color, 'LineWidth', 1, 'CapSize', 10, 'DisplayName', 'Overall');
    legend_handles(end+1) = h_overall;
    plotted_anything_for_legend = true;
    
    ytick_positions(end+1) = y_pos_overall;
    ytick_labels_list{end+1} = 'Overall';
    
    min_x_extent = min(min_x_extent, overall_mean_slope - overall_sem_slope);
    max_x_extent = max(max_x_extent, overall_mean_slope + overall_sem_slope);
end

% --- Plotting Individual Subject Data ---
subject_y_positions = [];
if num_valid_subjects > 0
    current_y = y_pos_overall + gap_after_overall;
    for k = 1:num_valid_subjects
        subject_y_positions(k) = current_y + (k-1) * inter_subject_gap;
    end
end

original_indices_of_valid_subjects = find(valid_subject_mask);
for k = 1:num_valid_subjects
    iS_original = original_indices_of_valid_subjects(k);
    
    current_color = custom_colors{mod(iS_original-1, length(custom_colors)) + 1};
    y_pos = subject_y_positions(k);
    
    h_subj = errorbar(ax, actual_subject_slopes(k), y_pos, actual_subject_errors(k), 'horizontal', 'o', ...
        'MarkerSize', 3, 'MarkerEdgeColor', current_color, 'MarkerFaceColor', current_color, ...
        'Color', current_color, 'LineWidth', 1, 'CapSize', 10, ...
        'DisplayName', sprintf('Subj %s', final_subject_initials{k}));
    
    if ~isempty(get(h_subj, 'DisplayName'))
         legend_handles(end+1) = h_subj;
    end
    plotted_anything_for_legend = true; 
    
    ytick_positions(end+1) = y_pos;
    ytick_labels_list{end+1} = final_subject_initials{k};

    min_x_extent = min(min_x_extent, actual_subject_slopes(k) - actual_subject_errors(k));
    max_x_extent = max(max_x_extent, actual_subject_slopes(k) + actual_subject_errors(k));
end

% --- Finalize Plot Styling ---
set(ax, 'FontSize', base_font_size);
if plotted_anything_for_legend
    lgd = legend(ax, legend_handles, 'Location', 'eastoutside', 'FontSize', base_font_size);
    lgd.Box = 'off';
    
    set(ax, 'YTick', ytick_positions);
    set(ax, 'YTickLabel', ytick_labels_list);
else
    text(ax, 0.5, 0.5, 'No data available to plot.', 'HorizontalAlignment', 'center', ...
         'Units', 'normalized', 'FontSize', base_font_size, 'Color', 'red');
    warning('No slopes were calculated. Check input data.');
    set(ax, 'YTick', []);
end

ylabel(ax, 'Subject', 'FontSize', base_font_size);
xlabel(ax, 'Slope of RT vs. Delay', 'FontSize', base_font_size);
title(ax, 'Effect of Delay on Reaction Time by Subject', 'FontSize', base_font_size, 'FontWeight', 'bold');
grid(ax, 'off');
ax.XAxis.MinorTick = 'on';
if ~isempty(ytick_positions)
    line(ax, [0 0], [0 max(ytick_positions)+1], 'Color', [0.7 0.7 0.7], 'LineStyle', '--', 'HandleVisibility', 'off');
end

% --- Axis Limit Logic ---
if ~isempty(ytick_positions)
    set(ax, 'YDir', 'reverse');
    ylim(ax, [min(ytick_positions) - inter_subject_gap*0.75, max(ytick_positions) + inter_subject_gap*0.5]);
    
    if isinf(min_x_extent) || isinf(max_x_extent) 
        min_x_extent = -0.05; max_x_extent = 0.05;
    end
    x_range = max_x_extent - min_x_extent;
    if x_range == 0; x_range = abs(max_x_extent)*0.2 + 0.01; end
    padding = x_range * 0.10;
    xlim(ax, [-0.01, max_x_extent + padding]);
else 
    xlim(ax, [-0.1 0.1]); 
    ylim(ax, [0 1]);
end

% --- Adjust layout for legend ---
drawnow;
if plotted_anything_for_legend && isvalid(lgd)
    try
        original_ax_units = get(ax, 'Units');
        set(ax, 'Units', 'normalized');
        drawnow;
        ax_pos = get(ax, 'Position');
        lgd_pos = get(lgd, 'Position');
        if ax_pos(1) + ax_pos(3) > lgd_pos(1)
            ax_pos(3) = lgd_pos(1) - ax_pos(1) - 0.05;
            set(ax, 'Position', ax_pos);
        end
        set(ax, 'Units', original_ax_units);
    catch ME_layout
        fprintf('Warning: Could not auto-adjust layout: %s\n', ME_layout.message);
    end
end

% --- Saving the Figure and Data ---
if exist('fig', 'var') && isvalid(fig)
    base_filename = 'wm_rt_slope_horizontal';
    date_str = datestr(now, 'yyyymmdd_HHMMSS');
    full_base_filename = [base_filename '_' date_str];
    save_folder = 'WM_Figures_RTSlopes'; % Changed folder name
    if ~exist(save_folder, 'dir'); mkdir(save_folder); end
    filepath_base = fullfile(save_folder, full_base_filename);
    
    saveas(fig, [filepath_base '.png']);
    fprintf('Figure saved as: %s.png\n', filepath_base);
    print(fig, [filepath_base '.eps'], '-depsc', '-painters');
    fprintf('Figure saved as: %s.eps\n', filepath_base);
    
    if plotted_anything_for_legend
        data_to_save = {};
        if ~isnan(overall_mean_slope)
            data_to_save(end+1, :) = {'Overall', overall_mean_slope, overall_sem_slope};
        end
        for i = 1:num_valid_subjects
            data_to_save(end+1, :) = {final_subject_initials{i}, actual_subject_slopes(i), actual_subject_errors(i)};
        end
        results_table = cell2table(data_to_save, 'VariableNames', {'Label', 'Mean_Slope', 'SEM_Slope'});
        csv_filename = [filepath_base '.csv'];
        writetable(results_table, csv_filename);
        fprintf('Data saved as: %s.csv\n', csv_filename);
    end
end

hold(ax, 'off');
end



function plot_wm_accuracy_slope_horizontal(metrics_mt)
% plot_wm_accuracy_slope_horizontal Calculates and plots the accuracy slope
% vs. delay for each subject.
%
%   - NEW: For each session, calculates the slope of [Accuracy vs. Delay].
%   - NEW: For each subject, averages these per-session slopes.
%   - NEW: Error bars represent the SEM of the slopes across sessions for each subject.
%   - Filters all data to only include trials where delay >= 0.5 seconds.
%   - An "Overall" slope is calculated by averaging the individual subject mean slopes.
%   - Y-axis labels are subject initials. 'Overall' is at the top.
%   - Saves figure as PNG, EPS, and data as CSV.

% --- Hardcoded Subject Information and Colors ---
subject_id_list_original = {
    'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan'
};
custom_colors = {
    [245, 124, 110]/255; [242, 181, 110]/255; [251, 231, 158]/255;
    [132, 195, 183]/255; [136, 215, 218]/255; [113, 184, 237]/255;
    [184, 174, 234]/255; [242, 168, 218]/255
};
overall_color = [0 0 0]; % Black for overall

% --- Font Size & Plotting Constants ---
base_font_size = 14;
y_pos_overall = 1; % 'Overall' is fixed at the top position
gap_after_overall = 1.2; % Gap between 'Overall' and the first subject
inter_subject_gap = 0.8; % Gap between subsequent subjects

% --- Input Validation ---
if nargin < 1
    error('Usage: %s(metrics_mt)', mfilename);
end
if ~iscell(metrics_mt)
    error('metrics_mt must be a cell array.');
end

% --- Figure Setup ---
figure_width = 450;
figure_height = 300;
fig = figure('Position', [100, 100, figure_width, figure_height]);
ax = gca;
hold(ax, 'on');

num_subjects_in_data = length(metrics_mt);
subject_slopes = NaN(1, num_subjects_in_data);
subject_slope_errors = NaN(1, num_subjects_in_data);
subject_initial_labels_all = cell(1, num_subjects_in_data);
valid_subject_mask = false(1, num_subjects_in_data);

% --- Data Extraction and Slope Calculation ---
for iS = 1:num_subjects_in_data
    if iS > length(metrics_mt) || isempty(metrics_mt{iS})
        continue;
    end
    
    % This will store one slope value from each session for the current subject
    all_session_slopes = [];
    
    % Loop through each session for the current subject
    for iD = 1:length(metrics_mt{iS})
        if ~isstruct(metrics_mt{iS}(iD)); continue; end
        session_data = metrics_mt{iS}(iD);

        % --- REVISED LOGIC: Calculate slope of Accuracy vs. Delay per session ---
        acc_field = 'accuracyMeanSE_wm';
        delay_cond_field = 'conditions_nDistr_delays_wm';

        has_acc = isfield(session_data, acc_field) && ~isempty(session_data.(acc_field));
        has_delays = isfield(session_data, delay_cond_field) && ~isempty(session_data.(delay_cond_field));

        if has_acc && has_delays
            accuracies = session_data.(acc_field)(:, 1);
            delay_conditions = session_data.(delay_cond_field);

            if size(delay_conditions, 2) >= 2 && size(delay_conditions, 1) == size(accuracies, 1)
                delays = delay_conditions(:, 2);
                
                % Filter to only include delays >= 0.5s
                filter_indices = delays >= 0.5 & ~isnan(accuracies);
                filtered_delays = delays(filter_indices);
                filtered_accuracies = accuracies(filter_indices);
                
                % A slope requires at least 2 unique data points
                if sum(~isnan(filtered_accuracies)) >= 2 && length(unique(filtered_delays)) >= 2
					try
						mdl = fitlm(filtered_delays, filtered_accuracies);
					catch
						continue;
					end
					
                    session_slope = mdl.Coefficients.Estimate(2);
                    all_session_slopes = [all_session_slopes; session_slope];
                end
            end
        end
    end

    % Now, average the slopes collected from all sessions for this subject
    if ~isempty(all_session_slopes)
        subject_slopes(iS) = mean(all_session_slopes, 'omitnan');
        
        % Calculate Standard Error of the Mean for the slopes
        if length(all_session_slopes) > 1
            subject_slope_errors(iS) = std(all_session_slopes, 'omitnan') / sqrt(length(all_session_slopes));
        else
            subject_slope_errors(iS) = 0; % No error if only one session
        end
        
        valid_subject_mask(iS) = true;
        
        if iS <= length(subject_id_list_original) && ~isempty(subject_id_list_original{iS})
            original_id = subject_id_list_original{iS};
            subject_initial_labels_all{iS} = sprintf('%c', upper(original_id(1)));
        else
            subject_initial_labels_all{iS} = sprintf('S%d', iS); 
        end
    end
end

% --- Filter for valid subjects and prepare for plotting ---
actual_subject_slopes = subject_slopes(valid_subject_mask);
actual_subject_errors = subject_slope_errors(valid_subject_mask);
final_subject_initials = subject_initial_labels_all(valid_subject_mask);
num_valid_subjects = length(actual_subject_slopes);

legend_handles = [];
plotted_anything_for_legend = false;
ytick_positions = [];
ytick_labels_list = {};
min_x_extent = Inf; 
max_x_extent = -Inf;

% --- Overall Slope Calculation and Plotting ---
overall_mean_slope = NaN;
overall_sem_slope = NaN;
if num_valid_subjects > 0
    overall_mean_slope = mean(actual_subject_slopes, 'omitnan');
    if num_valid_subjects > 1
        overall_sem_slope = std(actual_subject_slopes, 'omitnan') / sqrt(num_valid_subjects);
    else
        overall_sem_slope = 0;
    end
    
    h_overall = errorbar(ax, overall_mean_slope, y_pos_overall, overall_sem_slope, 'horizontal', 'o', ...
        'MarkerSize', 3, 'MarkerEdgeColor', overall_color, 'MarkerFaceColor', overall_color, ...
        'Color', overall_color, 'LineWidth', 1, 'CapSize', 10, 'DisplayName', 'Overall');
    legend_handles(end+1) = h_overall;
    plotted_anything_for_legend = true;
    
    ytick_positions(end+1) = y_pos_overall;
    ytick_labels_list{end+1} = 'Overall';
    
    min_x_extent = min(min_x_extent, overall_mean_slope - overall_sem_slope);
    max_x_extent = max(max_x_extent, overall_mean_slope + overall_sem_slope);
end

% --- Plotting Individual Subject Data ---
subject_y_positions = [];
if num_valid_subjects > 0
    current_y = y_pos_overall + gap_after_overall;
    for k = 1:num_valid_subjects
        subject_y_positions(k) = current_y + (k-1) * inter_subject_gap;
    end
end

original_indices_of_valid_subjects = find(valid_subject_mask);
for k = 1:num_valid_subjects
    iS_original = original_indices_of_valid_subjects(k);
    
    current_color = custom_colors{mod(iS_original-1, length(custom_colors)) + 1};
    y_pos = subject_y_positions(k);
    
    h_subj = errorbar(ax, actual_subject_slopes(k), y_pos, actual_subject_errors(k), 'horizontal', 'o', ...
        'MarkerSize', 3, 'MarkerEdgeColor', current_color, 'MarkerFaceColor', current_color, ...
        'Color', current_color, 'LineWidth', 1, 'CapSize', 10, ...
        'DisplayName', sprintf('Subj %s', final_subject_initials{k}));
    
    if ~isempty(get(h_subj, 'DisplayName'))
         legend_handles(end+1) = h_subj;
    end
    plotted_anything_for_legend = true; 
    
    ytick_positions(end+1) = y_pos;
    ytick_labels_list{end+1} = final_subject_initials{k};

    min_x_extent = min(min_x_extent, actual_subject_slopes(k) - actual_subject_errors(k));
    max_x_extent = max(max_x_extent, actual_subject_slopes(k) + actual_subject_errors(k));
end

% --- Finalize Plot Styling ---
set(ax, 'FontSize', base_font_size);
if plotted_anything_for_legend
    lgd = legend(ax, legend_handles, 'Location', 'eastoutside', 'FontSize', base_font_size);
    lgd.Box = 'off';
    
    set(ax, 'YTick', ytick_positions);
    set(ax, 'YTickLabel', ytick_labels_list);
else
    text(ax, 0.5, 0.5, 'No data available to plot.', 'HorizontalAlignment', 'center', ...
         'Units', 'normalized', 'FontSize', base_font_size, 'Color', 'red');
    warning('No slopes were calculated. Check input data.');
    set(ax, 'YTick', []);
end

ylabel(ax, 'Subject', 'FontSize', base_font_size);
% --- MODIFIED X-AXIS LABEL ---
xlabel(ax, 'Slope of Accuracy vs. Delay', 'FontSize', base_font_size);
title(ax, 'Effect of Delay on Accuracy by Subject', 'FontSize', base_font_size, 'FontWeight', 'bold');
grid(ax, 'off');
ax.XAxis.MinorTick = 'on';
if ~isempty(ytick_positions)
    line(ax, [0 0], [0 max(ytick_positions)+1], 'Color', [0.7 0.7 0.7], 'LineStyle', '--', 'HandleVisibility', 'off');
end

% --- Axis Limit Logic ---
if ~isempty(ytick_positions)
    set(ax, 'YDir', 'reverse');
    ylim(ax, [min(ytick_positions) - inter_subject_gap*0.75, max(ytick_positions) + inter_subject_gap*0.5]);
    
    if isinf(min_x_extent) || isinf(max_x_extent) 
        min_x_extent = -0.05; max_x_extent = 0.05;
    end
    x_range = max_x_extent - min_x_extent;
    if x_range == 0; x_range = abs(max_x_extent)*0.2 + 0.01; end
    padding = x_range * 0.10;
    xlim(ax, [min_x_extent - padding,0.01]);
else 
    xlim(ax, [-0.1 0.1]); 
    ylim(ax, [0 1]);
end

% --- Adjust layout for legend ---
drawnow;
if plotted_anything_for_legend && isvalid(lgd)
    try
        original_ax_units = get(ax, 'Units');
        set(ax, 'Units', 'normalized');
        drawnow;
        ax_pos = get(ax, 'Position');
        lgd_pos = get(lgd, 'Position');
        if ax_pos(1) + ax_pos(3) > lgd_pos(1)
            ax_pos(3) = lgd_pos(1) - ax_pos(1) - 0.05;
            set(ax, 'Position', ax_pos);
        end
        set(ax, 'Units', original_ax_units);
    catch ME_layout
        fprintf('Warning: Could not auto-adjust layout: %s\n', ME_layout.message);
    end
end

% --- Saving the Figure and Data ---
if exist('fig', 'var') && isvalid(fig)
    base_filename = 'wm_accuracy_vs_delay_slope_horizontal';
    date_str = datestr(now, 'yyyymmdd_HHMMSS');
    full_base_filename = [base_filename '_' date_str];
    save_folder = 'WM_Figures_AccuracySlopes';
    if ~exist(save_folder, 'dir'); mkdir(save_folder); end
    filepath_base = fullfile(save_folder, full_base_filename);
    
    saveas(fig, [filepath_base '.png']);
    fprintf('Figure saved as: %s.png\n', filepath_base);
    print(fig, [filepath_base '.eps'], '-depsc', '-painters');
    fprintf('Figure saved as: %s.eps\n', filepath_base);
    
    if plotted_anything_for_legend
        data_to_save = {};
        if ~isnan(overall_mean_slope)
            data_to_save(end+1, :) = {'Overall', overall_mean_slope, overall_sem_slope};
        end
        for i = 1:num_valid_subjects
            data_to_save(end+1, :) = {final_subject_initials{i}, actual_subject_slopes(i), actual_subject_errors(i)};
        end
        results_table = cell2table(data_to_save, 'VariableNames', {'Label', 'Mean_Slope', 'SEM_Slope'});
        csv_filename = [filepath_base '.csv'];
        writetable(results_table, csv_filename);
        fprintf('Data saved as: %s.csv\n', csv_filename);
    end
end

hold(ax, 'off');
end


function plot_cr_nback_slope_subject_avg(metrics_mt)
% plot_cr_nback_slope_subject_avg Creates a slope plot with subject-wise averaging.
%
% ... (previous comments) ...
%
% MODIFIED:
%   - CSV file now includes summary rows for:
%     1. The average slope for each subject across all TTE positions.
%     2. The grand average slope (and SEM) across all subjects and TTEs.
%   - **NEW (User Request)**:
%     - Plots dots at TTE=20 representing the mean slope (and SEM)
%       averaged across all TTEs for each subject.
%     - Plots a black square at TTE=20 for the grand overall mean (and SEM).
%     - Performs a one-sample t-test (vs 0) for each subject's slopes
%       and for the overall average.
%     - Saves these statistics to a new file: ..._slopes_stats.csv.
%   - **MODIFIED (User Request)**:
%     - Jitter range changed to +/- 0.2.
%     - Color palette updated.
%     - Jitter applied to the subject-average dots at TTE=20.
%
% Input:
%   metrics_mt - A cell array where each cell contains the data for one subject.

% --- Configuration ---
subject_id_list_hardcoded = {
    'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan'
};
% %%% --- MODIFIED: Updated Color Codes --- %%%
custom_colors_rgb = { 
    [241, 88, 70]/255; [255, 176, 80]/255; [251, 231, 158]/255; 
    [136, 215, 218]/255; [87, 169, 230]/255; [107, 114, 182]/255 
};
% %%% --- END MODIFICATION --- %%%

legend_labels = cell(size(subject_id_list_hardcoded));
for i = 1:numel(subject_id_list_hardcoded)
    name = subject_id_list_hardcoded{i};
    if ~isempty(name), legend_labels{i} = sprintf('Subject %c', name(1)); end
end
nSubjects = numel(metrics_mt);

% --- Setup for File Export ---
outputDir = 'CR_nBack_Slope_SubjectAvg';
if ~exist(outputDir, 'dir')
   mkdir(outputDir);
   fprintf('Created directory: %s\n', outputDir);
end

% --- Figure Creation ---
hFig = figure('Position', [100, 100, 600, 400], 'Color', 'w');
ax = gca;

% --- Plot Slopes and get data for CSV ---
% MODIFIED: Now returns statsData as well
[csvData, statsData] = plotSubjectAveragedSlopes(ax, metrics_mt, nSubjects, custom_colors_rgb, legend_labels);

% --- Save Data and Figures ---
try
    if ~isempty(csvData)
        slopesTable = cell2table(csvData(2:end,:), 'VariableNames', csvData(1,:));
        csvFilename = fullfile(outputDir, 'subject_avg_slopes_data.csv');
        writetable(slopesTable, csvFilename);
        fprintf('Successfully saved data to: %s\n', csvFilename);
    else
        warning('No slope data was generated to save.');
    end
    
    % --- NEW: Save Stats Data ---
    if ~isempty(statsData)
        statsTable = cell2table(statsData(2:end,:), 'VariableNames', statsData(1,:));
        statsCsvFilename = fullfile(outputDir, 'subject_avg_slopes_stats.csv');
        writetable(statsTable, statsCsvFilename);
        fprintf('Successfully saved stats to: %s\n', statsCsvFilename);
    else
        warning('No stats data was generated to save.');
    end
    % --- END NEW ---

    if isempty(csvData) && isempty(statsData)
        warning('No data or stats were generated.');
        return;
    end
    
    baseFilename = fullfile(outputDir, 'subject_avg_slopes_plot');
    saveas(hFig, [baseFilename, '.png']);
    fprintf('Successfully saved figure to: %s.png\n', baseFilename);
    print(hFig, [baseFilename '.eps'], '-depsc2', '-vector','-r300');
    fprintf('Successfully saved figure to: %s.eps\n', baseFilename);
catch E
    warning('Could not save files. Error: %s', E.message);
end
end

% --- NESTED FUNCTION: Plot Slopes with Subject-wise Averaging ---
% MODIFIED: Now returns csvData and statsData
function [csvData, statsData] = plotSubjectAveragedSlopes(ax, metrics_mt, nSubjects, colors, legend_labels)
    hold(ax, 'on');
    hPlots = gobjects(nSubjects + 1, 1);
    
    % --- Part 1: Calculate and plot subject-wise average lines ---
    subjectSlopesByTTE = struct();
    all_subject_slopes_by_tte = cell(1, nSubjects); % NEW: To store all slopes for t-tests
    
    % %%% --- MODIFIED: Jitter range set to +/- 0.2 --- %%%
    if nSubjects > 1
        jitter_values = linspace(-0.2, 0.2, nSubjects);
    else
        jitter_values = 0;
    end
    % %%% --- END MODIFICATION --- %%%

    
    for iS = 1:nSubjects
        % Aggregate all sessions for this subject
        allNBack_subj = [];
        allTTE_subj = [];
        for iD = 1:numel(metrics_mt{iS})
            data = metrics_mt{iS}(iD).nBack;
            if isempty(data), continue; end
            if iscell(data), keys = cellfun(@(c)c{1},data); vals=cellfun(@(c)c{2},data);
            else, keys=data(:,1)'; vals=data(:,2)'; end
            allNBack_subj = [allNBack_subj, keys(keys>=0)];
            allTTE_subj = [allTTE_subj, vals(keys>=0)];
        end
        
        uniqueTTE = unique(allTTE_subj);
        slopes = nan(numel(uniqueTTE), 1);
        for t = 1:numel(uniqueTTE)
            currentTTE = uniqueTTE(t);
            nBacks = allNBack_subj(allTTE_subj == currentTTE);
            
            [uniqueNBack, ~, idx] = unique(nBacks);
            counts = accumarray(idx, 1);
            
            if max(counts) > min(counts)
                normCounts = (counts - min(counts)) / (max(counts) - min(counts));
            else
                normCounts = ones(size(counts));
            end
            
            if numel(uniqueNBack) < 2
				p = polyfit([0; uniqueNBack(:)], [0; normCounts(:)], 1);
            else
				p = polyfit(uniqueNBack(:), normCounts(:), 1);
            end
            slopes(t) = p(1);
            
            fieldName = sprintf('TTE_%d', currentTTE);
            if ~isfield(subjectSlopesByTTE, fieldName)
                subjectSlopesByTTE.(fieldName) = nan(1, nSubjects); 
            end
            subjectSlopesByTTE.(fieldName)(iS) = slopes(t);
        end
        
        all_subject_slopes_by_tte{iS} = slopes; % NEW: Store all slopes for this subject
        
        valid = ~isnan(slopes);
        if any(valid)
            hPlots(iS) = plot(ax, uniqueTTE(valid) + jitter_values(iS), slopes(valid), '-o', ...
                'Color', [colors{iS}, 0.5], 'LineWidth', 1, ...
                'MarkerFaceColor', colors{iS}, 'MarkerSize', 3);
        end
    end
    
    % --- Part 2: Calculate and plot subject-wise grand average and SEM ---
    fieldNames = fieldnames(subjectSlopesByTTE);
    if isempty(fieldNames), csvData = {}; statsData = {}; return; end
    
    all_TTE_vals = cellfun(@(s) str2double(extractAfter(s, 'TTE_')), fieldNames);
    [sortedTTEs, sortIdx] = sort(all_TTE_vals);
    sortedFieldNames = fieldNames(sortIdx);
    
    avgSlopes = nan(numel(sortedFieldNames), 1);
    semSlopes = nan(numel(sortedFieldNames), 1);
    
    csvHeader = {'TTE', 'AvgSlope_SubjectWise', 'SEMSlope_SubjectWise'};
    subject_headers = cell(1, nSubjects);
    for iS=1:nSubjects
        subject_headers{iS} = sprintf('Subject_%d_Slope', iS);
    end
    csvHeader = [csvHeader, subject_headers];
    csvData = csvHeader;
    
    for f = 1:numel(sortedFieldNames)
        currentFieldName = sortedFieldNames{f};
        slopesForTTE_across_subjects = subjectSlopesByTTE.(currentFieldName);
        
        if numel(slopesForTTE_across_subjects) < nSubjects
            slopesForTTE_across_subjects(end+1:nSubjects) = NaN;
        end
        
        avgSlopes(f) = nanmean(slopesForTTE_across_subjects);
        valid_subjects = sum(~isnan(slopesForTTE_across_subjects));
        if valid_subjects > 1
            semSlopes(f) = nanstd(slopesForTTE_across_subjects) / sqrt(valid_subjects);
        else
            semSlopes(f) = 0;
        end
        
        row = {sortedTTEs(f), avgSlopes(f), semSlopes(f)};
        row = [row, num2cell(slopesForTTE_across_subjects)];
        csvData = [csvData; row];
    end
    
    hPlots(nSubjects + 1) = errorbar(ax, sortedTTEs, avgSlopes, semSlopes, '-o', ...
        'Color', 'k', 'LineWidth', 2.0, 'MarkerFaceColor', 'k', ...
        'MarkerSize', 3, 'CapSize', 5);
        
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
    % %%% --- NEW/MODIFIED CODE START: Summary Stats, Plots, & T-Tests --- %%% %
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
    
    numCols = size(csvData, 2);
    
    % --- 1. Calculate Overall TTE Avg & SEM for each subject ---
    subjectOverallAvgSlopes = nan(1, nSubjects);
    subjectOverallSEMSlopes = nan(1, nSubjects); % NEW
    
    for iS = 1:nSubjects
        slopes_for_subject = all_subject_slopes_by_tte{iS}; % Get all slopes
        subjectOverallAvgSlopes(iS) = nanmean(slopes_for_subject);
        
        n_valid_slopes = sum(~isnan(slopes_for_subject));
        if n_valid_slopes > 1
            subjectOverallSEMSlopes(iS) = nanstd(slopes_for_subject) / sqrt(n_valid_slopes);
        else
            subjectOverallSEMSlopes(iS) = 0;
        end
    end
    
    % --- 2. Calculate Grand Overall Avg (from subject averages) ---
    grandOverallAvgSlope = nanmean(subjectOverallAvgSlopes);
    valid_subject_avgs = sum(~isnan(subjectOverallAvgSlopes));
    grandOverallSEMSlope = 0;
    if valid_subject_avgs > 1
        grandOverallSEMSlope = nanstd(subjectOverallAvgSlopes) / sqrt(valid_subject_avgs);
    end
    
    % --- 3. Add summary rows to csvData ---
    spacerRow = cell(1, numCols); spacerRow{1} = '---';
    csvData = [csvData; spacerRow];
    
    overallTTE_row = cell(1, numCols);
    overallTTE_row{1} = 'Overall_TTE_Avg_Per_Subject';
    overallTTE_row(4:end) = num2cell(subjectOverallAvgSlopes);
    csvData = [csvData; overallTTE_row];
    
    grandOverall_row = cell(1, numCols);
    grandOverall_row{1} = 'Grand_Overall_Avg';
    grandOverall_row{2} = grandOverallAvgSlope;
    grandOverall_row{3} = grandOverallSEMSlope;
    csvData = [csvData; grandOverall_row];

    % --- 4. NEW: Perform T-Tests and create statsData ---
    statsData = {'Subject', 'T_Statistic', 'P_Value', 'DegreesOfFreedom', 'Mean_Slope', 'StdErr_Slope'};
    
    % Subject-wise t-tests
    for iS = 1:nSubjects
        slopes_to_test = all_subject_slopes_by_tte{iS};
        slopes_to_test = slopes_to_test(~isnan(slopes_to_test)); % Remove NaNs
        
        tstat = NaN; p = NaN; df = NaN;
        if numel(slopes_to_test) > 1
            [~, p, ~, stats] = ttest(slopes_to_test); % Test against mean=0
            tstat = stats.tstat;
            df = stats.df;
        end
        statsData = [statsData; {sprintf('Subject_%d', iS), tstat, p, df, ...
                     subjectOverallAvgSlopes(iS), subjectOverallSEMSlopes(iS)}];
    end
    
    % Overall t-test (on the subject averages)
    slopes_to_test_overall = subjectOverallAvgSlopes;
    slopes_to_test_overall = slopes_to_test_overall(~isnan(slopes_to_test_overall));
    
    tstat_ov = NaN; p_ov = NaN; df_ov = NaN;
    if numel(slopes_to_test_overall) > 1
        [~, p_ov, ~, stats_ov] = ttest(slopes_to_test_overall); % Test against mean=0
        tstat_ov = stats_ov.tstat;
        df_ov = stats_ov.df;
    end
    statsData = [statsData; {'Overall', tstat_ov, p_ov, df_ov, ...
                 grandOverallAvgSlope, grandOverallSEMSlope}];

    % --- 5. NEW: Plot summary dots on the right (at TTE=20) ---
    x_position_for_dots = 20;
    
    % Plot individual subject average dots
    for iS = 1:nSubjects
        % %%% --- MODIFIED: Added jitter to x-position --- %%%
        errorbar(ax, x_position_for_dots + jitter_values(iS), ...
                 subjectOverallAvgSlopes(iS), subjectOverallSEMSlopes(iS), ...
                 'o', 'Color', colors{iS}, 'MarkerFaceColor', colors{iS}, ...
                 'MarkerSize', 5, 'LineWidth', 1.5, 'CapSize', 4, ...
                 'HandleVisibility', 'off');
    end
    
    % Plot overall average "line" (as a black square)
    if ~isnan(grandOverallAvgSlope)
        errorbar(ax, x_position_for_dots, ... % This one remains centered
                 grandOverallAvgSlope, grandOverallSEMSlope, ...
                 'sk', 'MarkerFaceColor', 'k', 'MarkerSize', 8, ...
                 'LineWidth', 1.5, 'CapSize', 5, 'HandleVisibility', 'off');
    end
    
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
    % %%% --- NEW CODE END --- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
    
    
    % --- Finalize plot ---
    xlabel(ax, 'TTE (Trial of Error)');
    ylabel(ax, 'Normalized Slope');
    title(ax, 'Trend of N-Back Error Patterns vs. TTE');
    set(ax, 'FontSize', 12);
    
    grid(ax, 'off');
    ylim(ax, [-0.4 1.1]);
    set(ax, 'YTick', -0.4:0.2:1.0);
    
    % NEW: Adjust xlim to include TTE=20
    min_x = min(sortedTTEs);
    xlim(ax, [min_x - 1, x_position_for_dots + 1]); 
    
    
    legendLabelsWithAvg = [legend_labels(1:nSubjects); {'Average (Subject-wise SEM)'}];
    validPlots = isgraphics(hPlots);
    legend(hPlots(validPlots), legendLabelsWithAvg(validPlots), 'Location', 'eastoutside');
    hold(ax, 'off');
end



function plot_cr_nback_heatmap(metrics_mt)
% plot_cr_nback_heatmap_final Creates and exports a final heatmap of n-back data.
%
% This function generates a single figure containing a heatmap of N-Back
% Proportion vs. TTE.
%
% Key Features:
% - Guarantees each data cell is a square using daspect.
% - Uses a smooth, interpolated 256-color custom colormap.
% - Renders missing data (NaNs) as fully transparent.
% - Automatically creates 'CR_nBack_Heatmap' directory and saves all outputs.
%
% Input:
%   metrics_mt - A cell array where each cell contains the data for one subject.

% --- Setup for File Export ---
outputDir = 'CR_nBack_Heatmap';
if ~exist(outputDir, 'dir')
   mkdir(outputDir);
   fprintf('Created directory: %s\n', outputDir);
end

% --- Figure Creation ---
hFig = figure('Position', [100, 100, 550, 450], 'Color', 'w');
ax = gca; % Get current axes

% --- Generate heatmap and get data for CSV ---
[heatmapMatrix, csvData] = generateHeatmapData(metrics_mt);

if isempty(heatmapMatrix)
    warning('No data found to plot.');
    close(hFig);
    return;
end

% --- Define and Interpolate Custom Colormap ---
% 1. Define the 6 anchor colors from the user.
anchorCMap = [
    82, 147, 201;   % #5293c9 (for low values, near 0)
    151, 206, 255;  % #97ceff
    178, 208, 232;  % #b2d0e8
    236, 173, 196;  % #ecadc4
    231, 151, 180;  % #e797b4
    223, 129, 165   % #df81a5 (for high values, near 1)
] / 255;

% 2. Create a high-resolution (256-color) map by interpolating between the anchors.
original_positions = linspace(0, 1, size(anchorCMap, 1));
new_positions = linspace(0, 1, 256);
smoothCMap = interp1(original_positions, anchorCMap, new_positions, 'linear');


% --- Plotting with IMAGESC ---
h_img = imagesc(ax, heatmapMatrix);

% Make NaN values transparent.
set(h_img, 'AlphaData', ~isnan(heatmapMatrix));


% --- Finalize Plot Formatting ---
daspect(ax, [1 1 1]); % Set data aspect ratio to 1:1:1 to make cells square
colormap(ax, smoothCMap);
colorbar(ax);
title('N-Back Proportion by TTE');
xlabel('n-Back (Memory Lag)');
ylabel('TTE (Trial of Error)');
set(ax, 'FontSize', 12, 'YDir', 'normal', 'Color', 'none'); % Use transparent background


% --- Save Data and Figures ---
try
    % 1. Save heatmap data to CSV
    heatmapTable = cell2table(csvData(2:end,:), 'VariableNames', csvData(1,:));
    heatmapCsvFilename = fullfile(outputDir, 'heatmap_data.csv');
    writetable(heatmapTable, heatmapCsvFilename);
    fprintf('Successfully saved data to: %s\n', heatmapCsvFilename);

    % 2. Save figure as PNG and EPS
    baseFilename = fullfile(outputDir, 'heatmap_plot');
    saveas(hFig, [baseFilename, '.png']);
    fprintf('Successfully saved figure to: %s.png\n', baseFilename);
    print(hFig, [baseFilename '.eps'], '-depsc2', '-vector','-r300');
    fprintf('Successfully saved figure to: %s.eps\n', baseFilename);
catch E
    warning('Could not save files. Error: %s', E);
end

end

% --- NESTED FUNCTION: Generate Heatmap Data (no changes needed here) ---
function [heatmapMatrix, csvData] = generateHeatmapData(metrics_mt)
    nSubjects = numel(metrics_mt);
    TTEData = struct();
    csvData = {'TTE', 'nBack', 'AvgNormFreq'};
    heatmapMatrix = [];

    for iS = 1:nSubjects
        allNBack = [];
        allTTE = [];
        for iD = 1:numel(metrics_mt{iS})
            current_nBack_data = metrics_mt{iS}(iD).nBack;
            if isempty(current_nBack_data), continue; end
            if iscell(current_nBack_data)
                keys = cellfun(@(c) c{1}, current_nBack_data);
                vals = cellfun(@(c) c{2}, current_nBack_data);
            else
                keys = current_nBack_data(:, 1)';
                vals = current_nBack_data(:, 2)';
            end
            validIndices = keys >= 0;
            allNBack = [allNBack, keys(validIndices)];
            allTTE = [allTTE, vals(validIndices)];
        end

        uniqueTTE = unique(allTTE);
        for t = 1:numel(uniqueTTE)
            currentTTE = uniqueTTE(t);
            nBackAtCurrentTTE = allNBack(allTTE == currentTTE);
            if isempty(nBackAtCurrentTTE), continue; end
            [uniqueNBack, ~, idx] = unique(nBackAtCurrentTTE);
            counts = accumarray(idx, 1);
            relativeFrequency = counts / numel(nBackAtCurrentTTE);
            minFreq = min(relativeFrequency);
            maxFreq = max(relativeFrequency);
            if maxFreq > minFreq
                normalizedFrequency = (relativeFrequency - minFreq) / (maxFreq - minFreq);
            else
                normalizedFrequency = ones(size(relativeFrequency)) * 0.5;
            end
            fieldName = sprintf('TTE_%d', currentTTE);
            if ~isfield(TTEData, fieldName), TTEData.(fieldName) = []; end
            TTEData.(fieldName) = [TTEData.(fieldName); uniqueNBack(:), normalizedFrequency(:)];
        end
    end

    fieldNames = fieldnames(TTEData);
    if isempty(fieldNames), return; end

    all_TTE_vals = cellfun(@(s) str2double(extractAfter(s, 'TTE_')), fieldNames);
    maxNBack = 0;
    for f = 1:numel(fieldNames)
        data = TTEData.(fieldNames{f});
        if ~isempty(data) && ~isempty(data(:,1))
             maxNBack = max(maxNBack, max(data(:,1)));
        end
    end
    maxTTE = max(all_TTE_vals);
    if maxTTE == 0 || maxNBack == 0, return; end

    heatmapMatrix = nan(maxTTE, maxNBack);

    for f = 1:numel(fieldNames)
        currentTTE = str2double(extractAfter(fieldNames{f}, 'TTE_'));
        data = TTEData.(fieldNames{f});
        if ~isempty(data)
            uniqueNBack = unique(data(:, 1));
            for n = 1:numel(uniqueNBack)
                currentNBack = uniqueNBack(n);
                avgFreq = mean(data(data(:, 1) == currentNBack, 2));
                if currentTTE > 0 && currentNBack > 0 && currentTTE <= size(heatmapMatrix,1) && currentNBack <= size(heatmapMatrix,2)
                    heatmapMatrix(currentTTE, currentNBack) = avgFreq;
                    csvData = [csvData; {currentTTE, currentNBack, avgFreq}];
                end
            end
        end
    end

    for r = 1:size(heatmapMatrix, 1) % Rows are TTE
        for c = 1:size(heatmapMatrix, 2) % Columns are nBack
            if c > r
                heatmapMatrix(r, c) = nan;
            end
        end
    end
end


function plot_cr_nback_unstacked(metrics_mt)
% plot_cr_nback_unstacked Creates and exports a specific multi-panel figure.
%
% This function generates a figure with a single vertical column of subplots
% for TTE values from 2 to 8. Each subplot shows the min-max normalized 
% frequency distribution of n-back values.
%
% Key Features:
% - Plots only for TTE = 2 through 8 in a single column.
% - Figure size is 800x400, x-axis is limited to 10, and subplots have no titles.
% - Plots individual subject data and the cross-subject average with error bars.
% - Automatically creates 'CR_nBack_Unstacked' directory for all outputs.
% - Saves the plot as PNG and EPS files and exports data to a CSV file.
%
% Input:
%   metrics_mt - A cell array where each cell contains the data for one subject.

% --- Configuration ---
custom_colors_rgb = {
    [245, 124, 110]/255; [242, 181, 110]/255; [251, 231, 158]/255;
    [132, 195, 183]/255; [136, 215, 218]/255; [113, 184, 237]/255
};

% --- Data Aggregation ---
nSubjects = numel(metrics_mt);
if nSubjects > numel(custom_colors_rgb)
    error('Not enough custom colors for the number of subjects provided.');
end
TTEData = struct(); 

for iS = 1:nSubjects
    allNBack = [];
    allTTE = [];
    
    for iD = 1:numel(metrics_mt{iS})
        current_nBack_data = metrics_mt{iS}(iD).nBack;
        if isempty(current_nBack_data), continue; end
        
        if iscell(current_nBack_data)
            keys = cellfun(@(c) c{1}, current_nBack_data);
            vals = cellfun(@(c) c{2}, current_nBack_data);
        else
            keys = current_nBack_data(:, 1)';
            vals = current_nBack_data(:, 2)';
        end
        
        validIndices = keys >= 0;
        allNBack = [allNBack, keys(validIndices)];
        allTTE = [allTTE, vals(validIndices)];
    end
    
    uniqueTTE = unique(allTTE);
    for t = 1:numel(uniqueTTE)
        currentTTE = uniqueTTE(t);
        fieldName = sprintf('TTE_%d', currentTTE);
        
        if ~isfield(TTEData, fieldName)
            TTEData.(fieldName) = cell(nSubjects, 1);
        end
        
        nBackAtCurrentTTE = allNBack(allTTE == currentTTE);
        if isempty(nBackAtCurrentTTE), continue; end
        
        [uniqueNBack, ~, idx] = unique(nBackAtCurrentTTE);
        counts = accumarray(idx, 1);
        relativeFrequency = counts / numel(nBackAtCurrentTTE);
        
        minFreq = min(relativeFrequency);
        maxFreq = max(relativeFrequency);
        if (maxFreq > minFreq)
             normalizedFrequency = (relativeFrequency - minFreq) / (maxFreq - minFreq);
        else
             normalizedFrequency = ones(size(relativeFrequency)) * 0.5;
        end
        
        TTEData.(fieldName){iS} = [uniqueNBack(:), normalizedFrequency(:)];
    end
end

% --- Setup for File Export ---
outputDir = 'CR_nBack_Unstacked';
if ~exist(outputDir, 'dir')
   mkdir(outputDir);
   fprintf('Created directory: %s\n', outputDir);
end
csvData = {'TTE', 'nBack', 'AvgNormFreq', 'SEMNormFreq'};

% --- Figure and Subplot Generation ---

% Define the specific TTEs to plot and filter for available data
targetTTEs = 2:8;
allAvailableFieldNames = fieldnames(TTEData);
plotFieldNames = {};
for i = 1:numel(targetTTEs)
    fieldName = sprintf('TTE_%d', targetTTEs(i));
    if ismember(fieldName, allAvailableFieldNames)
        plotFieldNames{end+1} = fieldName;
    end
end

nSubplots = numel(plotFieldNames);
if nSubplots == 0
    warning('No data found for TTEs 2 through 8.');
    return;
end

% Set layout for a single column
nCols = 1;
nRows = nSubplots;

% Create the figure with the specified dimensions
hFig = figure('Position', [100, 100, 400, 400], 'Color', 'w');

for i = 1:nSubplots
    ax = subplot(nRows, nCols, i);
    hold(ax, 'on');
    
    currentFieldName = plotFieldNames{i};
    currentTTE = str2double(extractAfter(currentFieldName, 'TTE_'));
    
    % --- PLOT 1: Individual subject data (drawn first) ---
    for iS = 1:nSubjects
        subjectData = TTEData.(currentFieldName){iS};
        if ~isempty(subjectData)
            plot(ax, subjectData(:,1), subjectData(:,2), '-', 'Color', [custom_colors_rgb{iS}, 0.5], 'LineWidth', 1.2);
        end
    end
    
    % --- PLOT 2: Average and Error Bars (drawn on top) ---
    allSubjectDataForTTE = vertcat(TTEData.(currentFieldName){:});
    
    if ~isempty(allSubjectDataForTTE)
        uniqueNBack = unique(allSubjectDataForTTE(:, 1));
        avgFrequencies = arrayfun(@(n) mean(allSubjectDataForTTE(allSubjectDataForTTE(:, 1) == n, 2)), uniqueNBack);
        semFrequencies = arrayfun(@(n) std(allSubjectDataForTTE(allSubjectDataForTTE(:, 1) == n, 2)) / sqrt(sum(allSubjectDataForTTE(:, 1) == n)), uniqueNBack);
        
        for j = 1:numel(uniqueNBack)
            csvData = [csvData; {currentTTE, uniqueNBack(j), avgFrequencies(j), semFrequencies(j)}];
        end
        
        plot(ax, uniqueNBack, avgFrequencies, 'k-', 'LineWidth', 1.0);
        errorbar(ax, uniqueNBack, avgFrequencies, semFrequencies, 'k.', 'CapSize', 0, 'LineWidth', 1);
    end
    
    % --- Finalize Subplot Formatting ---
    grid(ax, 'off');
    xlim(ax, [0, 8]); % Set fixed x-axis limit
    ylim(ax, [0, 1.1]);
	yticks(ax, [0, 1]);
    
    % Add labels intelligently
    if i == round(nSubplots / 2)
        ylabel(ax, 'Norm. Freq.');
    end
    if i == nSubplots
        xlabel(ax, 'n-Back (Memory Lag)');
    end
    
    set(ax, 'FontSize', 10);
    hold(ax, 'off');
end

% --- Save Data and Figures ---
try
    plotDataTable = cell2table(csvData(2:end,:), 'VariableNames', csvData(1,:));
    csvFilename = fullfile(outputDir, 'nback_unstacked_data.csv');
    writetable(plotDataTable, csvFilename);
    fprintf('Successfully saved data to: %s\n', csvFilename);
    
    baseFilename = fullfile(outputDir, 'nback_distribution_plot2');
    saveas(hFig, [baseFilename, '.png']);
    fprintf('Successfully saved figure to: %s.png\n', baseFilename);
    print(hFig, [baseFilename '.eps'], '-depsc2', '-vector','-r300');
    fprintf('Successfully saved figure to: %s.eps\n', baseFilename);
catch E
    warning('Could not save files. Error: %s', E);
end

end


function plot_cr_nback(metrics_mt)
% plot_cr_nback Creates and exports a two-panel figure analyzing n-back data.
%
% This function generates a figure with two subplots:
%   1. N-Back Proportion by TTE: Visualizes normalized frequency of n-back
%      trials stratified by TTE (individual and average lines).
%   2. N-Back Frequency Slope vs. TTE: Shows individual subject slopes and the
%      across-subject average slope with SEM error bars.
%
% The function automatically creates a 'CR_nBack' directory and saves the
% plot as PNG and EPS files, and the underlying data as two CSV files.
%
% Input:
%   metrics_mt - A cell array where each cell contains the data for one subject.

% --- Configuration ---
subject_id_list_hardcoded = {
    'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan'
};
custom_colors_rgb = {
    [245, 124, 110]/255; [242, 181, 110]/255; [251, 231, 158]/255;
    [132, 195, 183]/255; [136, 215, 218]/255; [113, 184, 237]/255;
    [184, 174, 234]/255; [242, 168, 218]/255
};
legend_labels = cell(size(subject_id_list_hardcoded));
for i = 1:numel(subject_id_list_hardcoded)
    name = subject_id_list_hardcoded{i};
    if ~isempty(name), legend_labels{i} = sprintf('Subject %c', name(1)); end
end
nSubjects = numel(metrics_mt);
if nSubjects > numel(custom_colors_rgb), error('Not enough custom colors for the number of subjects provided.'); end
if nSubjects > numel(legend_labels), warning('More subjects in data than in legend list; legend may be incomplete.'); end

% --- Setup for File Export ---
outputDir = 'CR_nBack';
if ~exist(outputDir, 'dir')
   mkdir(outputDir);
   fprintf('Created directory: %s\n', outputDir);
end

% --- Figure Creation ---
hFig = figure('Position', [100, 100, 300, 600], 'Color', 'w');

% --- Panel 1: nBack proportion of trials subdivided by TTE with averages ---
ax1 = subplot(2, 1, 1);
proportionsCsvData = plotProportions(ax1, metrics_mt, nSubjects, custom_colors_rgb);

% --- Panel 2: nBack slopes ---
ax2 = subplot(2, 1, 2);
slopesCsvData = plotSlopes(ax2, metrics_mt, nSubjects, custom_colors_rgb, legend_labels);

% --- Save Data and Figures ---
try
    % 1. Save proportions data to CSV
    proportionsTable = cell2table(proportionsCsvData(2:end,:), 'VariableNames', proportionsCsvData(1,:));
    proportionsCsvFilename = fullfile(outputDir, 'proportions_data.csv');
    writetable(proportionsTable, proportionsCsvFilename);
    fprintf('Successfully saved data to: %s\n', proportionsCsvFilename);
    
    % 2. Save slopes data to CSV
    slopesTable = cell2table(slopesCsvData(2:end,:), 'VariableNames', slopesCsvData(1,:));
    slopesCsvFilename = fullfile(outputDir, 'slopes_data.csv');
    writetable(slopesTable, slopesCsvFilename);
    fprintf('Successfully saved data to: %s\n', slopesCsvFilename);

    % 3. Save figure as PNG and EPS
    baseFilename = fullfile(outputDir, 'nback_summary_plot');
    saveas(hFig, [baseFilename, '.png']);
    fprintf('Successfully saved figure to: %s.png\n', baseFilename);
    saveas(hFig, [baseFilename, '.eps'], 'epsc');
    fprintf('Successfully saved figure to: %s.eps\n', baseFilename);
catch E
    warning('Could not save files. Error: %s', E);
end

end

% --- NESTED FUNCTION 1: Plot Proportions ---
function csvData = plotProportions(ax, metrics_mt, nSubjects, colors)
    hold(ax, 'on');
    TTEData = struct();
    csvData = {'TTE', 'nBack', 'AvgNormFreq'};
    
    % First loop: Plot individual subject data
    for iS = 1:nSubjects
        allNBack = [];
        allTTE = [];
        for iD = 1:numel(metrics_mt{iS})
            current_nBack_data = metrics_mt{iS}(iD).nBack;
            if isempty(current_nBack_data), continue; end
            if iscell(current_nBack_data)
                keys = cellfun(@(c) c{1}, current_nBack_data);
                vals = cellfun(@(c) c{2}, current_nBack_data);
            else
                keys = current_nBack_data(:, 1)';
                vals = current_nBack_data(:, 2)';
            end
            
            validIndices = keys >= 0;
            allNBack = [allNBack, keys(validIndices)];
            allTTE = [allTTE, vals(validIndices)];
        end
        
        uniqueTTE = unique(allTTE);
        for t = 1:numel(uniqueTTE)
            currentTTE = uniqueTTE(t);
            nBackAtCurrentTTE = allNBack(allTTE == currentTTE);
            if isempty(nBackAtCurrentTTE), continue; end
            
            [uniqueNBack, ~, idx] = unique(nBackAtCurrentTTE);
            counts = accumarray(idx, 1);
            relativeFrequency = counts / numel(nBackAtCurrentTTE);
            
            minFreq = min(relativeFrequency);
            maxFreq = max(relativeFrequency);
            if maxFreq > minFreq
                normalizedFrequency = (relativeFrequency - minFreq) / (maxFreq - minFreq);
            else
                normalizedFrequency = ones(size(relativeFrequency)) * 0.5;
            end

            plot(ax, uniqueNBack, normalizedFrequency + currentTTE, '-', 'LineWidth', 1, 'Color', [colors{iS}, 0.4]);
            
            fieldName = sprintf('TTE_%d', currentTTE);
            if ~isfield(TTEData, fieldName), TTEData.(fieldName) = []; end
            TTEData.(fieldName) = [TTEData.(fieldName); uniqueNBack(:), normalizedFrequency(:)];
        end
    end
    
    % Second loop: Plot averages and collect data for CSV
    fieldNames = sort(fieldnames(TTEData));
    for f = 1:numel(fieldNames)
        currentTTE = str2double(extractAfter(fieldNames{f}, 'TTE_'));
        data = TTEData.(fieldNames{f});
        if ~isempty(data)
            uniqueNBack = unique(data(:, 1));
            avgFrequencies = arrayfun(@(n) mean(data(data(:, 1) == n, 2)), uniqueNBack);
            
            % Plot the average line
            plot(ax, uniqueNBack, avgFrequencies + currentTTE, 'k-', 'LineWidth', 1.0);
            
            % Add data to CSV cell array
            for k=1:numel(uniqueNBack)
                csvData = [csvData; {currentTTE, uniqueNBack(k), avgFrequencies(k)}];
            end
        end
    end
    
    % Finalize plot
    xlabel(ax, 'n-Back (Memory Lag)');
    ylabel(ax, 'Norm. Freq. Split by TTE');
    title(ax, 'N-Back Proportion by TTE');
    set(ax, 'TickLength', [0 0], 'FontSize', 14);
    grid(ax, 'off');
    hold(ax, 'off');
end

% --- NESTED FUNCTION 2: Plot Slopes ---
function csvData = plotSlopes(ax, metrics_mt, nSubjects, colors, legend_labels)
    hold(ax, 'on');
    hPlots = gobjects(nSubjects + 1, 1); % +1 for the average plot handle
    allSlopesByTTE = struct(); % Store all slopes for averaging
    
    % CSV Headers
    csvHeader = {'TTE', 'AvgSlope', 'SEMSlope'};
    for iS=1:nSubjects, csvHeader{end+1} = sprintf('Subject_%d_Slope', iS); end
    
    % *** CORRECTION HERE: Initialize csvData without extra curly braces ***
    csvData = csvHeader;
    
    % First loop: Calculate and plot individual slopes
    for iS = 1:nSubjects
        allNBack = [];
        allTTE = [];
        for iD = 1:numel(metrics_mt{iS})
            current_nBack_data = metrics_mt{iS}(iD).nBack;
            if isempty(current_nBack_data), continue; end
            if iscell(current_nBack_data)
                keys = cellfun(@(c) c{1}, current_nBack_data);
                vals = cellfun(@(c) c{2}, current_nBack_data);
            else
                keys = current_nBack_data(:, 1)';
                vals = current_nBack_data(:, 2)';
            end
            validIndices = keys >= 0;
            allNBack = [allNBack, keys(validIndices)];
            allTTE = [allTTE, vals(validIndices)];
        end
        
        uniqueTTE = unique(allTTE);
        slopes = nan(numel(uniqueTTE), 1);
        
        for t = 1:numel(uniqueTTE)
            currentTTE = uniqueTTE(t);
            nBackAtCurrentTTE = allNBack(allTTE == currentTTE);
            
            if numel(unique(nBackAtCurrentTTE)) < 2, continue; end
            
            [uniqueNBack, ~, idx] = unique(nBackAtCurrentTTE);
            counts = accumarray(idx, 1);
            p = polyfit(uniqueNBack, counts, 1);
            slopes(t) = p(1);
            
            % Store slope for averaging
            fieldName = sprintf('TTE_%d', currentTTE);
            if ~isfield(allSlopesByTTE, fieldName), allSlopesByTTE.(fieldName) = nan(1, nSubjects); end
            allSlopesByTTE.(fieldName)(iS) = p(1);
        end
        
        validIndices = ~isnan(slopes);
        if any(validIndices)
            hPlots(iS) = plot(ax, uniqueTTE(validIndices), slopes(validIndices), '-o', ...
                'Color', [colors{iS}, 0.6], 'LineWidth', 1, ...
                'MarkerFaceColor', colors{iS}, 'MarkerSize', 3);
        end
    end
    
    % Second loop: Plot average slope with error bars
    fieldNames = sort(fieldnames(allSlopesByTTE));
    avgTTEs = nan(numel(fieldNames), 1);
    avgSlopes = nan(numel(fieldNames), 1);
    semSlopes = nan(numel(fieldNames), 1);
    
    for f = 1:numel(fieldNames)
        currentTTE = str2double(extractAfter(fieldNames{f}, 'TTE_'));
        slopesForTTE = allSlopesByTTE.(fieldNames{f});
        validSlopes = slopesForTTE(~isnan(slopesForTTE));
        
        avgTTEs(f) = currentTTE;
        avgSlopes(f) = mean(validSlopes);
        semSlopes(f) = std(validSlopes) / sqrt(numel(validSlopes));
        
        % Add row to CSV data
        row = {currentTTE, avgSlopes(f), semSlopes(f)};
        row = [row, num2cell(slopesForTTE)];
        csvData = [csvData; row]; % This line will now work correctly
    end
    
    % Plot average with error bars
    hPlots(nSubjects + 1) = errorbar(ax, avgTTEs, avgSlopes, semSlopes, '-o', ...
        'Color', 'k', 'LineWidth', 1.0, 'MarkerFaceColor', 'k', ...
        'MarkerSize', 2, 'CapSize', 5);
        
    % Finalize plot
    xlabel(ax, 'TTE');
    ylabel(ax, 'Slope (Counts / n-Back)');
    set(ax, 'YScale', 'log', 'FontSize', 14);
    title(ax, 'N-Back Frequency Slope vs. TTE');
    yticks(ax, 10.^(-3:3));
    yticklabels(ax, {'10^{-3}', '10^{-2}', '10^{-1}', '10^{0}', '10^{1}', '10^{2}', '10^{3}'});
    axis(ax, 'padded');
    grid(ax, 'off');
    
    % Update legend to include 'Average'
    % validPlots = isgraphics(hPlots);
    % legendLabelsWithAvg = [legend_labels(1:nSubjects), {'Average'}];
    % legend(hPlots(validPlots), legendLabelsWithAvg(validPlots), 'Location', 'best');
    hold(ax, 'off');
end

function plot_wm_RT_by_condition_grouped(metrics_mt)
% plot_wm_RT_by_condition_grouped Plots WM Reaction Time for grouped conditions.
%   - Main condition points: Mean of ALL SESSIONS pooled for that condition, black dot (size 6) with error bar.
%   - Pairwise stats: Independent t-test (ttest2) on pooled session data between conditions in a pair.
%   - P-values, t-stats, Ns, and df displayed in command window. Stars displayed on figure.
%   - Subject data: Jittered colored dots (size 6), representing subject's mean RT for that condition.
%   - Connecting lines: For each subject, lines connect their means between paired conditions, using subject's color. (Currently Commented Out)
%   - Figure 600x300, Font 14, Legend right (no box), No grid.
%   - X-axis title "WM Condition". X-ticks are condition labels.
%   - Y-axis title "Average Reaction Time (s)".
%   - Saves as PNG.

% --- Configuration ---
plot_font_size = 14;
stat_font_size = plot_font_size - 3; 
% mean_se_text_font_size = plot_font_size - 5; % For Mean +/- SE text (if uncommented)
figure_width = 500; 
figure_height = 300; 
condition_labels = {'Overall', 'Low TDS', 'High TDS', '2 Distr', '3 Distr', 'No PSD', 'PSD Pres'};
num_conditions = length(condition_labels);

% Define X positions for conditions (grouped)
x_group_spacing = 1.2;  
x_within_pair_spacing = 0.8;
x_positions = zeros(1, num_conditions);
current_x = 1;
x_positions(1) = current_x; % Overall
current_x = current_x + x_group_spacing;
x_positions(2) = current_x; x_positions(3) = current_x + x_within_pair_spacing; % TDS
current_x = current_x + x_within_pair_spacing + x_group_spacing;
x_positions(4) = current_x; x_positions(5) = current_x + x_within_pair_spacing; % Distr
current_x = current_x + x_within_pair_spacing + x_group_spacing;
x_positions(6) = current_x; x_positions(7) = current_x + x_within_pair_spacing; % PSD

subject_id_list_hardcoded = {
    'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan' 
}; 
custom_colors_rgb = {
    [245, 124, 110]/255; [242, 181, 110]/255; [251, 231, 158]/255;
    [132, 195, 183]/255; [136, 215, 218]/255; [113, 184, 237]/255;
    [184, 174, 234]/255; [242, 168, 218]/255 
}; 
num_subjects_to_process = length(metrics_mt);
if num_subjects_to_process == 0; disp('Input metrics_mt is empty.'); return; end

% --- Data Aggregation ---
% subject_condition_means: stores one mean RT per subject per condition
subject_condition_means = nan(num_subjects_to_process, num_conditions);
subject_legend_labels = cell(1, num_subjects_to_process);
% pooled_session_RTs_by_condition: cell array, each cell contains ALL session RTs from ALL subjects for one condition
pooled_session_RTs_by_condition = cell(1, num_conditions); 
for k=1:num_conditions; pooled_session_RTs_by_condition{k} = []; end

for iS = 1:num_subjects_to_process
    if iS <= length(subject_id_list_hardcoded) && ~isempty(subject_id_list_hardcoded{iS})
        original_id = subject_id_list_hardcoded{iS}; initial = upper(original_id(1));
        subject_legend_labels{iS} = sprintf('Subject %c', initial);
    else; subject_legend_labels{iS} = sprintf('Subj %d', iS); end
    
    if iS > length(metrics_mt) || isempty(metrics_mt{iS}); continue; end
    
    session_means_for_this_subject_conditions = cell(1, num_conditions); 
    for k=1:num_conditions; session_means_for_this_subject_conditions{k} = []; end
    
    for iD = 1:length(metrics_mt{iS}) 
        if ~isstruct(metrics_mt{iS}(iD)); continue; end
        session_data = metrics_mt{iS}(iD);
        
        get_session_mean = @(field_name) extract_session_mean_safely(session_data, field_name);
        % IMPORTANT: Update these field names to your RT field names
        field_map = {'RT_MeanSE_wm', 'RT_MeanSE_LowTDS_wm', 'RT_MeanSE_HighTDS_wm', ...
                     'RT_MeanSE_2dist_wm', 'RT_MeanSE_3dist_wm', ...
                     'RT_MeanSE_noPostSampleDist_wm', 'RT_MeanSE_yesPostSampleDist_wm'};
        
        for k_cond = 1:num_conditions
            session_cond_RT = get_session_mean(field_map{k_cond}); % Changed from acc
            if ~isnan(session_cond_RT)
                session_means_for_this_subject_conditions{k_cond} = [session_means_for_this_subject_conditions{k_cond}, session_cond_RT];
                pooled_session_RTs_by_condition{k_cond} = [pooled_session_RTs_by_condition{k_cond}, session_cond_RT]; % Add to grand pool
            end
        end
    end 
    
    for k_cond = 1:num_conditions
        if ~isempty(session_means_for_this_subject_conditions{k_cond})
            subject_condition_means(iS, k_cond) = nanmean(session_means_for_this_subject_conditions{k_cond});
        end
    end
end

% Calculate overall pooled mean and SEM from ALL SESSIONS for each condition
overall_pooled_mean_RT = nan(1, num_conditions); % Changed
overall_pooled_sem_RT   = nan(1, num_conditions); % Changed
for k_cond = 1:num_conditions
    current_condition_all_sessions = pooled_session_RTs_by_condition{k_cond}; % Changed
    if ~isempty(current_condition_all_sessions)
        valid_pooled_sessions = current_condition_all_sessions(~isnan(current_condition_all_sessions));
        if ~isempty(valid_pooled_sessions)
            overall_pooled_mean_RT(k_cond) = mean(valid_pooled_sessions); % Changed
            if length(valid_pooled_sessions) > 1
                overall_pooled_sem_RT(k_cond) = std(valid_pooled_sessions) / sqrt(length(valid_pooled_sessions)); % Changed
            else; overall_pooled_sem_RT(k_cond) = 0; end 
        end
    end
end

% --- Plotting ---
fig_handle = figure('Position', [100, 100, figure_width, figure_height]);
ax = gca;
hold(ax, 'on');

jitter_range = 0.35 / x_within_pair_spacing; 
subject_legend_handles = gobjects(num_subjects_to_process, 1);
subject_has_legend_entry = false(num_subjects_to_process, 1);
plotted_anything_for_legend = false;

max_data_y_val = -Inf; min_data_y_val = Inf; 
% max_overall_y_plus_text = -Inf; % For text above bars (if uncommented)

% Plot individual subject dots (using subject_condition_means)
for k_cond = 1:num_conditions
    current_x_base = x_positions(k_cond);
    for iS = 1:num_subjects_to_process
        if ~isnan(subject_condition_means(iS, k_cond))
            subj_dot_color = custom_colors_rgb{mod(iS-1, length(custom_colors_rgb)) + 1};
            s = rng; 
            rng(iS*10 + k_cond, 'twister'); 
            x_jittered = current_x_base + (rand - 0.5) * jitter_range;
            rng(s); 
            h_dot = plot(ax, x_jittered, subject_condition_means(iS, k_cond), 'o', ...
                 'MarkerFaceColor', subj_dot_color, 'MarkerEdgeColor', subj_dot_color*0.8, 'MarkerSize', 4);
            if ~subject_has_legend_entry(iS)
                subject_legend_handles(iS) = h_dot; subject_has_legend_entry(iS) = true;
                plotted_anything_for_legend = true;
            end
            max_data_y_val = max(max_data_y_val, subject_condition_means(iS, k_cond));
            min_data_y_val = min(min_data_y_val, subject_condition_means(iS, k_cond));
        end
    end
end

% % Plot connecting lines for subjects within paired groups (OPTIONAL - CURRENTLY COMMENTED)
% paired_indices_for_lines = {[2,3], [4,5], [6,7]}; 
% for iS = 1:num_subjects_to_process
%     subj_line_color = custom_colors_rgb{mod(iS-1, length(custom_colors_rgb)) + 1};
%     for p_idx = 1:length(paired_indices_for_lines)
%         pair = paired_indices_for_lines{p_idx};
%         idx1_cond = pair(1); 
%         idx2_cond = pair(2);
% 
%         y1_val = subject_condition_means(iS, idx1_cond);
%         y2_val = subject_condition_means(iS, idx2_cond);
% 
%         if ~isnan(y1_val) && ~isnan(y2_val)
%             x1_pos_line = x_positions(idx1_cond); 
%             x2_pos_line = x_positions(idx2_cond);
%             plot(ax, [x1_pos_line, x2_pos_line], [y1_val, y2_val], ...
%                  '-', 'Color', subj_line_color, 'LineWidth', 0.75, 'HandleVisibility', 'off'); % Added HandleVisibility
%         end
%     end
% end

% Plot overall pooled average dots (black), error bars for ALL conditions
for k_cond = 1:num_conditions
    current_x_base = x_positions(k_cond);
    mean_val = overall_pooled_mean_RT(k_cond); % Changed
    sem_val = overall_pooled_sem_RT(k_cond);   % Changed
    
    if ~isnan(mean_val)
        if isnan(sem_val); sem_val = 0; end 
        errorbar(ax, current_x_base, mean_val, sem_val, ...
                 'k', 'LineWidth', 1.5, 'CapSize', 10, 'LineStyle','none');
        plot(ax, current_x_base, mean_val, 'o', ...
             'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k', 'MarkerSize', 4);
        
        max_data_y_val = max(max_data_y_val, mean_val + sem_val); 
        min_data_y_val = min(min_data_y_val, mean_val - sem_val);
        
        % % OPTIONAL: Text for Mean +/- SEM above error bars (adjust positioning for RT)
        % text_str = sprintf('%.2f\n%c%.2f', mean_val, char(177), sem_val); 
        % y_text_pos_offset = 0.05 * (max_data_y_val - min_data_y_val); % Relative offset
        % if y_text_pos_offset == 0; y_text_pos_offset = 0.05; end % Min offset if range is zero
        % y_text_pos = mean_val + sem_val + y_text_pos_offset;
        % text(ax, current_x_base, y_text_pos, text_str, ...
        %     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        %     'FontSize', mean_se_text_font_size, 'Color', 'k');
        % max_overall_y_plus_text = max(max_overall_y_plus_text, y_text_pos + y_text_pos_offset*0.5); 
    end
end

% Fallback if no data plotted
if isinf(max_data_y_val); max_data_y_val = 2.0; end % Sensible default max RT (e.g., 2 seconds)
if isinf(min_data_y_val); min_data_y_val = 0.0; end
% if isinf(max_overall_y_plus_text); max_overall_y_plus_text = max_data_y_val; end

y_span_data_for_scaling = max_data_y_val - min_data_y_val;
if y_span_data_for_scaling <= 1e-6; y_span_data_for_scaling = max(1.0, max_data_y_val); end % Use max_data_y_val or 1.0 if span is tiny

% Adjust Y positioning for stat bars based on RT data range
stat_bar_y_level_start_offset = 0.10 * y_span_data_for_scaling; 
stat_bar_y_level_min_abs_offset = 0.1 * max(1.0, max_data_y_val); % Min offset based on typical RT or 1s

% Current top of data (consider text if it were enabled)
% top_of_data_elements = max(max_data_y_val, max_overall_y_plus_text);
top_of_data_elements = max_data_y_val; % Simplified as text is commented

stat_bar_y_level = top_of_data_elements + max(stat_bar_y_level_start_offset, stat_bar_y_level_min_abs_offset);

star_text_y_offset_from_bar = 0.02 * y_span_data_for_scaling; 
if star_text_y_offset_from_bar < 0.02; star_text_y_offset_from_bar = 0.02; end % Min absolute offset for star visibility
cap_height_on_bar = 0.03 * y_span_data_for_scaling;
if cap_height_on_bar < 0.01 * max(1.0, max_data_y_val) ; cap_height_on_bar = 0.01 * max(1.0, max_data_y_val); end
if cap_height_on_bar == 0; cap_height_on_bar = 0.015; end


% --- Statistical Tests (ttest2 on pooled session data) and Significance Bars ---
stat_pairs_indices = {[2,3], [4,5], [6,7]}; 
y_max_for_all_stats_text = stat_bar_y_level; 

fprintf('\n--- Pairwise Statistical Test Results (Independent Two-Sample t-test on All Sessions for RT) ---\n');
for p_idx = 1:length(stat_pairs_indices)
    pair = stat_pairs_indices{p_idx};
    idx1 = pair(1); idx2 = pair(2);
    
    data1_all_sessions = pooled_session_RTs_by_condition{idx1}; % Changed
    data2_all_sessions = pooled_session_RTs_by_condition{idx2}; % Changed
    
    data1_valid_sessions = data1_all_sessions(~isnan(data1_all_sessions));
    data2_valid_sessions = data2_all_sessions(~isnan(data2_all_sessions));
    
    fprintf('Comparison: %s vs %s (Reaction Time)\n', condition_labels{idx1}, condition_labels{idx2});
    
    if length(data1_valid_sessions) >= 2 && length(data2_valid_sessions) >= 2
        [h_ttest, p_value, ci_ttest, stats_ttest] = ttest2(data1_valid_sessions, data2_valid_sessions); 
        
        fprintf('  N(%s) = %d sessions, N(%s) = %d sessions\n', condition_labels{idx1}, length(data1_valid_sessions), condition_labels{idx2}, length(data2_valid_sessions));
        fprintf('  t(%d) = %.3f, p = %.4f\n', stats_ttest.df, stats_ttest.tstat, p_value);
        
        if p_value < 0.001; stars = '***';
        elseif p_value < 0.01; stars = '**';
        elseif p_value < 0.05; stars = '*';
        else; stars = 'n.s.'; 
        end
        fprintf('  Significance on plot: %s\n', stars);
    else
        stars = 'n.d.'; 
        fprintf('  Not enough valid session data for test. N(%s)=%d, N(%s)=%d.\n', condition_labels{idx1}, length(data1_valid_sessions), condition_labels{idx2}, length(data2_valid_sessions));
    end
    fprintf('-----------------------------------------\n');
        
    x1_pos_bar = x_positions(idx1); 
    x2_pos_bar = x_positions(idx2);
    current_stat_bar_y = stat_bar_y_level + (p_idx-1) * (cap_height_on_bar + star_text_y_offset_from_bar + 0.03 * y_span_data_for_scaling); % Stagger bars slightly

    plot(ax, [x1_pos_bar, x2_pos_bar], [current_stat_bar_y, current_stat_bar_y], '-k', 'LineWidth', 1);
    plot(ax, [x1_pos_bar, x1_pos_bar], [current_stat_bar_y - cap_height_on_bar, current_stat_bar_y], '-k', 'LineWidth', 1);
    plot(ax, [x2_pos_bar, x2_pos_bar], [current_stat_bar_y - cap_height_on_bar, current_stat_bar_y], '-k', 'LineWidth', 1);
    
    y_pos_stars = current_stat_bar_y + star_text_y_offset_from_bar;
    text(ax, mean([x1_pos_bar,x2_pos_bar]), y_pos_stars, stars, ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', stat_font_size, 'FontWeight', 'bold', 'Color', 'k');
    
    y_max_for_all_stats_text = max(y_max_for_all_stats_text, y_pos_stars + 0.03 * y_span_data_for_scaling); 
end
fprintf('\n'); 
hold(ax, 'off');

% --- Aesthetics ---
set(ax, 'XTick', x_positions);
set(ax, 'XTickLabel', condition_labels);
xtickangle(ax, 30);
xlim(ax, [min(x_positions) - x_group_spacing*0.6, max(x_positions) + x_group_spacing*0.6]);
xlabel(ax, 'WM Condition', 'FontSize', plot_font_size);
ylabel(ax, 'Average Reaction Time (s)', 'FontSize', plot_font_size); % Changed
title(ax, 'WM: Reaction Time by Condition (Pooled Session Means & Stats)', 'FontSize', plot_font_size + 1, 'FontWeight','bold'); % Changed
ax.FontSize = plot_font_size;
grid(ax, 'off');

% Dynamic Y-Limits for RT
padding_factor = 0.05; % 5% padding
y_data_actual_span = max_data_y_val - min_data_y_val;
if y_data_actual_span <= 1e-6; y_data_actual_span = max(0.5, max_data_y_val); end % Use a sensible span if data is flat

final_ylim_min = max(0, min_data_y_val - padding_factor * y_data_actual_span); 
% final_ylim_max_from_data_and_text = max(max_data_y_val + padding_factor * y_data_actual_span, max_overall_y_plus_text); % If text enabled
final_ylim_max_from_data = max_data_y_val + padding_factor * y_data_actual_span;

final_ylim_max = max(final_ylim_max_from_data, y_max_for_all_stats_text); 
if final_ylim_max <= final_ylim_min; final_ylim_max = final_ylim_min + max(0.05*final_ylim_min); end % Ensure some range, at least 0.2s or 10% of min
ylim(ax, [final_ylim_min, final_ylim_max]);

% --- Legend ---
lgd = matlab.graphics.illustration.Legend.empty();
if plotted_anything_for_legend
    valid_legend_handles = subject_legend_handles(subject_has_legend_entry);
    valid_legend_labels = subject_legend_labels(subject_has_legend_entry);
    if ~isempty(valid_legend_handles)
        lgd = legend(ax, valid_legend_handles, valid_legend_labels, ...
               'Location', 'eastoutside', 'FontSize', plot_font_size - 2);
        lgd.Box = 'off';
        % Optional: Adjust figure width slightly if legend makes it too cramped (advanced)
        % drawnow;
        % lgd_pos = get(lgd, 'OuterPosition'); % Normalized units
        % ax_pos = get(ax, 'Position'); % Normalized units
        % if ax_pos(1) + ax_pos(3) > lgd_pos(1) - 0.02 % If axis overlaps legend
        %     set(ax, 'Position', [ax_pos(1) ax_pos(2) lgd_pos(1)-ax_pos(1)-0.03 ax_pos(4)]);
        % end
    end
end

% --- Saving Outputs ---
if exist('fig_handle', 'var') && isvalid(fig_handle)
    figure(fig_handle);
    base_filename = 'wm_RT_by_condition_grouped';
    date_str = datestr(now, 'yyyymmdd_HHMMSS');
    full_base_filename = [base_filename '_' date_str];
    save_folder = 'WM_Figures_RT_GroupedConditions';

    if ~exist(save_folder, 'dir')
       try
           mkdir(save_folder);
       catch ME_mkdir
           fprintf('Error creating save dir ''%s'': %s. Saving to current dir.\n', save_folder, ME_mkdir.message);
           save_folder = '.';
       end
    end

    filepath_base = fullfile(save_folder, full_base_filename);

    % Save Figure (PNG and EPS)
    try
        png_filename = [filepath_base '.png'];
        saveas(fig_handle, png_filename);
        fprintf('Figure saved as: %s\n', png_filename);
		hold(ax, 'off');
        eps_filename = [filepath_base '.eps'];
        print(fig_handle, eps_filename, '-depsc');
        fprintf('Figure saved as: %s\n', eps_filename);
    catch ME_save
        fprintf('Error saving figure: %s\n', ME_save.message);
    end

    % Save Data to CSVs
    try
        % --- Summary Data CSV ---
        condition_headers = matlab.lang.makeValidName(condition_labels);

        row_labels = {'Overall_Pooled_Mean_RT'; 'Overall_Pooled_SEM_RT'};
        data_matrix = [overall_pooled_mean_RT; overall_pooled_sem_RT];

        plotted_subj_indices = find(subject_has_legend_entry);
        for i = 1:length(plotted_subj_indices)
            subj_idx = plotted_subj_indices(i);
            row_labels{end+1,1} = subject_legend_labels{subj_idx};
            data_matrix = [data_matrix; subject_condition_means(subj_idx, :)];
        end

        summary_table = array2table(data_matrix, 'VariableNames', condition_headers);
        summary_table = addvars(summary_table, row_labels, 'Before', 1, 'NewVariableNames', 'Group');

        csv_summary_filename = [filepath_base '_rt_summary_data.csv'];
        writetable(summary_table, csv_summary_filename);
        fprintf('Summary RT data saved to: %s\n', csv_summary_filename);

        % --- Statistics CSV ---
        stats_comparisons_list = {};
        stats_n1_list = [];
        stats_n2_list = [];
        stats_t_list = [];
        stats_df_list = [];
        stats_p_list = [];

        stat_pairs_to_process = {[2,3], [4,5], [6,7]};
        for p_idx = 1:length(stat_pairs_to_process)
            pair = stat_pairs_to_process{p_idx};
            idx1 = pair(1);
            idx2 = pair(2);

            data1_valid = pooled_session_RTs_by_condition{idx1}(~isnan(pooled_session_RTs_by_condition{idx1}));
            data2_valid = pooled_session_RTs_by_condition{idx2}(~isnan(pooled_session_RTs_by_condition{idx2}));

            stats_comparisons_list{end+1,1} = sprintf('%s vs %s', condition_labels{idx1}, condition_labels{idx2});
            stats_n1_list(end+1,1) = length(data1_valid);
            stats_n2_list(end+1,1) = length(data2_valid);

            if length(data1_valid) >= 2 && length(data2_valid) >= 2
                [~, p_val, ~, stats] = ttest2(data1_valid, data2_valid);
                stats_t_list(end+1,1) = stats.tstat;
                stats_df_list(end+1,1) = stats.df;
                stats_p_list(end+1,1) = p_val;
            else
                stats_t_list(end+1,1) = NaN;
                stats_df_list(end+1,1) = NaN;
                stats_p_list(end+1,1) = NaN;
            end
        end

        if ~isempty(stats_comparisons_list)
             stats_table = table(stats_comparisons_list, stats_n1_list, stats_n2_list, stats_t_list, stats_df_list, stats_p_list, ...
                'VariableNames', {'Comparison', 'N1_Sessions', 'N2_Sessions', 'T_Statistic', 'DF', 'PValue'});

            csv_stats_filename = [filepath_base '_rt_statistics.csv'];
            writetable(stats_table, csv_stats_filename);
            fprintf('Statistical results saved to: %s\n', csv_stats_filename);
        end

    catch ME_csv
        fprintf('Error saving CSV data: %s\n', ME_csv.message);
    end

else
    fprintf('Figure handle not valid/created. Figure not saved.\n');
end

end % End of main function

% Helper function to safely extract session mean
function session_mean = extract_session_mean_safely(session_data, field_name)
    session_mean = NaN;
    if isfield(session_data, field_name) && ~isempty(session_data.(field_name))
        current_field_data = session_data.(field_name);
        % Expecting field_name to provide data where column 1 is the mean
        if ismatrix(current_field_data) && size(current_field_data,1) >= 1 
            if size(current_field_data, 2) >=1 % Ensure at least one column
                relevant_data = current_field_data(:,1); % Take the first column
            else 
                relevant_data = current_field_data(:); % Should not be hit if checks pass
            end
            if ~isempty(relevant_data) && ~all(isnan(relevant_data))
                 session_mean = nanmean(relevant_data); % Mean of (potentially multiple) means if field had >1 row
            end
        elseif isscalar(current_field_data) && ~isnan(current_field_data) % Handle scalar case
            session_mean = current_field_data;
        end
    end
end

function plot_wm_RT_vs_delay(metrics_mt)
% plot_wm_RT_vs_delay Plots WM Reaction Time vs. delay time with stats.
%   - Significance display: Stars for p<0.05, "n.s." for p>=0.05. No raw p-values.
%   - Legend fixed and styled.
%   - X-ticks are unique delay values. X-axis title: "Delay (s)".
%   - Y-limits are dynamic based on data (min 0). Y-axis title: "Average Reaction Time (s)".
%   - Subject legend labels: "Subject X (initial)".
%   - Overall: black line with SEM error bars (from ALL sessions pooled).
%   - Figure size 350x220, font 14, legend right (no box), no grid.
%   - Saves as PNG.

% --- Hardcoded Subject Information and Colors ---
subject_id_list_original = {
    'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan'
};
custom_colors = {
    [245, 124, 110]/255; [242, 181, 110]/255; [251, 231, 158]/255;
    [132, 195, 183]/255; [136, 215, 218]/255; [113, 184, 237]/255;
    [184, 174, 234]/255; [242, 168, 218]/255
};
overall_color = [0 0 0];

% --- Font Size & Plotting Constants ---
base_font_size = 14;
stat_font_size = base_font_size - 2; % Slightly smaller for stars/ns

% --- Input Validation ---
if nargin < 1; error('Usage: %s(metrics_mt)', mfilename); end
if ~iscell(metrics_mt); error('metrics_mt must be a cell array.'); end

% --- Figure Setup ---
figure_width = 600;
figure_height = 400; % Adjusted from 200 to 220 as in original
fig = figure('Position', [100, 100, figure_width, figure_height]);
ax = gca;
hold(ax, 'on');
num_subjects_in_data = length(metrics_mt);

% --- Step 1: Identify all unique delay times ---
all_delays_encountered = [];
for iS = 1:num_subjects_in_data
    if iS > length(metrics_mt) || isempty(metrics_mt{iS}); continue; end
    for iD = 1:length(metrics_mt{iS})
        if isstruct(metrics_mt{iS}(iD)) && isfield(metrics_mt{iS}(iD), 'conditions_nDistr_delays_wm') && ...
           ~isempty(metrics_mt{iS}(iD).conditions_nDistr_delays_wm) && size(metrics_mt{iS}(iD).conditions_nDistr_delays_wm,2) >= 2
            all_delays_encountered = [all_delays_encountered; metrics_mt{iS}(iD).conditions_nDistr_delays_wm(:, 2)];
        end
    end
end
if isempty(all_delays_encountered)
    warning('No delay time data found. Cannot generate plot.');
    text(ax, 0.5, 0.5, 'No WM delay data found.', 'HorizontalAlignment', 'center', 'Units', 'normalized', 'FontSize', base_font_size, 'Color', 'red');
    hold(ax,'off'); if isvalid(fig); close(fig); end; return;
end
master_unique_sorted_delays = sort(unique(all_delays_encountered));
num_master_delays = length(master_unique_sorted_delays);

% --- Initialize data storage ---
pooled_session_RTs_by_delay = cell(1, num_master_delays); % Changed from accuracies
subject_means_on_master_delays = NaN(num_subjects_in_data, num_master_delays);
subject_sems_on_master_delays = NaN(num_subjects_in_data, num_master_delays);
subject_has_data_mask = false(1, num_subjects_in_data);

% --- Process data per session, aggregate for overall and subject ---
for iS = 1:num_subjects_in_data
    if iS > length(metrics_mt) || isempty(metrics_mt{iS}); continue; end
    current_subject_session_RTs_by_delay = cell(1, num_master_delays); % Changed
    sessions_processed_for_subject = 0;
    for iD = 1:length(metrics_mt{iS})
        if ~isstruct(metrics_mt{iS}(iD)); continue; end
        session_data = metrics_mt{iS}(iD);
        % Ensure field name is RT_MeanSE_wm
        if isfield(session_data, 'conditions_nDistr_delays_wm') && isfield(session_data, 'RT_MeanSE_wm') && ... % Changed
           ~isempty(session_data.conditions_nDistr_delays_wm) && ~isempty(session_data.RT_MeanSE_wm) && ... % Changed
           size(session_data.conditions_nDistr_delays_wm, 1) == size(session_data.RT_MeanSE_wm, 1) && ... % Changed
           size(session_data.conditions_nDistr_delays_wm, 2) >= 2 && size(session_data.RT_MeanSE_wm, 2) >= 1 % Changed
            
            sessions_processed_for_subject = sessions_processed_for_subject + 1;
            session_delays = session_data.conditions_nDistr_delays_wm(:, 2);
            session_RTs_col1 = session_data.RT_MeanSE_wm(:, 1); % Changed from accuracyMeanSE_wm

            for k_delay = 1:num_master_delays
                target_delay = master_unique_sorted_delays(k_delay);
                idx_matching_delay_in_session = (abs(session_delays - target_delay) < 1e-6);
                if any(idx_matching_delay_in_session)
                    RTs_for_this_delay_this_session = session_RTs_col1(idx_matching_delay_in_session); % Changed
                    mean_RT_for_delay_this_session = mean(RTs_for_this_delay_this_session, 'omitnan'); % Changed
                    if ~isnan(mean_RT_for_delay_this_session)
                        current_subject_session_RTs_by_delay{k_delay} = [current_subject_session_RTs_by_delay{k_delay}, mean_RT_for_delay_this_session]; % Changed
                        pooled_session_RTs_by_delay{k_delay} = [pooled_session_RTs_by_delay{k_delay}, mean_RT_for_delay_this_session]; % Changed
                    end
                end
            end
        end
    end 
    if sessions_processed_for_subject > 0; subject_has_data_mask(iS) = true; end

    for k_delay = 1:num_master_delays
        session_RTs_for_subj_at_delay = current_subject_session_RTs_by_delay{k_delay}; % Changed
        if ~isempty(session_RTs_for_subj_at_delay)
            subject_means_on_master_delays(iS, k_delay) = mean(session_RTs_for_subj_at_delay, 'omitnan');
            num_valid_sessions = sum(~isnan(session_RTs_for_subj_at_delay));
            if num_valid_sessions > 1; subject_sems_on_master_delays(iS, k_delay) = std(session_RTs_for_subj_at_delay, 'omitnan') / sqrt(num_valid_sessions);
            else; subject_sems_on_master_delays(iS, k_delay) = 0; end % SEM is 0 if only one session contributes
        end
    end
end 

% --- Calculate Overall Plot Data & Y-limit data collection ---
overall_RT_means_on_master_delays = NaN(1, num_master_delays); % Changed
overall_RT_sems_on_master_delays = NaN(1, num_master_delays);  % Changed
all_y_minus_sem_collected = []; all_y_plus_sem_collected = [];  
max_data_y_extent = -Inf; % Track max y of data lines (mean+sem)

for k_delay = 1:num_master_delays
    all_sess_RTs_at_delay = pooled_session_RTs_by_delay{k_delay}; % Changed
    if ~isempty(all_sess_RTs_at_delay)
        current_mean = mean(all_sess_RTs_at_delay, 'omitnan');
        overall_RT_means_on_master_delays(k_delay) = current_mean; % Changed
        num_valid_pooled_sessions = sum(~isnan(all_sess_RTs_at_delay));
        current_sem = 0;
        if num_valid_pooled_sessions > 1; current_sem = std(all_sess_RTs_at_delay, 'omitnan') / sqrt(num_valid_pooled_sessions); end
        overall_RT_sems_on_master_delays(k_delay) = current_sem; % Changed
        if ~isnan(current_mean)
            all_y_minus_sem_collected = [all_y_minus_sem_collected, current_mean - current_sem];
            all_y_plus_sem_collected = [all_y_plus_sem_collected, current_mean + current_sem];
            max_data_y_extent = max(max_data_y_extent, current_mean + current_sem);
        end
    end
end

% --- Plotting Data Lines ---
legend_handles = [];
plotted_anything_for_legend = false;
valid_subject_indices_for_plot = find(subject_has_data_mask);

for k_subj_plot = 1:length(valid_subject_indices_for_plot)
    iS = valid_subject_indices_for_plot(k_subj_plot);
    subj_label_for_legend = sprintf('Subj %d', iS); 
    if iS <= length(subject_id_list_original) && ~isempty(subject_id_list_original{iS})
        original_id = subject_id_list_original{iS}; initial = upper(original_id(1));
        subj_label_for_legend = sprintf('Subject %c', initial); 
    end
    if iS <= length(custom_colors); subj_color = custom_colors{iS}; else; subj_color = rand(1,3); end
    
    current_means = subject_means_on_master_delays(iS, :); current_sems = subject_sems_on_master_delays(iS, :);
    if any(~isnan(current_means))
        h_subj = errorbar(ax, master_unique_sorted_delays, current_means, current_sems, ...
            '-o', 'Color', subj_color, 'MarkerFaceColor', subj_color, 'MarkerEdgeColor', subj_color, ...
            'LineWidth', 1.5, 'CapSize', 5, 'MarkerSize', 5, 'DisplayName', subj_label_for_legend);
        legend_handles(end+1) = h_subj; plotted_anything_for_legend = true;
        
        valid_idx_subj = ~isnan(current_means);
        means_to_collect = current_means(valid_idx_subj);
        sems_to_collect = current_sems(valid_idx_subj); sems_to_collect(isnan(sems_to_collect)) = 0;
        all_y_minus_sem_collected = [all_y_minus_sem_collected, means_to_collect - sems_to_collect];
        all_y_plus_sem_collected = [all_y_plus_sem_collected, means_to_collect + sems_to_collect];
        max_data_y_extent = max(max_data_y_extent, max(means_to_collect + sems_to_collect,[],'omitnan'));
    end
end

if any(~isnan(overall_RT_means_on_master_delays)) % Changed
    h_overall = errorbar(ax, master_unique_sorted_delays, overall_RT_means_on_master_delays, overall_RT_sems_on_master_delays, ... % Changed
        '-o', 'Color', overall_color, 'MarkerFaceColor', overall_color, 'MarkerEdgeColor', overall_color, ...
        'LineWidth', 2, 'CapSize', 5, 'MarkerSize', 5, 'DisplayName', 'Overall');
    legend_handles(end+1) = h_overall;
    plotted_anything_for_legend = true;
end

% --- Basic Plot Styling (Labels, Title, Ticks) ---
set(ax, 'FontSize', base_font_size);
xlabel(ax, 'Delay (s)', 'FontSize', base_font_size); 
ylabel(ax, 'Average Reaction Time (s)', 'FontSize', base_font_size); % Changed
title(ax, 'WM: Reaction Time vs. Delay', 'FontSize', base_font_size, 'FontWeight', 'bold'); % Changed
grid(ax, 'off'); 

if ~isempty(master_unique_sorted_delays) && plotted_anything_for_legend
    set(ax, 'XTick', master_unique_sorted_delays);
    if length(master_unique_sorted_delays) > 6; xtickangle(ax, 45); end
else
    set(ax, 'XTick', []); xlim(ax, [0 1]); 
end

% --- Create Legend (before stats, so its position isn't affected by stat bars initially) ---
lgd = matlab.graphics.illustration.Legend.empty(); % Initialize to empty
if plotted_anything_for_legend
    lgd = legend(ax, legend_handles, 'Location', 'eastoutside', 'FontSize', base_font_size);
    lgd.Box = 'off';
end

% --- Perform T-tests and Add Significance Bars ---
y_max_for_stats_plotting = -Inf; % Will track the highest point reached by stats annotations
if num_master_delays >= 2 && any(~isnan(overall_RT_means_on_master_delays)) % Changed
    data_longest_delay_pooled = pooled_session_RTs_by_delay{num_master_delays}; % Changed
    data_longest_delay_pooled_clean = data_longest_delay_pooled(~isnan(data_longest_delay_pooled));
    
    if ~isempty(data_longest_delay_pooled_clean) && length(data_longest_delay_pooled_clean) >=2
        if isinf(max_data_y_extent)
            % Fallback if no data plotted - for RT, 1.0 is not a good default.
            % Let's try to make it somewhat data-driven or a common RT max like 2s or 3s.
            % For now, let's use a placeholder, will be refined by final_ylim_max later.
             max_data_y_extent = nanmax(overall_RT_means_on_master_delays); % Use max mean if available
             if isnan(max_data_y_extent) || isinf(max_data_y_extent)
                 max_data_y_extent = 2.0; % Arbitrary fallback if means are also empty
             end
        end
        y_current_level_for_bar = max_data_y_extent + 0.05 * max_data_y_extent; % Start 5% of max_data_y_extent above max data
        
        min_rt_val_overall = min(all_y_minus_sem_collected, [], 'omitnan');
        max_rt_val_overall = max(all_y_plus_sem_collected, [], 'omitnan');
        if isempty(min_rt_val_overall) || isnan(min_rt_val_overall) || isinf(min_rt_val_overall); min_rt_val_overall = 0; end;
        if isempty(max_rt_val_overall) || isnan(max_rt_val_overall) || isinf(max_rt_val_overall); max_rt_val_overall = y_current_level_for_bar; end;
        
        plot_y_range = max(0.2, max_rt_val_overall - max(0, min_rt_val_overall)); % Min range of 0.2s for calc
        stat_bar_vertical_step = plot_y_range * 0.08; 
        if stat_bar_vertical_step < 0.02 * max(1,max_rt_val_overall); stat_bar_vertical_step = 0.02 * max(1,max_rt_val_overall); end % Min step relative to max RT or 1s
        if stat_bar_vertical_step == 0; stat_bar_vertical_step = 0.05; end % Absolute minimum if range is tiny

        stat_cap_height = stat_bar_vertical_step * 0.1; % Adjusted cap height scaling
        text_v_offset_above_bar = stat_bar_vertical_step * 0.1; % Adjusted text offset

        for k_stat = 1:(num_master_delays - 1)
            data_current_delay_pooled = pooled_session_RTs_by_delay{k_stat}; % Changed
            data_current_delay_pooled_clean = data_current_delay_pooled(~isnan(data_current_delay_pooled));
            text_to_display = 'n.d.'; 
            if length(data_current_delay_pooled_clean) >= 2 && length(data_longest_delay_pooled_clean) >=2
                [~, p_val] = ttest2(data_current_delay_pooled_clean, data_longest_delay_pooled_clean);
                
                if p_val < 0.001; text_to_display = '***';
                elseif p_val < 0.01; text_to_display = '**';
                elseif p_val < 0.05; text_to_display = '*';
                else; text_to_display = 'n.s.'; 
                end
            end
            x1 = master_unique_sorted_delays(k_stat);
            x2 = master_unique_sorted_delays(num_master_delays);
            
            plot(ax, [x1, x2], [y_current_level_for_bar, y_current_level_for_bar], '-k', 'LineWidth', 0.75, HandleVisibility='off');
            plot(ax, [x1, x1], [y_current_level_for_bar - stat_cap_height, y_current_level_for_bar], '-k', 'LineWidth', 0.75, HandleVisibility='off');
            plot(ax, [x2, x2], [y_current_level_for_bar - stat_cap_height, y_current_level_for_bar], '-k', 'LineWidth', 0.75, HandleVisibility='off');
            
            text(ax, mean([x1,x2]), y_current_level_for_bar + text_v_offset_above_bar, text_to_display, ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', stat_font_size, 'Color', 'k');
            y_current_level_for_bar = y_current_level_for_bar + stat_bar_vertical_step; 
        end
        y_max_for_stats_plotting = y_current_level_for_bar - stat_bar_vertical_step + text_v_offset_above_bar*2; 
    end
end

% --- Final Y Lim based on data AND stats ---
final_ylim_min = 0; 
% Default upper limit for RT; can be much larger than 1. Let's make it data-driven.
default_max_rt = 2.0; % Default if no data points to derive from
if ~isempty(all_y_plus_sem_collected)
    default_max_rt = max(all_y_plus_sem_collected, [], 'omitnan');
    if isnan(default_max_rt) || isinf(default_max_rt) || isempty(default_max_rt)
        default_max_rt = 2.0;
    end
end
final_ylim_max = default_max_rt * 1.1; % Start with 10% above max data point

if ~isempty(all_y_minus_sem_collected)
    min_data_y_val = min(all_y_minus_sem_collected,[],'omitnan'); 
    max_data_y_val = max(all_y_plus_sem_collected,[],'omitnan');
    
    min_data_y_val = max(0, min_data_y_val); % RT cannot be negative
    % max_data_y_val is not capped at 1 for RT
    
    plot_data_range = max_data_y_val - min_data_y_val;
    padding = 0.05 * default_max_rt; % Padding relative to a sensible RT scale
    if plot_data_range > 1e-6; padding = plot_data_range * 0.10; end
    if padding == 0; padding = 0.1; end % Minimum padding
    
    final_ylim_min = max(0, min_data_y_val - padding);
    tentative_ylim_max = max_data_y_val + padding;
    
    if ~isinf(y_max_for_stats_plotting) % If stats were plotted
        final_ylim_max = max(tentative_ylim_max, y_max_for_stats_plotting + padding*0.5);
    else
        final_ylim_max = tentative_ylim_max;
    end
    
    if (final_ylim_max - final_ylim_min) < (0.2 * default_max_rt) && default_max_rt > 0 % Ensure a minimum visible range, relative to typical RT
        mid_point = (final_ylim_max + final_ylim_min) / 2;
        mid_point = max(0.1 * default_max_rt, mid_point); % Ensure midpoint is reasonable
        final_ylim_min = mid_point - (0.1 * default_max_rt);
        final_ylim_max = mid_point + (0.1 * default_max_rt);
    end
    final_ylim_min = max(0, final_ylim_min); 
    % final_ylim_max can be > 1.1, no specific cap needed other than being reasonable
    if final_ylim_max <= final_ylim_min; final_ylim_max = final_ylim_min + 0.2; end % Ensure max > min
else % Fallback if all_y_minus_sem_collected is empty
    final_ylim_min = 0;
    final_ylim_max = default_max_rt; % Use the default max determined earlier
     if ~isinf(y_max_for_stats_plotting) % If stats were plotted
        final_ylim_max = max(final_ylim_max, y_max_for_stats_plotting + 0.1);
    end
end
ylim(ax, [final_ylim_min, final_ylim_max]);

% --- Adjust XLim after all plotting ---
if ~isempty(master_unique_sorted_delays) && plotted_anything_for_legend
    xlim_padding = 0.1 * (max(master_unique_sorted_delays) - min(master_unique_sorted_delays));
    if xlim_padding == 0 || isnan(xlim_padding); xlim_padding = 0.5; end
    current_xlim = [min(master_unique_sorted_delays) - xlim_padding, max(master_unique_sorted_delays) + xlim_padding];
    if current_xlim(1) >= current_xlim(2); current_xlim = [master_unique_sorted_delays(1)-0.5, master_unique_sorted_delays(1)+0.5]; end
    xlim(ax, current_xlim);
end

% --- Adjust layout for legend (if it was created) ---
drawnow; 
if plotted_anything_for_legend && ~isempty(lgd) && isvalid(lgd) && strcmp(lgd.Location, 'eastoutside')
    try 
        original_ax_units = get(ax, 'Units'); original_lgd_units = get(lgd, 'Units');
        set(ax, 'Units', 'normalized'); set(lgd, 'Units', 'normalized');
        drawnow; pause(0.1); % Allow redraw
        ax_pos_norm = get(ax, 'Position'); lgd_outer_pos_norm = get(lgd, 'OuterPosition');
        
        % Ensure legend doesn't overlap axis if it's outside
        max_allowable_ax_width = lgd_outer_pos_norm(1) - ax_pos_norm(1) - 0.05; % 0.05 is a small buffer
        
        if ax_pos_norm(3) > max_allowable_ax_width && max_allowable_ax_width > 0.05 % If axis is too wide and there's space to shrink
            set(ax, 'Position', [ax_pos_norm(1), ax_pos_norm(2), max_allowable_ax_width, ax_pos_norm(4)]);
        end
        set(ax, 'Units', original_ax_units); set(lgd, 'Units', original_lgd_units);
    catch ME_layout
        fprintf('Warning: Could not auto-adjust layout for legend: %s\n', ME_layout.message);
        % Restore units if they were changed and an error occurred
        % if exist('original_ax_units','var') && isvalid(ax); set(ax, 'Units', original_ax_units); end
        % if exist('original_lgd_units','var') && isvalid(lgd); set(lgd, 'Units', original_lgd_units); end
    end
end

% --- Saving Outputs ---
if exist('fig', 'var') && isvalid(fig)
    figure(fig);
    base_filename = 'wm_RT_vs_delay_stats';
    date_str = datestr(now, 'yyyymmdd_HHMMSS');
    full_base_filename = [base_filename '_' date_str];
    save_folder = 'WM_Figures_RT_Delay';

    if ~exist(save_folder, 'dir')
        try; mkdir(save_folder);
        catch ME_mkdir
            fprintf('Error creating save dir ''%s'': %s. Saving to current dir.\n', save_folder, ME_mkdir.message);
            save_folder = '.';
        end
    end

    filepath_base = fullfile(save_folder, full_base_filename);

    % Save Figure (PNG and EPS)
    try
        png_filename = [filepath_base '.png'];
        saveas(fig, png_filename);
        fprintf('Figure saved as: %s\n', png_filename);

        eps_filename = [filepath_base '.eps'];
        print(fig, eps_filename, '-depsc');
        fprintf('Figure saved as: %s\n', eps_filename);
    catch ME_save
        fprintf('Error saving figure: %s\n', ME_save.message);
    end

    % Save Data to CSVs
    try
        % --- Summary Data CSV ---
        delay_headers = arrayfun(@(x) sprintf('Delay_%.3f_s', x), master_unique_sorted_delays, 'UniformOutput', false);
        
        row_labels = {'Overall_Mean_RT'; 'Overall_SE_RT'};
        data_matrix = [overall_RT_means_on_master_delays; overall_RT_sems_on_master_delays];
        
        plotted_subj_indices = find(subject_has_data_mask);
        for i = 1:length(plotted_subj_indices)
            subj_idx = plotted_subj_indices(i);
            
            subj_label = sprintf('Subj_%d', subj_idx);
            if subj_idx <= length(subject_id_list_original) && ~isempty(subject_id_list_original{subj_idx})
                subj_label = sprintf('Subject_%s', subject_id_list_original{subj_idx});
            end

            row_labels{end+1,1} = [subj_label '_Mean_RT'];
            row_labels{end+1,1} = [subj_label '_SE_RT'];
            data_matrix = [data_matrix; subject_means_on_master_delays(subj_idx,:); subject_sems_on_master_delays(subj_idx,:)];
        end

        summary_table = array2table(data_matrix, 'VariableNames', delay_headers);
        summary_table = addvars(summary_table, row_labels, 'Before', 1, 'NewVariableNames', 'Series');

        csv_summary_filename = [filepath_base '_rt_summary_data.csv'];
        writetable(summary_table, csv_summary_filename);
        fprintf('Summary RT data saved to: %s\n', csv_summary_filename);

        % --- Statistics CSV ---
        stats_comparisons = {};
        stats_p_values = [];
        stats_t_values = [];
        stats_df = [];

        if num_master_delays >= 2
            data_longest_delay_pooled = pooled_session_RTs_by_delay{num_master_delays};
            data_longest_delay_clean = data_longest_delay_pooled(~isnan(data_longest_delay_pooled));

            for k_stat = 1:(num_master_delays - 1)
                x1 = master_unique_sorted_delays(k_stat);
                x2 = master_unique_sorted_delays(num_master_delays);
                stats_comparisons{end+1,1} = sprintf('Delay %.3f vs Delay %.3f', x1, x2);
                
                data_current_delay_pooled = pooled_session_RTs_by_delay{k_stat};
                data_current_delay_clean = data_current_delay_pooled(~isnan(data_current_delay_pooled));

                if length(data_current_delay_clean) >= 2 && length(data_longest_delay_clean) >= 2
                    [~, p_val, ~, stats] = ttest2(data_current_delay_clean, data_longest_delay_clean);
                    stats_p_values(end+1,1) = p_val;
                    stats_t_values(end+1,1) = stats.tstat;
                    stats_df(end+1,1) = stats.df;
                else
                    stats_p_values(end+1,1) = NaN;
                    stats_t_values(end+1,1) = NaN;
                    stats_df(end+1,1) = NaN;
                end
            end
        end
        
        if ~isempty(stats_comparisons)
            stats_table = table(stats_comparisons, stats_t_values, stats_df, stats_p_values, 'VariableNames', {'Comparison', 'T_Statistic', 'DF', 'PValue'});
            csv_stats_filename = [filepath_base '_rt_statistics.csv'];
            writetable(stats_table, csv_stats_filename);
            fprintf('Statistical results saved to: %s\n', csv_stats_filename);
        end

    catch ME_csv
        fprintf('Error saving CSV data: %s\n', ME_csv.message);
    end

else
    fprintf('Figure handle not valid/created. Figure not saved.\n');
end
hold(ax, 'off');
end


function plot_wm_accuracy_by_condition_grouped(metrics_mt)
% plot_wm_accuracy_by_condition_grouped Plots WM accuracy for grouped conditions.
%   - Main condition points: Mean of ALL SESSIONS pooled for that condition, black dot (size 6) with error bar.
%   - Pairwise stats: Independent t-test (ttest2) on pooled session data between conditions in a pair.
%   - P-values, t-stats, Ns, and df displayed in command window. Stars displayed on figure.
%   - Subject data: Jittered colored dots (size 6), representing subject's mean accuracy for that condition.
%   - Connecting lines: For each subject, lines connect their means between paired conditions, using subject's color.
%   - Figure 800x350, Font 14, Legend right (no box), No grid.
%   - X-axis title "WM Condition". X-ticks are condition labels.
%   - Saves as PNG.

% --- Configuration ---
plot_font_size = 14;
stat_font_size = plot_font_size - 3; % For stars
mean_se_text_font_size = plot_font_size - 5; % For Mean +/- SE text
figure_width = 500; 
figure_height = 300; % Adjusted based on previous versions
condition_labels = {'Overall', 'Low TDS', 'High TDS', '2 Distr', '3 Distr', 'No PSD', 'PSD Pres'};
num_conditions = length(condition_labels);

x_group_spacing = 1.2;  
x_within_pair_spacing = 0.8;
x_positions = zeros(1, num_conditions);
current_x = 1;
x_positions(1) = current_x; % Overall
current_x = current_x + x_group_spacing;
x_positions(2) = current_x; x_positions(3) = current_x + x_within_pair_spacing; % TDS
current_x = current_x + x_within_pair_spacing + x_group_spacing;
x_positions(4) = current_x; x_positions(5) = current_x + x_within_pair_spacing; % Distr
current_x = current_x + x_within_pair_spacing + x_group_spacing;
x_positions(6) = current_x; x_positions(7) = current_x + x_within_pair_spacing; % PSD

subject_id_list_hardcoded = {
    'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan' 
}; % Add more if needed
custom_colors_rgb = {
    [245, 124, 110]/255; [242, 181, 110]/255; [251, 231, 158]/255;
    [132, 195, 183]/255; [136, 215, 218]/255; [113, 184, 237]/255;
    [184, 174, 234]/255; [242, 168, 218]/255 
}; % Add more if needed

num_subjects_to_process = length(metrics_mt);
if num_subjects_to_process == 0; disp('Input metrics_mt is empty.'); return; end

% --- Data Aggregation ---
% subject_condition_means: stores one mean value per subject per condition (average of THEIR sessions for that condition)
subject_condition_means = nan(num_subjects_to_process, num_conditions);
subject_legend_labels = cell(1, num_subjects_to_process);
% pooled_session_accuracies_by_condition: cell array, each cell contains ALL session accuracies from ALL subjects for one condition
pooled_session_accuracies_by_condition = cell(1, num_conditions); 
for k=1:num_conditions; pooled_session_accuracies_by_condition{k} = []; end

for iS = 1:num_subjects_to_process
    if iS <= length(subject_id_list_hardcoded) && ~isempty(subject_id_list_hardcoded{iS})
        original_id = subject_id_list_hardcoded{iS}; initial = upper(original_id(1));
        subject_legend_labels{iS} = sprintf('Subject %c', initial);
    else; subject_legend_labels{iS} = sprintf('Subj %d', iS); end
    
    if iS > length(metrics_mt) || isempty(metrics_mt{iS}); continue; end
    
    session_means_for_this_subject_conditions = cell(1, num_conditions); 
    for k=1:num_conditions; session_means_for_this_subject_conditions{k} = []; end

    for iD = 1:length(metrics_mt{iS}) 
        if ~isstruct(metrics_mt{iS}(iD)); continue; end
        session_data = metrics_mt{iS}(iD);
        
        get_session_mean = @(field_name) extract_session_mean_safely(session_data, field_name);
        field_map = {'accuracyMeanSE_wm', 'accuracyMeanSE_LowTDS_wm', 'accuracyMeanSE_HighTDS_wm', ...
                     'accuracyMeanSE_2dist_wm', 'accuracyMeanSE_3dist_wm', ...
                     'accuracyMeanSE_noPostSampleDist_wm', 'accuracyMeanSE_yesPostSampleDist_wm'};
        
        for k_cond = 1:num_conditions
            session_cond_acc = get_session_mean(field_map{k_cond}); 
            if ~isnan(session_cond_acc)
                session_means_for_this_subject_conditions{k_cond} = [session_means_for_this_subject_conditions{k_cond}, session_cond_acc];
                pooled_session_accuracies_by_condition{k_cond} = [pooled_session_accuracies_by_condition{k_cond}, session_cond_acc]; % Add to grand pool
            end
        end
    end 
    
    for k_cond = 1:num_conditions
        if ~isempty(session_means_for_this_subject_conditions{k_cond})
            subject_condition_means(iS, k_cond) = nanmean(session_means_for_this_subject_conditions{k_cond});
        end
    end
end

% Calculate overall pooled mean and SEM from ALL SESSIONS for each condition
overall_pooled_mean_accuracy = nan(1, num_conditions);
overall_pooled_sem_accuracy   = nan(1, num_conditions);
for k_cond = 1:num_conditions
    current_condition_all_sessions = pooled_session_accuracies_by_condition{k_cond};
    if ~isempty(current_condition_all_sessions)
        valid_pooled_sessions = current_condition_all_sessions(~isnan(current_condition_all_sessions));
        if ~isempty(valid_pooled_sessions)
            overall_pooled_mean_accuracy(k_cond) = mean(valid_pooled_sessions);
            if length(valid_pooled_sessions) > 1
                overall_pooled_sem_accuracy(k_cond) = std(valid_pooled_sessions) / sqrt(length(valid_pooled_sessions));
            else; overall_pooled_sem_accuracy(k_cond) = 0; end % SEM is 0 if only one session
        end
    end
end

% --- Plotting ---
fig_handle = figure('Position', [100, 100, figure_width, figure_height]);
ax = gca;
hold(ax, 'on');

jitter_range = 0.35 / x_within_pair_spacing; 
subject_legend_handles = gobjects(num_subjects_to_process, 1);
subject_has_legend_entry = false(num_subjects_to_process, 1);
plotted_anything_for_legend = false;

max_data_y_val = -Inf; min_data_y_val = Inf; 
max_overall_y_plus_text = -Inf; 

% Plot individual subject dots (using subject_condition_means)
for k_cond = 1:num_conditions
    current_x_base = x_positions(k_cond);
    for iS = 1:num_subjects_to_process
        if ~isnan(subject_condition_means(iS, k_cond))
            subj_dot_color = custom_colors_rgb{mod(iS-1, length(custom_colors_rgb)) + 1};
            % Jitter based on subject and condition to be somewhat consistent if function is re-run (though rand is still pseudo-random)
            s = rng; % Store current rng state
            rng(iS*10 + k_cond, 'twister'); % Seed for somewhat deterministic jitter per dot
            x_jittered = current_x_base + (rand - 0.5) * jitter_range;
            rng(s); % Restore rng state

            h_dot = plot(ax, x_jittered, subject_condition_means(iS, k_cond), 'o', ...
                 'MarkerFaceColor', subj_dot_color, 'MarkerEdgeColor', subj_dot_color*0.8, 'MarkerSize', 4); %%% CHANGED: MarkerSize 6
            if ~subject_has_legend_entry(iS)
                subject_legend_handles(iS) = h_dot; subject_has_legend_entry(iS) = true;
                plotted_anything_for_legend = true;
            end
            max_data_y_val = max(max_data_y_val, subject_condition_means(iS, k_cond));
            min_data_y_val = min(min_data_y_val, subject_condition_means(iS, k_cond));
        end
    end
end

% % Plot connecting lines for subjects within paired groups
% paired_indices_for_lines = {[2,3], [4,5], [6,7]}; % Low/High TDS, 2/3 Distr, No/Yes PSD
% for iS = 1:num_subjects_to_process
%     subj_line_color = custom_colors_rgb{mod(iS-1, length(custom_colors_rgb)) + 1};
%     for p_idx = 1:length(paired_indices_for_lines)
%         pair = paired_indices_for_lines{p_idx};
%         idx1_cond = pair(1); 
%         idx2_cond = pair(2);
% 
%         y1_val = subject_condition_means(iS, idx1_cond);
%         y2_val = subject_condition_means(iS, idx2_cond);
% 
%         if ~isnan(y1_val) && ~isnan(y2_val)
%             x1_pos_line = x_positions(idx1_cond); % Use base x-positions for lines
%             x2_pos_line = x_positions(idx2_cond);
%             plot(ax, [x1_pos_line, x2_pos_line], [y1_val, y2_val], ...
%                  '-', 'Color', subj_line_color, 'LineWidth', 0.75);
%         end
%     end
% end

% Plot overall pooled average dots (black), error bars, and Mean +/- SE text for ALL conditions
for k_cond = 1:num_conditions
    current_x_base = x_positions(k_cond);
    mean_val = overall_pooled_mean_accuracy(k_cond); % This is from ALL sessions pooled
    sem_val = overall_pooled_sem_accuracy(k_cond);   % This is from ALL sessions pooled
    
    if ~isnan(mean_val)
        if isnan(sem_val); sem_val = 0; end 
        errorbar(ax, current_x_base, mean_val, sem_val, ...
                 'k', 'LineWidth', 1.5, 'CapSize', 10, 'LineStyle','none');
        plot(ax, current_x_base, mean_val, 'o', ...
             'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k', 'MarkerSize', 6); %%% CHANGED: MarkerSize 6
        
        max_data_y_val = max(max_data_y_val, mean_val + sem_val); % Update based on these main points too
        min_data_y_val = min(min_data_y_val, mean_val - sem_val);
        
        % text_str = sprintf('%.2f\n%c%.2f', mean_val, char(177), sem_val); 
        % y_text_pos = mean_val + sem_val + 0.025; 
        % y_text_pos = max(y_text_pos, mean_val + 0.03); % Ensure text is above marker
        % text(ax, current_x_base, y_text_pos, text_str, ...
        %     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        %     'FontSize', mean_se_text_font_size, 'Color', 'k');
        % max_overall_y_plus_text = max(max_overall_y_plus_text, y_text_pos + 0.04); 
    end
end

if isinf(max_data_y_val); max_data_y_val = 1.0; end 
if isinf(min_data_y_val); min_data_y_val = 0.0; end
if isinf(max_overall_y_plus_text); max_overall_y_plus_text = max_data_y_val; end

y_span_data = max_data_y_val - min_data_y_val;
if y_span_data <= 1e-6; y_span_data = 0.5; end 

stat_bar_y_level_start_offset = 0.10 * y_span_data; 
stat_bar_y_level = max(max_data_y_val, max_overall_y_plus_text) + stat_bar_y_level_start_offset;
min_stat_bar_y_level = max(max_data_y_val, max_overall_y_plus_text) + 0.08; 
stat_bar_y_level = max(stat_bar_y_level, min_stat_bar_y_level);

star_text_y_offset_from_bar = 0.01 * y_span_data; 
cap_height_on_bar = 0.015 * y_span_data;

% --- Statistical Tests (ttest2 on pooled session data) and Significance Bars ---
stat_pairs_indices = {[2,3], [4,5], [6,7]}; 
y_max_for_all_stats_text = stat_bar_y_level; 

fprintf('\n--- Pairwise Statistical Test Results (Independent Two-Sample t-test on All Sessions) ---\n');
for p_idx = 1:length(stat_pairs_indices)
    pair = stat_pairs_indices{p_idx};
    idx1 = pair(1); idx2 = pair(2);
    
    % Data for ttest2: all session accuracies for each condition in the pair
    data1_all_sessions = pooled_session_accuracies_by_condition{idx1};
    data2_all_sessions = pooled_session_accuracies_by_condition{idx2};
    
    % Remove NaNs for ttest2
    data1_valid_sessions = data1_all_sessions(~isnan(data1_all_sessions));
    data2_valid_sessions = data2_all_sessions(~isnan(data2_all_sessions));

    
    fprintf('Comparison: %s vs %s\n', condition_labels{idx1}, condition_labels{idx2});
    
    % Ensure enough data points in each group for ttest2 (typically >1)
    if length(data1_valid_sessions) >= 2 && length(data2_valid_sessions) >= 2
        % Perform independent two-sample t-test
        [h_ttest, p_value, ci_ttest, stats_ttest] = ttest2(data1_valid_sessions, data2_valid_sessions); 
        
        fprintf('  N(%s) = %d sessions, N(%s) = %d sessions\n', condition_labels{idx1}, length(data1_valid_sessions), condition_labels{idx2}, length(data2_valid_sessions));
        fprintf('  t(%d) = %.3f, p = %.4f\n', stats_ttest.df, stats_ttest.tstat, p_value);
        
        if p_value < 0.001; stars = '***';
        elseif p_value < 0.01; stars = '**';
        elseif p_value < 0.05; stars = '*';
        else; stars = 'n.s.'; 
        end
        fprintf('  Significance on plot: %s\n', stars);
    else
        stars = 'n.d.'; % Not determined
        fprintf('  Not enough valid session data for test. N(%s)=%d, N(%s)=%d.\n', condition_labels{idx1}, length(data1_valid_sessions), condition_labels{idx2}, length(data2_valid_sessions));
    end
    fprintf('-----------------------------------------\n');
        
    x1_pos_bar = x_positions(idx1); % Significance bar connects the main pooled means
    x2_pos_bar = x_positions(idx2);
    plot(ax, [x1_pos_bar, x2_pos_bar], [stat_bar_y_level, stat_bar_y_level], '-k', 'LineWidth', 1);
    plot(ax, [x1_pos_bar, x1_pos_bar], [stat_bar_y_level - cap_height_on_bar, stat_bar_y_level], '-k', 'LineWidth', 1);
    plot(ax, [x2_pos_bar, x2_pos_bar], [stat_bar_y_level - cap_height_on_bar, stat_bar_y_level], '-k', 'LineWidth', 1);
    
    y_pos_stars = stat_bar_y_level + star_text_y_offset_from_bar;
    text(ax, mean([x1_pos_bar,x2_pos_bar]), y_pos_stars, stars, ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', stat_font_size, 'FontWeight', 'bold', 'Color', 'k');
    
    y_max_for_all_stats_text = max(y_max_for_all_stats_text, y_pos_stars + 0.03); 
end
fprintf('\n'); 
hold(ax, 'off');

% --- Aesthetics ---
set(ax, 'XTick', x_positions);
set(ax, 'XTickLabel', condition_labels);
xtickangle(ax, 30);
xlim(ax, [min(x_positions) - x_group_spacing*0.6, max(x_positions) + x_group_spacing*0.6]);
xlabel(ax, 'WM Condition', 'FontSize', plot_font_size);
ylabel(ax, 'Average Accuracy', 'FontSize', plot_font_size);
title(ax, 'WM: Accuracy by Condition (Pooled Session Means & Stats)', 'FontSize', plot_font_size + 1, 'FontWeight','bold'); % Updated title
ax.FontSize = plot_font_size;
grid(ax, 'off');

final_ylim_min = max(0, min_data_y_val - 0.05 * y_span_data); % Ensure min_data_y_val is updated by all plotted points
final_ylim_max_from_data_and_text = max(max_data_y_val + 0.05 * y_span_data, max_overall_y_plus_text);
final_ylim_max = min(1.1, max(final_ylim_max_from_data_and_text, y_max_for_all_stats_text)); 
if final_ylim_max <= final_ylim_min; final_ylim_max = final_ylim_min + 0.1; end % Ensure some range
ylim(ax, [final_ylim_min, final_ylim_max]);

% --- Legend ---
lgd = matlab.graphics.illustration.Legend.empty();
if plotted_anything_for_legend
    valid_legend_handles = subject_legend_handles(subject_has_legend_entry);
    valid_legend_labels = subject_legend_labels(subject_has_legend_entry);
    if ~isempty(valid_legend_handles)
        lgd = legend(ax, valid_legend_handles, valid_legend_labels, ...
               'Location', 'eastoutside', 'FontSize', plot_font_size - 2);
        lgd.Box = 'off';
    end
end

% --- Saving Outputs ---
if exist('fig_handle', 'var') && isvalid(fig_handle)
    figure(fig_handle);
    base_filename = 'wm_accuracy_by_condition_grouped';
    date_str = datestr(now, 'yyyymmdd_HHMMSS');
    full_base_filename = [base_filename '_' date_str];
    save_folder = 'WM_Figures_GroupedConditions';

    if ~exist(save_folder, 'dir')
       try
           mkdir(save_folder);
       catch ME_mkdir
           fprintf('Error creating save dir ''%s'': %s. Saving to current dir.\n', save_folder, ME_mkdir.message);
           save_folder = '.';
       end
    end

    filepath_base = fullfile(save_folder, full_base_filename);

    % Save Figure (PNG and EPS)
    try
        png_filename = [filepath_base '.png'];
        saveas(fig_handle, png_filename);
        fprintf('Figure saved as: %s\n', png_filename);

        eps_filename = [filepath_base '.eps'];
        print(fig_handle, eps_filename, '-depsc');
        fprintf('Figure saved as: %s\n', eps_filename);
    catch ME_save
        fprintf('Error saving figure: %s\n', ME_save.message);
    end

    % Save Data to CSVs
    try
        % --- Summary Data CSV ---
        condition_headers = matlab.lang.makeValidName(condition_labels);

        row_labels = {'Overall_Pooled_Mean'; 'Overall_Pooled_SEM'};
        data_matrix = [overall_pooled_mean_accuracy; overall_pooled_sem_accuracy];

        plotted_subj_indices = find(subject_has_legend_entry);
        for i = 1:length(plotted_subj_indices)
            subj_idx = plotted_subj_indices(i);
            row_labels{end+1,1} = subject_legend_labels{subj_idx};
            data_matrix = [data_matrix; subject_condition_means(subj_idx, :)];
        end

        summary_table = array2table(data_matrix, 'VariableNames', condition_headers);
        summary_table = addvars(summary_table, row_labels, 'Before', 1, 'NewVariableNames', 'Group');

        csv_summary_filename = [filepath_base '_summary_data.csv'];
        writetable(summary_table, csv_summary_filename);
        fprintf('Summary data saved to: %s\n', csv_summary_filename);

        % --- Statistics CSV ---
        stats_comparisons_list = {};
        stats_n1_list = [];
        stats_n2_list = [];
        stats_t_list = [];
        stats_df_list = [];
        stats_p_list = [];

        stat_pairs_to_process = {[2,3], [4,5], [6,7]};
        for p_idx = 1:length(stat_pairs_to_process)
            pair = stat_pairs_to_process{p_idx};
            idx1 = pair(1);
            idx2 = pair(2);

            data1_valid = pooled_session_accuracies_by_condition{idx1}(~isnan(pooled_session_accuracies_by_condition{idx1}));
            data2_valid = pooled_session_accuracies_by_condition{idx2}(~isnan(pooled_session_accuracies_by_condition{idx2}));

            stats_comparisons_list{end+1,1} = sprintf('%s vs %s', condition_labels{idx1}, condition_labels{idx2});
            stats_n1_list(end+1,1) = length(data1_valid);
            stats_n2_list(end+1,1) = length(data2_valid);

            if length(data1_valid) >= 2 && length(data2_valid) >= 2
                [~, p_val, ~, stats] = ttest2(data1_valid, data2_valid);
                stats_t_list(end+1,1) = stats.tstat;
                stats_df_list(end+1,1) = stats.df;
                stats_p_list(end+1,1) = p_val;
            else
                stats_t_list(end+1,1) = NaN;
                stats_df_list(end+1,1) = NaN;
                stats_p_list(end+1,1) = NaN;
            end
        end

        if ~isempty(stats_comparisons_list)
             stats_table = table(stats_comparisons_list, stats_n1_list, stats_n2_list, stats_t_list, stats_df_list, stats_p_list, ...
                'VariableNames', {'Comparison', 'N1_Sessions', 'N2_Sessions', 'T_Statistic', 'DF', 'PValue'});

            csv_stats_filename = [filepath_base '_statistics.csv'];
            writetable(stats_table, csv_stats_filename);
            fprintf('Statistical results saved to: %s\n', csv_stats_filename);
        end

    catch ME_csv
        fprintf('Error saving CSV data: %s\n', ME_csv.message);
    end

else
    fprintf('Figure handle not valid/created. Figure not saved.\n');
end
end % End of main function






function plot_wm_accuracy_vs_delay(metrics_mt)
% plot_wm_accuracy_vs_delay Plots WM accuracy vs. delay time with stats.
%   - Significance display: Stars for p<0.05, "n.s." for p>=0.05. No raw p-values.
%   - Legend fixed and styled.
%   - X-ticks are unique delay values. X-axis title: "Delay (s)".
%   - Y-limits are dynamic based on data, within [0,1].
%   - Subject legend labels: "Subject X (initial)".
%   - Overall: black line with SEM error bars (from ALL sessions pooled).
%   - Figure size 350x200, font 14, legend right (no box), no grid.
%   - Saves as PNG.

% --- Hardcoded Subject Information and Colors ---
subject_id_list_original = {
    'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan'
};
custom_colors = {
    [245, 124, 110]/255; [242, 181, 110]/255; [251, 231, 158]/255;
    [132, 195, 183]/255; [136, 215, 218]/255; [113, 184, 237]/255;
    [184, 174, 234]/255; [242, 168, 218]/255
};
overall_color = [0 0 0];

% --- Font Size & Plotting Constants ---
base_font_size = 14;
stat_font_size = base_font_size - 2; % Slightly larger for stars/ns

% --- Input Validation ---
if nargin < 1; error('Usage: %s(metrics_mt)', mfilename); end
if ~iscell(metrics_mt); error('metrics_mt must be a cell array.'); end

% --- Figure Setup ---
figure_width = 600;
figure_height = 400;
fig = figure('Position', [100, 100, figure_width, figure_height]);
ax = gca;
hold(ax, 'on');

num_subjects_in_data = length(metrics_mt);

% --- Step 1: Identify all unique delay times ---
all_delays_encountered = [];
for iS = 1:num_subjects_in_data
    if iS > length(metrics_mt) || isempty(metrics_mt{iS}); continue; end
    for iD = 1:length(metrics_mt{iS})
        if isstruct(metrics_mt{iS}(iD)) && isfield(metrics_mt{iS}(iD), 'conditions_nDistr_delays_wm') && ...
           ~isempty(metrics_mt{iS}(iD).conditions_nDistr_delays_wm) && size(metrics_mt{iS}(iD).conditions_nDistr_delays_wm,2) >= 2
            all_delays_encountered = [all_delays_encountered; metrics_mt{iS}(iD).conditions_nDistr_delays_wm(:, 2)];
        end
    end
end
if isempty(all_delays_encountered)
    warning('No delay time data found. Cannot generate plot.');
    text(ax, 0.5, 0.5, 'No WM delay data found.', 'HorizontalAlignment', 'center', 'Units', 'normalized', 'FontSize', base_font_size, 'Color', 'red');
    hold(ax,'off'); if isvalid(fig); close(fig); end; return;
end
master_unique_sorted_delays = sort(unique(all_delays_encountered));
num_master_delays = length(master_unique_sorted_delays);

% --- Initialize data storage ---
pooled_session_accuracies_by_delay = cell(1, num_master_delays);
subject_means_on_master_delays = NaN(num_subjects_in_data, num_master_delays);
subject_sems_on_master_delays = NaN(num_subjects_in_data, num_master_delays);
subject_has_data_mask = false(1, num_subjects_in_data);

% --- Process data per session, aggregate for overall and subject ---
for iS = 1:num_subjects_in_data
    if iS > length(metrics_mt) || isempty(metrics_mt{iS}); continue; end
    current_subject_session_accuracies_by_delay = cell(1, num_master_delays);
    sessions_processed_for_subject = 0;
    for iD = 1:length(metrics_mt{iS})
        if ~isstruct(metrics_mt{iS}(iD)); continue; end
        session_data = metrics_mt{iS}(iD);
        if isfield(session_data, 'conditions_nDistr_delays_wm') && isfield(session_data, 'accuracyMeanSE_wm') && ...
           ~isempty(session_data.conditions_nDistr_delays_wm) && ~isempty(session_data.accuracyMeanSE_wm) && ...
           size(session_data.conditions_nDistr_delays_wm, 1) == size(session_data.accuracyMeanSE_wm, 1) && ...
           size(session_data.conditions_nDistr_delays_wm, 2) >= 2 && size(session_data.accuracyMeanSE_wm, 2) >= 1
            sessions_processed_for_subject = sessions_processed_for_subject + 1;
            session_delays = session_data.conditions_nDistr_delays_wm(:, 2);
            session_accuracies_col1 = session_data.accuracyMeanSE_wm(:, 1);
            for k_delay = 1:num_master_delays
                target_delay = master_unique_sorted_delays(k_delay);
                idx_matching_delay_in_session = (abs(session_delays - target_delay) < 1e-6);
                if any(idx_matching_delay_in_session)
                    accuracies_for_this_delay_this_session = session_accuracies_col1(idx_matching_delay_in_session);
                    mean_acc_for_delay_this_session = mean(accuracies_for_this_delay_this_session, 'omitnan');
                    if ~isnan(mean_acc_for_delay_this_session)
                        current_subject_session_accuracies_by_delay{k_delay} = [current_subject_session_accuracies_by_delay{k_delay}, mean_acc_for_delay_this_session];
                        pooled_session_accuracies_by_delay{k_delay} = [pooled_session_accuracies_by_delay{k_delay}, mean_acc_for_delay_this_session];
                    end
                end
            end
        end
    end 
    if sessions_processed_for_subject > 0; subject_has_data_mask(iS) = true; end
    for k_delay = 1:num_master_delays
        session_accs_for_subj_at_delay = current_subject_session_accuracies_by_delay{k_delay};
        if ~isempty(session_accs_for_subj_at_delay)
            subject_means_on_master_delays(iS, k_delay) = mean(session_accs_for_subj_at_delay, 'omitnan');
            num_valid_sessions = sum(~isnan(session_accs_for_subj_at_delay));
            if num_valid_sessions > 1; subject_sems_on_master_delays(iS, k_delay) = std(session_accs_for_subj_at_delay, 'omitnan') / sqrt(num_valid_sessions);
            else; subject_sems_on_master_delays(iS, k_delay) = 0; end
        end
    end
end 

% --- Calculate Overall Plot Data & Y-limit data collection ---
overall_means_on_master_delays = NaN(1, num_master_delays);
overall_sems_on_master_delays = NaN(1, num_master_delays);
all_y_minus_sem_collected = []; all_y_plus_sem_collected = [];  
max_data_y_extent = -Inf; % Track max y of data lines (mean+sem)

for k_delay = 1:num_master_delays
    all_sess_accs_at_delay = pooled_session_accuracies_by_delay{k_delay};
    if ~isempty(all_sess_accs_at_delay)
        current_mean = mean(all_sess_accs_at_delay, 'omitnan');
        overall_means_on_master_delays(k_delay) = current_mean;
        num_valid_pooled_sessions = sum(~isnan(all_sess_accs_at_delay));
        current_sem = 0;
        if num_valid_pooled_sessions > 1; current_sem = std(all_sess_accs_at_delay, 'omitnan') / sqrt(num_valid_pooled_sessions); end
        overall_sems_on_master_delays(k_delay) = current_sem;
        if ~isnan(current_mean)
            all_y_minus_sem_collected = [all_y_minus_sem_collected, current_mean - current_sem];
            all_y_plus_sem_collected = [all_y_plus_sem_collected, current_mean + current_sem];
            max_data_y_extent = max(max_data_y_extent, current_mean + current_sem);
        end
    end
end

% --- Plotting Data Lines ---
legend_handles = [];
plotted_anything_for_legend = false;



valid_subject_indices_for_plot = find(subject_has_data_mask);
for k_subj_plot = 1:length(valid_subject_indices_for_plot)
    iS = valid_subject_indices_for_plot(k_subj_plot);
    subj_label_for_legend = sprintf('Subj %d', iS); 
    if iS <= length(subject_id_list_original) && ~isempty(subject_id_list_original{iS})
        original_id = subject_id_list_original{iS}; initial = upper(original_id(1));
        subj_label_for_legend = sprintf('Subject %c', initial); 
    end
    if iS <= length(custom_colors); subj_color = custom_colors{iS}; else; subj_color = rand(1,3); end
    current_means = subject_means_on_master_delays(iS, :); current_sems = subject_sems_on_master_delays(iS, :);
    if any(~isnan(current_means))
        h_subj = errorbar(ax, master_unique_sorted_delays, current_means, current_sems, ...
            '-o', 'Color', subj_color, 'MarkerFaceColor', subj_color, 'MarkerEdgeColor', subj_color, ...
            'LineWidth', 1.5, 'CapSize', 5, 'MarkerSize', 5, 'DisplayName', subj_label_for_legend);
        legend_handles(end+1) = h_subj; plotted_anything_for_legend = true;
        valid_idx_subj = ~isnan(current_means); means_to_collect = current_means(valid_idx_subj);
        sems_to_collect = current_sems(valid_idx_subj); sems_to_collect(isnan(sems_to_collect)) = 0;
        all_y_minus_sem_collected = [all_y_minus_sem_collected, means_to_collect - sems_to_collect];
        all_y_plus_sem_collected = [all_y_plus_sem_collected, means_to_collect + sems_to_collect];
        max_data_y_extent = max(max_data_y_extent, max(means_to_collect + sems_to_collect,[],'omitnan'));
    end
end


if any(~isnan(overall_means_on_master_delays))
    h_overall = errorbar(ax, master_unique_sorted_delays, overall_means_on_master_delays, overall_sems_on_master_delays, ...
        '-o', 'Color', overall_color, 'MarkerFaceColor', overall_color, 'MarkerEdgeColor', overall_color, ...
        'LineWidth', 2, 'CapSize', 5, 'MarkerSize', 5, 'DisplayName', 'Overall');
    legend_handles(end+1) = h_overall;
    plotted_anything_for_legend = true;
end

% --- Basic Plot Styling (Labels, Title, Ticks) ---
set(ax, 'FontSize', base_font_size);
xlabel(ax, 'Delay (s)', 'FontSize', base_font_size); 
ylabel(ax, 'Average Accuracy', 'FontSize', base_font_size);
title(ax, 'WM: Accuracy vs. Delay', 'FontSize', base_font_size, 'FontWeight', 'bold');
grid(ax, 'off'); 
if ~isempty(master_unique_sorted_delays) && plotted_anything_for_legend
    set(ax, 'XTick', master_unique_sorted_delays);
    if length(master_unique_sorted_delays) > 6; xtickangle(ax, 45); end
else
    set(ax, 'XTick', []); xlim(ax, [0 1]); 
end

% --- Create Legend (before stats, so its position isn't affected by stat bars initially) ---
lgd = matlab.graphics.illustration.Legend.empty(); % Initialize to empty
if plotted_anything_for_legend
    lgd = legend(ax, legend_handles, 'Location', 'eastoutside', 'FontSize', base_font_size);
    lgd.Box = 'off';
end

% --- Perform T-tests and Add Significance Bars ---
y_max_for_stats_plotting = -Inf; % Will track the highest point reached by stats annotations
if num_master_delays >= 2 && any(~isnan(overall_means_on_master_delays))
    data_longest_delay_pooled = pooled_session_accuracies_by_delay{num_master_delays};
    data_longest_delay_pooled_clean = data_longest_delay_pooled(~isnan(data_longest_delay_pooled));
    
    if ~isempty(data_longest_delay_pooled_clean) && length(data_longest_delay_pooled_clean) >=2
        % Determine starting Y for the first stat bar
        if isinf(max_data_y_extent); max_data_y_extent = 1.0; end % Fallback if no data plotted
        y_current_level_for_bar = max_data_y_extent + 0.05; % Start 5% above max data
        
        % Estimate a reasonable vertical step for bars
        plot_y_range = max(0.2, max_data_y_extent - max(0, min(all_y_minus_sem_collected,[],'omitnan'))); % Min range of 0.2 for calc
        stat_bar_vertical_step = plot_y_range * 0.08; % Each bar takes about 12% of data y-range
        if stat_bar_vertical_step < 0.02; stat_bar_vertical_step = 0.02; end % Minimum step
        stat_cap_height = stat_bar_vertical_step * 0.05; 
        text_v_offset_above_bar = stat_bar_vertical_step * 0.05;

        for k_stat = 1:(num_master_delays - 1)
            data_current_delay_pooled = pooled_session_accuracies_by_delay{k_stat};
            data_current_delay_pooled_clean = data_current_delay_pooled(~isnan(data_current_delay_pooled));

            text_to_display = 'n.d.'; % Default for "no data" or "not determined"
            if length(data_current_delay_pooled_clean) >= 2 && length(data_longest_delay_pooled_clean) >=2
                [~, p_val] = ttest2(data_current_delay_pooled_clean, data_longest_delay_pooled_clean);
                
                if p_val < 0.001; text_to_display = '***';
                elseif p_val < 0.01; text_to_display = '**';
                elseif p_val < 0.05; text_to_display = '*';
                else; text_to_display = 'n.s.'; % Non-significant
                end
            elseif length(data_current_delay_pooled_clean) < 2
                 % Not enough data in current delay group
            end

            x1 = master_unique_sorted_delays(k_stat);
            x2 = master_unique_sorted_delays(num_master_delays);
            
            plot(ax, [x1, x2], [y_current_level_for_bar, y_current_level_for_bar], '-k', 'LineWidth', 0.75, HandleVisibility='off');
            plot(ax, [x1, x1], [y_current_level_for_bar - stat_cap_height, y_current_level_for_bar], '-k', 'LineWidth', 0.75, HandleVisibility='off');
            plot(ax, [x2, x2], [y_current_level_for_bar - stat_cap_height, y_current_level_for_bar], '-k', 'LineWidth', 0.75, HandleVisibility='off');
            
            text(ax, mean([x1,x2]), y_current_level_for_bar + text_v_offset_above_bar, text_to_display, ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', stat_font_size, 'Color', 'k');

            y_current_level_for_bar = y_current_level_for_bar + stat_bar_vertical_step; 
        end
        y_max_for_stats_plotting = y_current_level_for_bar - stat_bar_vertical_step + text_v_offset_above_bar*2; % Top of last text
    end
end

% --- Final Y Lim based on data AND stats ---
final_ylim_min = 0; final_ylim_max = 1.05; % Defaults for accuracy
if ~isempty(all_y_minus_sem_collected)
    min_data_y_val = min(all_y_minus_sem_collected,[],'omitnan'); max_data_y_val = max(all_y_plus_sem_collected,[],'omitnan');
    min_data_y_val = max(0, min_data_y_val); max_data_y_val = min(1, max_data_y_val);
    plot_data_range = max_data_y_val - min_data_y_val;
    padding = 0.05; 
    if plot_data_range > 1e-6; padding = plot_data_range * 0.10; end
    final_ylim_min = max(0, min_data_y_val - padding);
    tentative_ylim_max = min(1, max_data_y_val + padding);
    if ~isinf(y_max_for_stats_plotting) % If stats were plotted
        final_ylim_max = max(tentative_ylim_max, y_max_for_stats_plotting + padding*0.5);
    else
        final_ylim_max = tentative_ylim_max;
    end
    if (final_ylim_max - final_ylim_min) < 0.2 
        mid_point = (final_ylim_max + final_ylim_min) / 2; mid_point = max(0.1, min(0.9, mid_point));
        final_ylim_min = mid_point - 0.1; final_ylim_max = mid_point + 0.1;
    end
    final_ylim_min = max(0, final_ylim_min); final_ylim_max = min(1.1, final_ylim_max); % Allow slight overshoot for stats
    if final_ylim_max <= final_ylim_min; final_ylim_max = final_ylim_min + 0.1; end
end
ylim(ax, [final_ylim_min, final_ylim_max]);

% --- Adjust XLim after all plotting ---
if ~isempty(master_unique_sorted_delays) && plotted_anything_for_legend
    xlim_padding = 0.1 * (max(master_unique_sorted_delays) - min(master_unique_sorted_delays));
    if xlim_padding == 0 || isnan(xlim_padding); xlim_padding = 0.5; end
    current_xlim = [min(master_unique_sorted_delays) - xlim_padding, max(master_unique_sorted_delays) + xlim_padding];
    if current_xlim(1) >= current_xlim(2); current_xlim = [master_unique_sorted_delays(1)-0.5, master_unique_sorted_delays(1)+0.5]; end
    xlim(ax, current_xlim);
end

% --- Adjust layout for legend (if it was created) ---
drawnow; 
if plotted_anything_for_legend && ~isempty(lgd) && isvalid(lgd) && strcmp(lgd.Location, 'eastoutside')
    try 
        original_ax_units = get(ax, 'Units'); original_lgd_units = get(lgd, 'Units');
        set(ax, 'Units', 'normalized'); set(lgd, 'Units', 'normalized');
        drawnow; pause(0.1);
        ax_pos_norm = get(ax, 'Position'); lgd_outer_pos_norm = get(lgd, 'OuterPosition');
        max_allowable_ax_width = lgd_outer_pos_norm(1) - ax_pos_norm(1) - 0.05; 
        if ax_pos_norm(3) > max_allowable_ax_width && max_allowable_ax_width > 0.05 
            set(ax, 'Position', [ax_pos_norm(1), ax_pos_norm(2), max_allowable_ax_width, ax_pos_norm(4)]);
        end
        set(ax, 'Units', original_ax_units); set(lgd, 'Units', original_lgd_units);
    catch ME_layout
        fprintf('Warning: Could not auto-adjust layout: %s\n', ME_layout.message);
        % if exist('original_ax_units','var') && isvalid(original_ax_units); set(ax, 'Units', original_ax_units); end
        % if exist('original_lgd_units','var') && isvalid(original_lgd_units); set(lgd, 'Units', original_lgd_units); end
    end
end

% --- Saving Outputs ---
% MODIFIED: This section now saves figures (PNG, EPS) and data (CSV).
if exist('fig', 'var') && isvalid(fig)
    figure(fig); 
    base_filename = 'wm_accuracy_vs_delay_stats'; 
    date_str = datestr(now, 'yyyymmdd_HHMMSS');
    full_base_filename = [base_filename '_' date_str]; 
    save_folder = 'WM_Figures_AccuracyDelay';
    
    if ~exist(save_folder, 'dir')
        try; mkdir(save_folder); 
        catch ME_mkdir; fprintf('Error creating save dir ''%s'': %s. Saving to current dir.\n', save_folder, ME_mkdir.message); save_folder = '.'; end
    end
    
    filepath_base = fullfile(save_folder, full_base_filename);

    % Save Figure (PNG and EPS)
    try
        png_filename = [filepath_base '.png']; 
        saveas(fig, png_filename); 
        fprintf('Figure saved as: %s\n', png_filename);

        eps_filename = [filepath_base '.eps'];
        print(fig, eps_filename, '-depsc');
        fprintf('Figure saved as: %s\n', eps_filename);
    catch ME_save
        fprintf('Error saving figure: %s\n', ME_save.message); 
    end

    % Save Data to CSVs
    try
        % --- Summary Data CSV ---
        delay_headers = arrayfun(@(x) sprintf('Delay_%.3f_s', x), master_unique_sorted_delays, 'UniformOutput', false);
        
        row_labels = {'Overall_Mean'; 'Overall_SE'};
        data_matrix = [overall_means_on_master_delays; overall_sems_on_master_delays];
        
        plotted_subj_indices = find(subject_has_data_mask);
        for i = 1:length(plotted_subj_indices)
            subj_idx = plotted_subj_indices(i);
            
            subj_label = sprintf('Subj_%d', subj_idx); 
            if subj_idx <= length(subject_id_list_original) && ~isempty(subject_id_list_original{subj_idx})
                original_id = subject_id_list_original{subj_idx};
                subj_label = sprintf('Subject_%s', original_id); 
            end

            row_labels{end+1,1} = [subj_label '_Mean'];
            row_labels{end+1,1} = [subj_label '_SE'];
            data_matrix = [data_matrix; subject_means_on_master_delays(subj_idx,:); subject_sems_on_master_delays(subj_idx,:)];
        end

        summary_table = array2table(data_matrix, 'VariableNames', delay_headers);
        summary_table = addvars(summary_table, row_labels, 'Before', 1, 'NewVariableNames', 'Series');

        csv_summary_filename = [filepath_base '_summary_data.csv'];
        writetable(summary_table, csv_summary_filename);
        fprintf('Summary data saved to: %s\n', csv_summary_filename);

        % --- Statistics CSV ---
        if ~isempty(stats_comparisons)
            stats_table = table(stats_comparisons', stats_p_values', 'VariableNames', {'Comparison', 'PValue'});
            csv_stats_filename = [filepath_base '_statistics.csv'];
            writetable(stats_table, csv_stats_filename);
            fprintf('Statistical results saved to: %s\n', csv_stats_filename);
        end

    catch ME_csv
        fprintf('Error saving CSV data: %s\n', ME_csv.message);
    end
end
hold(ax, 'off');
end




function plot_cr_stimuli_at_threshold_accuracy(metrics_mt, accuracy_threshold)
% plot_cr_stimuli_at_threshold_accuracy Calculates and plots the estimated
% number of stimuli at which accuracy drops to a threshold, plotted horizontally.
%   - Overall average (from ALL SESSIONS pooled) is at the top.
%   - Subject labels are initials on the Y-axis.
%   - Figure size 500x200, font size 14, legend on right, no box, no grid.
%   - Y-axis title is "Subject".
%   - Saves figure as PNG, EPS, and saves data as CSV.
%   - Acc_NumStim is assumed 19 rows (for stimuli 2-20), Col 1 is accuracy.

% --- Default Threshold ---
if nargin < 2 || isempty(accuracy_threshold)
    accuracy_threshold = 0.75; % Default to 75%
end
if accuracy_threshold <= 0 || accuracy_threshold >= 1
    warning('Accuracy threshold "%.2f" is outside (0,1). Using 0.75.', accuracy_threshold);
    accuracy_threshold = 0.75;
end

% --- Hardcoded Subject Information and Colors ---
subject_id_list_original = {
    'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan'
};
custom_colors = {
    [245, 124, 110]/255; [242, 181, 110]/255; [251, 231, 158]/255; % Colors 1-3
    [132, 195, 183]/255; [136, 215, 218]/255; [113, 184, 237]/255; % Colors 4-6
    [184, 174, 234]/255; [242, 168, 218]/255 % Colors 7-8
};
custom_colors = {
    [0.945, 0.345, 0.275]; % #F15846
    [1.0, 0.690, 0.314];   % #FFB050
    [0.984, 0.906, 0.620]; % #FBE79E
    [0.533, 0.843, 0.855]; % #88D7DA
    [0.341, 0.663, 0.902]; % #57A9E6
    [0.420, 0.447, 0.714]  % #6B72B6
};
overall_color = [0 0 0]; 

% --- Font Size & Plotting Constants ---
base_font_size = 14;
y_pos_overall = 1; 
gap_after_overall = 2.0; 
inter_subject_gap = 1.0;
epsilon = 1e-7; 

% --- Expected input Acc_NumStim properties ---
expected_acc_rows = 19; 

% --- Input Validation ---
if nargin < 1
    error('Usage: %s(metrics_mt, [accuracy_threshold])', mfilename);
end
if ~iscell(metrics_mt)
    error('metrics_mt must be a cell array.');
end

% --- Figure Setup ---
figure_width = 400;  % *** MODIFIED: Figure size updated
figure_height = 240; % *** MODIFIED: Figure size updated
fig = figure('Position', [100, 100, figure_width, figure_height]);
ax = gca;
hold(ax, 'on');

num_subjects_in_data = length(metrics_mt);
subject_metric_all_subjects = NaN(1, num_subjects_in_data);
subject_sem_all_subjects = NaN(1, num_subjects_in_data);
subject_initial_labels_all = cell(1, num_subjects_in_data);
valid_subject_mask = false(1, num_subjects_in_data);
all_sessions_metric_pooled = []; 

% --- Data Calculation and Subject-Level Aggregation ---
for iS = 1:num_subjects_in_data
    session_metric_list_for_this_subject = [];
    if iS > length(metrics_mt) || isempty(metrics_mt{iS}); continue; end
    for iD = 1:length(metrics_mt{iS})
        if ~isstruct(metrics_mt{iS}(iD)); continue; end
        session_data = metrics_mt{iS}(iD);
        estimated_stim_at_thresh = NaN; 
        if isfield(session_data, 'Acc_NumStim') && ~isempty(session_data.Acc_NumStim)
            acc_data_matrix = session_data.Acc_NumStim;
            
            if size(acc_data_matrix, 1) == expected_acc_rows && size(acc_data_matrix, 2) >= 1
                accuracy_session_raw = acc_data_matrix(:, 1);
                stimulus_offset = 1; 
                actual_stim_counts_map = (1:expected_acc_rows)' + stimulus_offset;
                valid_idx = ~isnan(accuracy_session_raw);
                accuracy_valid = accuracy_session_raw(valid_idx);
                stim_counts_for_valid_acc = actual_stim_counts_map(valid_idx);
                if length(accuracy_valid) >= 2
                    accuracy_noisy = accuracy_valid + (rand(size(accuracy_valid)) * epsilon) - (epsilon/2);
                    [sorted_acc_noisy, sort_idx] = sort(accuracy_noisy); 
                    sorted_stim_counts = stim_counts_for_valid_acc(sort_idx);
                    
                    [unique_sorted_acc, ia] = unique(sorted_acc_noisy, 'stable');
                    unique_corresponding_stim_counts = sorted_stim_counts(ia);
                    if length(unique_sorted_acc) >= 2
                        try
                            estimated_stim_at_thresh = interp1(unique_sorted_acc, unique_corresponding_stim_counts, accuracy_threshold, 'pchip', 'extrap');
                        catch ME_interp
                            subject_id_str = sprintf('Subj %d', iS);
                            if iS <= length(subject_id_list_original); subject_id_str = subject_id_list_original{iS}; end
                            fprintf('Warning: Interpolation failed for %s, Session %d: %s. Data issues likely.\n', subject_id_str, iD, ME_interp.message);
                            estimated_stim_at_thresh = NaN;
                        end
                    end
                end
            end
            
            if ~isnan(estimated_stim_at_thresh)
                session_metric_list_for_this_subject = [session_metric_list_for_this_subject, estimated_stim_at_thresh];
                all_sessions_metric_pooled = [all_sessions_metric_pooled, estimated_stim_at_thresh];
            end
        end
    end
    if ~isempty(session_metric_list_for_this_subject)
        subject_metric_all_subjects(iS) = mean(session_metric_list_for_this_subject, 'omitnan');
        num_valid_sessions_for_subj = sum(~isnan(session_metric_list_for_this_subject));
        if num_valid_sessions_for_subj > 1
            subject_sem_all_subjects(iS) = std(session_metric_list_for_this_subject, 'omitnan') / sqrt(num_valid_sessions_for_subj);
        else
            subject_sem_all_subjects(iS) = 0; 
        end
        valid_subject_mask(iS) = true;
        if iS <= length(subject_id_list_original) && ~isempty(subject_id_list_original{iS})
            original_id = subject_id_list_original{iS};
            subject_initial_labels_all{iS} = sprintf('%c', upper(original_id(1)));
        else
            subject_initial_labels_all{iS} = sprintf('S%d', iS); 
        end
    end
end
% --- Filter for valid subjects and prepare for plotting ---
actual_per_subject_metric_means = subject_metric_all_subjects(valid_subject_mask);
actual_per_subject_sems = subject_sem_all_subjects(valid_subject_mask);
final_subject_initials_for_ytick = subject_initial_labels_all(valid_subject_mask); % Renamed for clarity
num_valid_subjects_plotted = length(actual_per_subject_metric_means);
legend_handles = [];
plotted_anything_for_legend = false;
ytick_positions = [];
ytick_labels_list = {};

% --- Overall Average Calculation and Plotting (Topmost) ---
overall_mean_pooled = NaN;
overall_sem_pooled = NaN;
if ~isempty(all_sessions_metric_pooled)
    overall_mean_pooled = mean(all_sessions_metric_pooled, 'omitnan');
    non_nan_pooled_sessions_count = sum(~isnan(all_sessions_metric_pooled));
    if non_nan_pooled_sessions_count > 1
        overall_sem_pooled = std(all_sessions_metric_pooled, 'omitnan') / sqrt(non_nan_pooled_sessions_count);
    else
        overall_sem_pooled = 0; 
    end
    
    if ~isnan(overall_mean_pooled)
        % *** MODIFIED: Swapped x/y, plotting horizontally, updated MarkerSize ***
        h_overall = errorbar(ax, overall_mean_pooled, y_pos_overall, overall_sem_pooled, 'horizontal', 'o', ...
            'MarkerSize', 3, 'MarkerEdgeColor', overall_color, 'MarkerFaceColor', overall_color, ...
            'Color', overall_color, 'LineWidth', 1.5, 'CapSize', 10, 'DisplayName', 'Overall');
        legend_handles(end+1) = h_overall;
        plotted_anything_for_legend = true;
        ytick_positions(end+1) = y_pos_overall;
        ytick_labels_list{end+1} = 'Overall';
    end
end
% --- Plotting Subject Data ---
subject_y_positions = [];
if num_valid_subjects_plotted > 0
    current_y = y_pos_overall + gap_after_overall; 
    for k_subj = 1:num_valid_subjects_plotted
        subject_y_positions(k_subj) = current_y + (k_subj-1)*inter_subject_gap;
    end
end
original_indices_of_valid_subjects = find(valid_subject_mask);
for k = 1:num_valid_subjects_plotted
    iS_original_index = original_indices_of_valid_subjects(k);
    current_subject_initial_for_plot = final_subject_initials_for_ytick{k};
    if iS_original_index <= length(custom_colors); current_color = custom_colors{iS_original_index};
    else; current_color = rand(1,3); end
    
    y_pos = subject_y_positions(k);
    % *** MODIFIED: Swapped x/y, plotting horizontally, updated MarkerSize ***
    h_subj = errorbar(ax, actual_per_subject_metric_means(k), y_pos, actual_per_subject_sems(k), 'horizontal', 'o', ...
        'MarkerSize', 3, 'MarkerEdgeColor', current_color, 'MarkerFaceColor', current_color, ...
        'Color', current_color, 'LineWidth', 1.5, 'CapSize', 10, ...
        'DisplayName', current_subject_initial_for_plot);
    legend_handles(end+1) = h_subj;
    plotted_anything_for_legend = true; 
    ytick_positions(end+1) = y_pos;
    ytick_labels_list{end+1} = current_subject_initial_for_plot;
end
% --- Finalize Plot Styling ---
set(ax, 'FontSize', base_font_size);
if plotted_anything_for_legend
    lgd = legend(ax, legend_handles, 'Location', 'eastoutside', 'FontSize', base_font_size);
    lgd.Box = 'off';
    % *** MODIFIED: Changed from XTick to YTick ***
    if ~isempty(ytick_positions)
        set(ax, 'YTick', ytick_positions);
        set(ax, 'YTickLabel', ytick_labels_list);
    else; set(ax, 'YTick', []); set(ax, 'YTickLabel', {}); end
else
    text(ax, 0.5, 0.5, 'No data to plot.', 'HorizontalAlignment', 'center', 'Units', 'normalized', 'FontSize', base_font_size, 'Color', 'red');
    warning('No data plotted. Check .Acc_NumStim field and calculation method.');
    set(ax, 'YTick', []); set(ax, 'YTickLabel', {});
end

% *** MODIFIED: Swapped axis labels ***
x_axis_label_str = sprintf('Est. Stimuli at %.0f%% Acc.', accuracy_threshold * 100);
ylabel(ax, 'Subject', 'FontSize', base_font_size); 
xlabel(ax, x_axis_label_str, 'FontSize', base_font_size);
title(ax, 'CR: Est. Stimuli at Accuracy Threshold', 'FontSize', base_font_size, 'FontWeight', 'bold');
grid(ax, 'off'); 

% *** MODIFIED: Axis limit logic is now swapped for x and y ***
if ~isempty(ytick_positions)
    axis(ax, 'tight'); 
    xl = xlim(ax);
    min_x_val = 0.5; 
    
    max_x_data_point = -Inf;
    if ~isnan(overall_mean_pooled) && ~isnan(overall_sem_pooled)
        max_x_data_point = max(max_x_data_point, overall_mean_pooled + overall_sem_pooled);
    end
    if ~isempty(actual_per_subject_metric_means)
        valid_means = actual_per_subject_metric_means(~isnan(actual_per_subject_metric_means) & ~isnan(actual_per_subject_sems));
        valid_sems = actual_per_subject_sems(~isnan(actual_per_subject_metric_means) & ~isnan(actual_per_subject_sems));
        if ~isempty(valid_means); max_x_data_point = max(max_x_data_point, max(valid_means + valid_sems)); end
    end
    if isinf(max_x_data_point) || isnan(max_x_data_point); max_x_data_point = max(1, xl(2)); end 
    
    xlim(ax, [min_x_val, max_x_data_point * 1.1 + 1]);
    ylim(ax, [min(ytick_positions) - 0.5*inter_subject_gap, max(ytick_positions) + 0.5*inter_subject_gap]);
    
    % *** NEW: Reverse Y-axis so 'Overall' is at the top ***
    set(ax, 'YDir', 'reverse');
else 
    xlim(ax, [0 1]); ylim(ax, [0 1]);
end

% --- Adjust layout for legend ---
drawnow; pause(0.1);
if plotted_anything_for_legend && exist('lgd', 'var') && isvalid(lgd) && strcmp(lgd.Location, 'eastoutside')
    try
        original_ax_units = get(ax, 'Units'); original_lgd_units = get(lgd, 'Units');
        set(ax, 'Units', 'normalized'); set(lgd, 'Units', 'normalized');
        drawnow; pause(0.1);
        ax_pos_norm = get(ax, 'Position'); lgd_outer_pos_norm = get(lgd, 'OuterPosition');
        max_allowable_ax_width = lgd_outer_pos_norm(1) - ax_pos_norm(1) - 0.05; 
        if ax_pos_norm(3) > max_allowable_ax_width && max_allowable_ax_width > 0.05 
            set(ax, 'Position', [ax_pos_norm(1), ax_pos_norm(2), max_allowable_ax_width, ax_pos_norm(4)]);
        end
        set(ax, 'Units', original_ax_units); set(lgd, 'Units', original_lgd_units);
    catch ME_layout
        fprintf('Warning: Could not auto-adjust layout for legend: %s\n', ME_layout.message);
        if exist('original_ax_units','var'); set(ax, 'Units', original_ax_units); end
        if exist('original_lgd_units','var'); set(lgd, 'Units', original_lgd_units); end
    end
end
% --- Saving Figure and Data ---
if exist('fig', 'var') && isvalid(fig)
    figure(fig);
    base_filename = sprintf('cr_stim_at_%.0fpct_acc_horizontal', accuracy_threshold * 100); % Updated filename
    date_str = datestr(now, 'yyyymmdd_HHMMSS');
    full_base_filename = [base_filename '_' date_str];
    save_folder = 'CR_Figures_EstStimAtAcc';
    if ~exist(save_folder, 'dir')
        try; mkdir(save_folder);
        catch ME_mkdir
            fprintf('Error creating save dir ''%s'': %s. Saving to current dir.\n', save_folder, ME_mkdir.message);
            save_folder = '.';
        end
    end
    filepath_base = fullfile(save_folder, full_base_filename);
    
    try
        eps_filename = [filepath_base '.eps'];
        print(fig, eps_filename, '-depsc', '-painters');
        fprintf('Figure saved as: %s\n', eps_filename);
    catch ME_save_eps
        fprintf('Error saving figure as EPS: %s\n', ME_save_eps.message);
    end
    
    try
        png_filename = [filepath_base '.png']; 
        saveas(fig, png_filename);
        fprintf('Figure saved as: %s\n', png_filename);
    catch ME_save_png
        fprintf('Error saving figure as PNG: %s\n', ME_save_png.message);
    end
    
    if plotted_anything_for_legend
        data_to_save = {};
        
        if ~isnan(overall_mean_pooled)
            data_to_save(end+1, :) = {'Overall', overall_mean_pooled, overall_sem_pooled};
        end
        
        for i = 1:num_valid_subjects_plotted
            data_to_save(end+1, :) = {final_subject_initials_for_ytick{i}, actual_per_subject_metric_means(i), actual_per_subject_sems(i)};
        end
        
        try
            results_table = cell2table(data_to_save, 'VariableNames', {'Label', 'MeanEstimatedStimuli', 'SEM'});
            csv_filename = [filepath_base '.csv'];
            writetable(results_table, csv_filename);
            fprintf('Data saved as: %s\n', csv_filename);
        catch ME_save_csv
            fprintf('Error saving data as CSV: %s\n', ME_save_csv.message);
        end
    end
else
    fprintf('Figure handle not valid/created. Figure not saved.\n');
end
hold(ax, 'off');
% End of function
end

function plot_cr_avg_trials_to_error(metrics_mt)
% plot_cr_avg_trials_to_error Plots average trials to error for CR task horizontally.
%   - "Overall" average is from ALL SESSIONS pooled, shown at the top.
%   - Subject labels are initials on the Y-axis.
%   - Figure size 600x400, font size 14, legend on right, no box, no grid.
%   - Y-axis title is "Subject".
%   - Saves figure as PNG, EPS, and saves data as CSV.
%   Assumes metrics_mt{iS}(iD).AvgTrialsToError is [mean, median, SE, count]
%   for that session, and we use AvgTrialsToError(1) for session mean.

% --- Hardcoded Subject Information and Colors ---
subject_id_list_original = {
    'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan'
};
custom_colors = {
    [245, 124, 110]/255; [242, 181, 110]/255; [251, 231, 158]/255; % Colors 1-3
    [132, 195, 183]/255; [136, 215, 218]/255; [113, 184, 237]/255; % Colors 4-6
    [184, 174, 234]/255; [242, 168, 218]/255 % Colors 7-8
};
overall_color = [0 0 0]; % Black for overall

% --- Font Size & Plotting Constants ---
base_font_size = 14;
y_pos_overall = 1; % Overall point fixed at y=1
gap_after_overall = 1.2; % Increased gap between Overall and first subject
inter_subject_gap = 0.8; % Gap between subjects

% --- Input Validation ---
if nargin < 1
    error('Usage: plot_cr_avg_trials_to_error(metrics_mt)');
end
if ~iscell(metrics_mt)
    error('metrics_mt must be a cell array.');
end

% --- Figure Setup ---
figure_width = 500;
figure_height = 200;
fig = figure('Position', [100, 100, figure_width, figure_height]);
ax = gca;
hold(ax, 'on');

num_subjects_in_data = length(metrics_mt);
subject_means_all_subjects = NaN(1, num_subjects_in_data);
subject_sems_all_subjects = NaN(1, num_subjects_in_data);
subject_initial_labels_all = cell(1, num_subjects_in_data);
valid_subject_mask = false(1, num_subjects_in_data);
all_sessions_mean_tte_pooled = [];

% --- Data Extraction and Subject-Level Calculation ---
for iS = 1:num_subjects_in_data
    session_mean_tte_list_for_this_subject = [];
    if iS > length(metrics_mt) || isempty(metrics_mt{iS})
        continue;
    end
    num_sessions_for_subject = length(metrics_mt{iS});
    for iD = 1:num_sessions_for_subject
        if ~isstruct(metrics_mt{iS}(iD)); continue; end
        session_data = metrics_mt{iS}(iD);
        if isfield(session_data, 'AvgTrialsToError') && ...
           ~isempty(session_data.AvgTrialsToError) && ...
           isnumeric(session_data.AvgTrialsToError) && ...
           length(session_data.AvgTrialsToError) >= 1
            
            session_mean = session_data.AvgTrialsToError(1);
            if ~isnan(session_mean)
                session_mean_tte_list_for_this_subject = [session_mean_tte_list_for_this_subject, session_mean];
                all_sessions_mean_tte_pooled = [all_sessions_mean_tte_pooled, session_mean];
            end
        end
    end
    if ~isempty(session_mean_tte_list_for_this_subject)
        subject_means_all_subjects(iS) = mean(session_mean_tte_list_for_this_subject, 'omitnan');
        if length(session_mean_tte_list_for_this_subject) > 1
            subject_sems_all_subjects(iS) = std(session_mean_tte_list_for_this_subject, 'omitnan') / sqrt(length(session_mean_tte_list_for_this_subject));
        else
            subject_sems_all_subjects(iS) = 0; 
        end
        valid_subject_mask(iS) = true;
        if iS <= length(subject_id_list_original) && ~isempty(subject_id_list_original{iS})
            original_id = subject_id_list_original{iS};
            subject_initial_labels_all{iS} = sprintf('%c', upper(original_id(1)));
        else
            subject_initial_labels_all{iS} = sprintf('S%d', iS); 
        end
    end
end

% --- Filter for valid subjects and prepare for plotting ---
actual_per_subject_means = subject_means_all_subjects(valid_subject_mask);
actual_per_subject_sems = subject_sems_all_subjects(valid_subject_mask);
final_subject_initials_for_ytick = subject_initial_labels_all(valid_subject_mask);
num_valid_subjects_plotted = length(actual_per_subject_means);
legend_handles = [];
plotted_anything_for_legend = false;
ytick_positions = [];
ytick_labels_list = {};

% --- Overall Average Calculation and Plotting ---
overall_mean_pooled = NaN;
overall_sem_pooled = NaN;
if ~isempty(all_sessions_mean_tte_pooled)
    overall_mean_pooled = mean(all_sessions_mean_tte_pooled, 'omitnan');
    non_nan_pooled_sessions_count = sum(~isnan(all_sessions_mean_tte_pooled));
    if non_nan_pooled_sessions_count > 1
        overall_sem_pooled = std(all_sessions_mean_tte_pooled, 'omitnan') / sqrt(non_nan_pooled_sessions_count);
    else
        overall_sem_pooled = 0;
    end
    
    % *** MODIFIED: Swapped x/y, plotting horizontally ***
    h_overall = errorbar(ax, overall_mean_pooled, y_pos_overall, overall_sem_pooled, 'horizontal', 'o', ...
        'MarkerSize', 3, 'MarkerEdgeColor', overall_color, 'MarkerFaceColor', overall_color, ...
        'Color', overall_color, 'LineWidth',1, 'CapSize', 10, 'DisplayName', 'Overall');
    legend_handles(end+1) = h_overall;
    plotted_anything_for_legend = true;
    
    ytick_positions(end+1) = y_pos_overall;
    ytick_labels_list{end+1} = 'Overall';
end

% --- Plotting Subject Data ---
subject_y_positions = [];
if num_valid_subjects_plotted > 0
    current_y = y_pos_overall + gap_after_overall;
    for k_subj = 1:num_valid_subjects_plotted
        subject_y_positions(k_subj) = current_y + (k_subj-1)*inter_subject_gap;
    end
end
original_indices_of_valid_subjects = find(valid_subject_mask);
for k = 1:num_valid_subjects_plotted
    iS_original_index = original_indices_of_valid_subjects(k);
    
    current_subject_initial_for_plot = final_subject_initials_for_ytick{k};
    
    if iS_original_index <= length(custom_colors)
        current_color = custom_colors{iS_original_index};
    else
        current_color = rand(1,3);
    end
    
    y_pos = subject_y_positions(k);
    
    % *** MODIFIED: Swapped x/y, plotting horizontally ***
    h_subj = errorbar(ax, actual_per_subject_means(k), y_pos, actual_per_subject_sems(k), 'horizontal', 'o', ...
        'MarkerSize', 3, 'MarkerEdgeColor', current_color, 'MarkerFaceColor', current_color, ...
        'Color', current_color, 'LineWidth', 1, 'CapSize', 10, ...
        'DisplayName', current_subject_initial_for_plot);
    legend_handles(end+1) = h_subj;
    plotted_anything_for_legend = true; 
    ytick_positions(end+1) = y_pos;
    ytick_labels_list{end+1} = current_subject_initial_for_plot;
end

% --- Finalize Plot Styling ---
set(ax, 'FontSize', base_font_size);
if plotted_anything_for_legend
    lgd = legend(ax, legend_handles, 'Location', 'eastoutside', 'FontSize', base_font_size);
    lgd.Box = 'off';
    
    % *** MODIFIED: Changed from XTick to YTick ***
    if ~isempty(ytick_positions)
        set(ax, 'YTick', ytick_positions);
        set(ax, 'YTickLabel', ytick_labels_list);
    else
        set(ax, 'YTick', []); 
        set(ax, 'YTickLabel', {});
    end
else
    text(ax, 0.5, 0.5, 'No data available to plot.', ...
         'HorizontalAlignment', 'center', 'Units', 'normalized', 'FontSize', base_font_size, 'Color', 'red');
    warning('No data was plotted. Check input `metrics_mt` and ensure .AvgTrialsToError field is present and valid.');
    set(ax, 'YTick', []);
    set(ax, 'YTickLabel', {});
end

% *** MODIFIED: Swapped axis labels ***
ylabel(ax, 'Subject', 'FontSize', base_font_size);
xlabel(ax, 'Average Trials to Error', 'FontSize', base_font_size);
title(ax, 'CR: Avg. Trials to Error', 'FontSize', base_font_size, 'FontWeight', 'bold');
grid(ax, 'off'); 

% *** MODIFIED: Axis limit logic is now swapped for x and y ***
if ~isempty(ytick_positions)
    axis(ax, 'tight'); 
    xl = xlim(ax);
    min_x_val = 2; % This now applies to the X-axis
    
    max_x_data_point = -Inf;
    if ~isnan(overall_mean_pooled) && ~isnan(overall_sem_pooled)
        max_x_data_point = max(max_x_data_point, overall_mean_pooled + overall_sem_pooled);
    end
    if ~isempty(actual_per_subject_means)
        valid_means = actual_per_subject_means(~isnan(actual_per_subject_means) & ~isnan(actual_per_subject_sems));
        valid_sems = actual_per_subject_sems(~isnan(actual_per_subject_means) & ~isnan(actual_per_subject_sems));
        if ~isempty(valid_means)
             max_x_data_point = max(max_x_data_point, max(valid_means + valid_sems));
        end
    end
    if isinf(max_x_data_point) || isnan(max_x_data_point); max_x_data_point = max(1, xl(2)); end 
    
    xlim(ax, [min_x_val, max_x_data_point * 1.1 + eps]);
    ylim(ax, [min(ytick_positions) - 0.75*inter_subject_gap, max(ytick_positions) + 0.5*inter_subject_gap]);
    
    % *** NEW: Reverse Y-axis so 'Overall' is at the top ***
    set(ax, 'YDir', 'reverse');
else 
    xlim(ax, [0 1]); 
    ylim(ax, [0 1]);
end

% --- Adjust layout for legend ---
drawnow;
pause(0.1);
if plotted_anything_for_legend && exist('lgd', 'var') && isvalid(lgd)
    try
        if strcmp(lgd.Location, 'eastoutside')
            original_ax_units = get(ax, 'Units');
            original_lgd_units = get(lgd, 'Units');
            set(ax, 'Units', 'normalized');
            set(lgd, 'Units', 'normalized');
            drawnow; pause(0.1);
            ax_pos_norm = get(ax, 'Position');
            lgd_outer_pos_norm = get(lgd, 'OuterPosition');
            max_allowable_ax_width = lgd_outer_pos_norm(1) - ax_pos_norm(1) - 0.05; 
            if ax_pos_norm(3) > max_allowable_ax_width && max_allowable_ax_width > 0.05 
                set(ax, 'Position', [ax_pos_norm(1), ax_pos_norm(2), max_allowable_ax_width, ax_pos_norm(4)]);
            end
            set(ax, 'Units', original_ax_units);
            set(lgd, 'Units', original_lgd_units);
        end
    catch ME_layout
        fprintf('Warning: Could not auto-adjust layout for legend: %s\n', ME_layout.message);
        if exist('original_ax_units','var'); set(ax, 'Units', original_ax_units); end
        if exist('original_lgd_units','var'); set(lgd, 'Units', original_lgd_units); end
    end
end

% --- Saving the Figure and Data ---
if exist('fig', 'var') && isvalid(fig)
    figure(fig);
    base_filename = 'cr_avg_trials_to_error_horizontal'; % Appended horizontal to filename
    date_str = datestr(now, 'yyyymmdd_HHMMSS');
    full_base_filename = [base_filename '_' date_str];
    save_folder = 'CR_Figures_TrialsToError';
    if ~exist(save_folder, 'dir')
        try mkdir(save_folder);
        catch ME_mkdir
            fprintf('Error creating save directory ''%s'': %s. Saving to current directory.\n', save_folder, ME_mkdir.message);
            save_folder = '.';
        end
    end
    filepath_base = fullfile(save_folder, full_base_filename);
    
    try
        eps_filename = [filepath_base '.eps'];
        print(fig, eps_filename, '-depsc', '-painters'); 
        fprintf('Figure saved as: %s\n', eps_filename);
    catch ME_save_eps
        fprintf('Error saving figure as EPS: %s\n', ME_save_eps.message);
    end
    
    try
        png_filename = [filepath_base '.png'];
        saveas(fig, png_filename);
        fprintf('Figure saved as: %s\n', png_filename);
    catch ME_save_png
        fprintf('Error saving figure as PNG: %s\n', ME_save_png.message);
    end
    
    if plotted_anything_for_legend
        data_to_save = {};
        
        if ~isnan(overall_mean_pooled)
            data_to_save(end+1, :) = {'Overall', overall_mean_pooled, overall_sem_pooled};
        end
        
        for i = 1:num_valid_subjects_plotted
            data_to_save(end+1, :) = {final_subject_initials_for_ytick{i}, actual_per_subject_means(i), actual_per_subject_sems(i)};
        end
        
        try
            results_table = cell2table(data_to_save, 'VariableNames', {'Label', 'MeanTrialsToError', 'SEM'});
            csv_filename = [filepath_base '.csv'];
            writetable(results_table, csv_filename);
            fprintf('Data saved as: %s\n', csv_filename);
        catch ME_save_csv
            fprintf('Error saving data as CSV: %s\n', ME_save_csv.message);
        end
    end
else
    fprintf('Figure handle not valid or figure not created. Figure not saved.\n');
end
hold(ax, 'off');
% End of function
end



function plot_cr_rt_intercept_by_subject(metrics_mt)
% plot_cr_rt_intercept_by_subject Calculates and plots the average RT intercept by subject.
%   - For each session, fits a line to RT vs. Stimuli (2-20) to get an intercept.
%   - Plots the average of these session intercepts for each subject.
%   - "Overall" average is from ALL SESSIONS pooled, shown at the top.
%   - Subject labels are initials on the Y-axis.
%   - Figure size 600x400, font size 14, legend on right, no box, no grid.
%   - Saves figure as PNG, EPS, and saves data as CSV.
%
% INPUT:
%   metrics_mt - Cell array (iS indexes subjects).
%                metrics_mt{iS}(iD) is a structure for one session,
%                containing .RT_NumStim for the CR task.

% --- Hardcoded Subject Information and Colors ---
subject_id_list_original = {
    'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan'
};
custom_colors = {
    [245, 124, 110]/255; [242, 181, 110]/255; [251, 231, 158]/255; % Colors 1-3
    [132, 195, 183]/255; [136, 215, 218]/255; [113, 184, 237]/255; % Colors 4-6
    [184, 174, 234]/255; [242, 168, 218]/255 % Colors 7-8
};
overall_color = [0 0 0]; % Black for overall

% --- Font Size & Plotting Constants ---
base_font_size = 14;
y_pos_overall = 1; 
gap_after_overall = 1.2; 
inter_subject_gap = 0.8;

% --- Expected Input Data Format ---
expected_input_num_rows = 19; % For stimuli 2-20
expected_input_num_cols = 4;

% --- Input Validation ---
if nargin < 1
    error('Usage: plot_cr_rt_intercept_by_subject(metrics_mt)');
end
if ~iscell(metrics_mt)
    error('metrics_mt must be a cell array.');
end

% --- Figure Setup ---
figure_width = 500;
figure_height = 200;
fig = figure('Position', [100, 100, figure_width, figure_height]);
ax = gca;
hold(ax, 'on');

num_subjects_in_data = length(metrics_mt);
subject_mean_intercepts = NaN(1, num_subjects_in_data);
subject_sem_intercepts = NaN(1, num_subjects_in_data);
subject_initial_labels_all = cell(1, num_subjects_in_data);
valid_subject_mask = false(1, num_subjects_in_data);
all_sessions_intercept_pooled = []; 

% --- Data Extraction and Intercept Calculation ---
for iS = 1:num_subjects_in_data
    session_intercept_list_for_this_subject = [];
    if iS > length(metrics_mt) || isempty(metrics_mt{iS})
        continue;
    end
    num_sessions_for_subject = length(metrics_mt{iS});
    for iD = 1:num_sessions_for_subject
        if ~isstruct(metrics_mt{iS}(iD)); continue; end
        session_data = metrics_mt{iS}(iD);
        
        if isfield(session_data, 'RT_NumStim') && ~isempty(session_data.RT_NumStim)
            rt_stim_data = session_data.RT_NumStim;
            if size(rt_stim_data, 1) == expected_input_num_rows && size(rt_stim_data, 2) >= 1
                
                mean_rt_2_to_20 = rt_stim_data(:, 1);
                stim_counts = (2:20)'; % X-values for the fit
                
                % Remove NaN values for a clean linear fit
                valid_idx = ~isnan(mean_rt_2_to_20);
                if sum(valid_idx) >= 2 % Need at least 2 points to fit a line
                    x_fit = stim_counts(valid_idx);
                    y_fit = mean_rt_2_to_20(valid_idx);
                    
                    % Fit a 1st degree polynomial (linear line)
                    coeffs = polyfit(x_fit, y_fit, 1);
                    session_intercept = coeffs(2); % The second coefficient is the intercept
                    
                    session_intercept_list_for_this_subject(end+1) = session_intercept;
                    all_sessions_intercept_pooled(end+1) = session_intercept;
                end
            end
        end
    end
    
    % Aggregate the intercepts for the subject
    if ~isempty(session_intercept_list_for_this_subject)
        subject_mean_intercepts(iS) = mean(session_intercept_list_for_this_subject, 'omitnan');
        num_valid_sessions_for_subj = sum(~isnan(session_intercept_list_for_this_subject));
        if num_valid_sessions_for_subj > 1
            subject_sem_intercepts(iS) = std(session_intercept_list_for_this_subject, 'omitnan') / sqrt(num_valid_sessions_for_subj);
        else
            subject_sem_intercepts(iS) = 0; 
        end
        valid_subject_mask(iS) = true;
        if iS <= length(subject_id_list_original) && ~isempty(subject_id_list_original{iS})
            original_id = subject_id_list_original{iS};
            subject_initial_labels_all{iS} = sprintf('%c', upper(original_id(1)));
        else
            subject_initial_labels_all{iS} = sprintf('S%d', iS); 
        end
    end
end

% --- Filter for valid subjects and prepare for plotting ---
actual_per_subject_means = subject_mean_intercepts(valid_subject_mask);
actual_per_subject_sems = subject_sem_intercepts(valid_subject_mask);
final_subject_initials_for_ytick = subject_initial_labels_all(valid_subject_mask);
num_valid_subjects_plotted = length(actual_per_subject_means);
legend_handles = [];
plotted_anything_for_legend = false;
ytick_positions = [];
ytick_labels_list = {};

% --- Overall Average Calculation and Plotting ---
overall_mean_pooled = NaN;
overall_sem_pooled = NaN;
if ~isempty(all_sessions_intercept_pooled)
    overall_mean_pooled = mean(all_sessions_intercept_pooled, 'omitnan');
    non_nan_pooled_sessions_count = sum(~isnan(all_sessions_intercept_pooled));
    if non_nan_pooled_sessions_count > 1
        overall_sem_pooled = std(all_sessions_intercept_pooled, 'omitnan') / sqrt(non_nan_pooled_sessions_count);
    else
        overall_sem_pooled = 0;
    end
    
    h_overall = errorbar(ax, overall_mean_pooled, y_pos_overall, overall_sem_pooled, 'horizontal', 'o', ...
        'MarkerSize', 5, 'MarkerEdgeColor', overall_color, 'MarkerFaceColor', overall_color, ...
        'Color', overall_color, 'LineWidth',1, 'CapSize', 10, 'DisplayName', 'Overall');
    legend_handles(end+1) = h_overall;
    plotted_anything_for_legend = true;
    
    ytick_positions(end+1) = y_pos_overall;
    ytick_labels_list{end+1} = 'Overall';
end

% --- Plotting Subject Data ---
subject_y_positions = [];
if num_valid_subjects_plotted > 0
    current_y = y_pos_overall + gap_after_overall;
    for k_subj = 1:num_valid_subjects_plotted
        subject_y_positions(k_subj) = current_y + (k_subj-1)*inter_subject_gap;
    end
end
original_indices_of_valid_subjects = find(valid_subject_mask);
for k = 1:num_valid_subjects_plotted
    iS_original_index = original_indices_of_valid_subjects(k);
    current_subject_initial_for_plot = final_subject_initials_for_ytick{k};
    if iS_original_index <= length(custom_colors)
        current_color = custom_colors{iS_original_index};
    else
        current_color = rand(1,3);
    end
    
    y_pos = subject_y_positions(k);
    
    h_subj = errorbar(ax, actual_per_subject_means(k), y_pos, actual_per_subject_sems(k), 'horizontal', 'o', ...
        'MarkerSize', 5, 'MarkerEdgeColor', current_color, 'MarkerFaceColor', current_color, ...
        'Color', current_color, 'LineWidth', 1, 'CapSize', 10, ...
        'DisplayName', current_subject_initial_for_plot);
    legend_handles(end+1) = h_subj;
    plotted_anything_for_legend = true; 
    ytick_positions(end+1) = y_pos;
    ytick_labels_list{end+1} = current_subject_initial_for_plot;
end

% --- Finalize Plot Styling ---
set(ax, 'FontSize', base_font_size);
if plotted_anything_for_legend
    lgd = legend(ax, legend_handles, 'Location', 'eastoutside', 'FontSize', base_font_size);
    lgd.Box = 'off';
    
    if ~isempty(ytick_positions)
        set(ax, 'YTick', ytick_positions);
        set(ax, 'YTickLabel', ytick_labels_list);
    else
        set(ax, 'YTick', []); 
        set(ax, 'YTickLabel', {});
    end
else
    text(ax, 0.5, 0.5, 'No data available to plot.', ...
         'HorizontalAlignment', 'center', 'Units', 'normalized', 'FontSize', base_font_size, 'Color', 'red');
    warning('No data was plotted. Check input `metrics_mt` and ensure .RT_NumStim field is present and valid.');
    set(ax, 'YTick', []);
    set(ax, 'YTickLabel', {});
end
ylabel(ax, 'Subject', 'FontSize', base_font_size);
xlabel(ax, 'RT Intercept (s)', 'FontSize', base_font_size);
title(ax, 'CR: Average Intercept of Reaction Time', 'FontSize', base_font_size, 'FontWeight', 'bold');
grid(ax, 'off'); 
if ~isempty(ytick_positions)
    axis(ax, 'tight'); 
    xlim(ax, [0.6, max(xlim)*1.1]); 
    ylim(ax, [min(ytick_positions) - 0.75*inter_subject_gap, max(ytick_positions) + 0.5*inter_subject_gap]);
    set(ax, 'YDir', 'reverse');
else 
    xlim(ax, [0 1]); 
    ylim(ax, [0 1]);
end

% --- Adjust layout for legend ---
drawnow;
pause(0.1);
if plotted_anything_for_legend && exist('lgd', 'var') && isvalid(lgd)
    try
        if strcmp(lgd.Location, 'eastoutside')
            original_ax_units = get(ax, 'Units');
            original_lgd_units = get(lgd, 'Units');
            set(ax, 'Units', 'normalized');
            set(lgd, 'Units', 'normalized');
            drawnow; pause(0.1);
            ax_pos_norm = get(ax, 'Position');
            lgd_outer_pos_norm = get(lgd, 'OuterPosition');
            max_allowable_ax_width = lgd_outer_pos_norm(1) - ax_pos_norm(1) - 0.05; 
            if ax_pos_norm(3) > max_allowable_ax_width && max_allowable_ax_width > 0.05 
                set(ax, 'Position', [ax_pos_norm(1), ax_pos_norm(2), max_allowable_ax_width, ax_pos_norm(4)]);
            end
            set(ax, 'Units', original_ax_units);
            set(lgd, 'Units', original_lgd_units);
        end
    catch ME_layout
        fprintf('Warning: Could not auto-adjust layout for legend: %s\n', ME_layout.message);
        if exist('original_ax_units','var'); set(ax, 'Units', original_ax_units); end
        if exist('original_lgd_units','var'); set(lgd, 'Units', original_lgd_units); end
    end
end

% --- Saving the Figure and Data ---
if exist('fig', 'var') && isvalid(fig)
    figure(fig);
    base_filename = 'cr_rt_intercept_by_subject';
    date_str = datestr(now, 'yyyymmdd_HHMMSS');
    full_base_filename = [base_filename '_' date_str];
    save_folder = 'CR_Figures_RT_Intercept';
    if ~exist(save_folder, 'dir')
        try mkdir(save_folder);
        catch ME_mkdir
            fprintf('Error creating save directory ''%s'': %s. Saving to current directory.\n', save_folder, ME_mkdir.message);
            save_folder = '.';
        end
    end
    filepath_base = fullfile(save_folder, full_base_filename);
    
    try
        eps_filename = [filepath_base '.eps'];
        print(fig, eps_filename, '-depsc', '-painters');
        fprintf('Figure saved as: %s\n', eps_filename);
    catch ME_save_eps
        fprintf('Error saving figure as EPS: %s\n', ME_save_eps.message);
    end
    
    try
        png_filename = [filepath_base '.png'];
        saveas(fig, png_filename);
        fprintf('Figure saved as: %s\n', png_filename);
    catch ME_save_png
        fprintf('Error saving figure as PNG: %s\n', ME_save_png.message);
    end
    
    if plotted_anything_for_legend
        data_to_save = {};
        if ~isnan(overall_mean_pooled)
            data_to_save(end+1, :) = {'Overall', overall_mean_pooled, overall_sem_pooled};
        end
        for i = 1:num_valid_subjects_plotted
            data_to_save(end+1, :) = {final_subject_initials_for_ytick{i}, actual_per_subject_means(i), actual_per_subject_sems(i)};
        end
        
        try
            results_table = cell2table(data_to_save, 'VariableNames', {'Label', 'MeanRTIntercept', 'SEM'});
            csv_filename = [filepath_base '.csv'];
            writetable(results_table, csv_filename);
            fprintf('Data saved as: %s\n', csv_filename);
        catch ME_save_csv
            fprintf('Error saving data as CSV: %s\n', ME_save_csv.message);
        end
    end
else
    fprintf('Figure handle not valid or figure not created. Figure not saved.\n');
end

hold(ax, 'off');
% End of function
end



function plot_cr_rt_slope_by_subject(metrics_mt)
% plot_cr_rt_slope_by_subject Calculates and plots the average RT slope by subject.
%   - For each session, fits a line to RT vs. Stimuli (2-20) to get a slope.
%   - Plots the average of these session slopes for each subject.
%   - "Overall" average is from ALL SESSIONS pooled, shown at the top.
%   - Subject labels are initials on the Y-axis.
%   - Figure size 600x400, font size 14, legend on right, no box, no grid.
%   - Saves figure as PNG, EPS, and saves data as CSV.
%
% INPUT:
%   metrics_mt - Cell array (iS indexes subjects).
%                metrics_mt{iS}(iD) is a structure for one session,
%                containing .RT_NumStim for the CR task.

% --- Hardcoded Subject Information and Colors ---
subject_id_list_original = {
    'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan'
};
custom_colors = {
    [245, 124, 110]/255; [242, 181, 110]/255; [251, 231, 158]/255; % Colors 1-3
    [132, 195, 183]/255; [136, 215, 218]/255; [113, 184, 237]/255; % Colors 4-6
    [184, 174, 234]/255; [242, 168, 218]/255 % Colors 7-8
};
overall_color = [0 0 0]; % Black for overall

% --- Font Size & Plotting Constants ---
base_font_size = 14;
y_pos_overall = 1; 
gap_after_overall = 1.2; 
inter_subject_gap = 0.8;

% --- Expected Input Data Format ---
expected_input_num_rows = 19; % For stimuli 2-20
expected_input_num_cols = 4;

% --- Input Validation ---
if nargin < 1
    error('Usage: plot_cr_rt_slope_by_subject(metrics_mt)');
end
if ~iscell(metrics_mt)
    error('metrics_mt must be a cell array.');
end

% --- Figure Setup ---
figure_width = 500;
figure_height = 200;
fig = figure('Position', [100, 100, figure_width, figure_height]);
ax = gca;
hold(ax, 'on');

num_subjects_in_data = length(metrics_mt);
subject_mean_slopes = NaN(1, num_subjects_in_data);
subject_sem_slopes = NaN(1, num_subjects_in_data);
subject_initial_labels_all = cell(1, num_subjects_in_data);
valid_subject_mask = false(1, num_subjects_in_data);
all_sessions_slope_pooled = []; 

% --- Data Extraction and Slope Calculation ---
for iS = 1:num_subjects_in_data
    session_slope_list_for_this_subject = [];
    if iS > length(metrics_mt) || isempty(metrics_mt{iS})
        continue;
    end
    num_sessions_for_subject = length(metrics_mt{iS});
    for iD = 1:num_sessions_for_subject
        if ~isstruct(metrics_mt{iS}(iD)); continue; end
        session_data = metrics_mt{iS}(iD);
        
        if isfield(session_data, 'RT_NumStim') && ~isempty(session_data.RT_NumStim)
            rt_stim_data = session_data.RT_NumStim;
            if size(rt_stim_data, 1) == expected_input_num_rows && size(rt_stim_data, 2) >= 1
                
                mean_rt_2_to_20 = rt_stim_data(:, 1);
                stim_counts = (2:20)'; % X-values for the fit
                
                % Remove NaN values for a clean linear fit
                valid_idx = ~isnan(mean_rt_2_to_20);
                if sum(valid_idx) >= 2 % Need at least 2 points to fit a line
                    x_fit = stim_counts(valid_idx);
                    y_fit = mean_rt_2_to_20(valid_idx);
                    
                    % Fit a 1st degree polynomial (linear line)
                    coeffs = polyfit(x_fit, y_fit, 1);
                    session_slope = coeffs(1); % The first coefficient is the slope
                    
                    session_slope_list_for_this_subject(end+1) = session_slope;
                    all_sessions_slope_pooled(end+1) = session_slope;
                end
            end
        end
    end
    
    % Aggregate the slopes for the subject
    if ~isempty(session_slope_list_for_this_subject)
        subject_mean_slopes(iS) = mean(session_slope_list_for_this_subject, 'omitnan');
        num_valid_sessions_for_subj = sum(~isnan(session_slope_list_for_this_subject));
        if num_valid_sessions_for_subj > 1
            subject_sem_slopes(iS) = std(session_slope_list_for_this_subject, 'omitnan') / sqrt(num_valid_sessions_for_subj);
        else
            subject_sem_slopes(iS) = 0; 
        end
        valid_subject_mask(iS) = true;
        if iS <= length(subject_id_list_original) && ~isempty(subject_id_list_original{iS})
            original_id = subject_id_list_original{iS};
            subject_initial_labels_all{iS} = sprintf('%c', upper(original_id(1)));
        else
            subject_initial_labels_all{iS} = sprintf('S%d', iS); 
        end
    end
end

% --- Filter for valid subjects and prepare for plotting ---
actual_per_subject_means = subject_mean_slopes(valid_subject_mask);
actual_per_subject_sems = subject_sem_slopes(valid_subject_mask);
final_subject_initials_for_ytick = subject_initial_labels_all(valid_subject_mask);
num_valid_subjects_plotted = length(actual_per_subject_means);
legend_handles = [];
plotted_anything_for_legend = false;
ytick_positions = [];
ytick_labels_list = {};

% --- Overall Average Calculation and Plotting ---
overall_mean_pooled = NaN;
overall_sem_pooled = NaN;
if ~isempty(all_sessions_slope_pooled)
    overall_mean_pooled = mean(all_sessions_slope_pooled, 'omitnan');
    non_nan_pooled_sessions_count = sum(~isnan(all_sessions_slope_pooled));
    if non_nan_pooled_sessions_count > 1
        overall_sem_pooled = std(all_sessions_slope_pooled, 'omitnan') / sqrt(non_nan_pooled_sessions_count);
    else
        overall_sem_pooled = 0;
    end
    
    h_overall = errorbar(ax, overall_mean_pooled, y_pos_overall, overall_sem_pooled, 'horizontal', 'o', ...
        'MarkerSize', 5, 'MarkerEdgeColor', overall_color, 'MarkerFaceColor', overall_color, ...
        'Color', overall_color, 'LineWidth',1, 'CapSize', 10, 'DisplayName', 'Overall');
    legend_handles(end+1) = h_overall;
    plotted_anything_for_legend = true;
    
    ytick_positions(end+1) = y_pos_overall;
    ytick_labels_list{end+1} = 'Overall';
end

% --- Plotting Subject Data ---
subject_y_positions = [];
if num_valid_subjects_plotted > 0
    current_y = y_pos_overall + gap_after_overall;
    for k_subj = 1:num_valid_subjects_plotted
        subject_y_positions(k_subj) = current_y + (k_subj-1)*inter_subject_gap;
    end
end
original_indices_of_valid_subjects = find(valid_subject_mask);
for k = 1:num_valid_subjects_plotted
    iS_original_index = original_indices_of_valid_subjects(k);
    current_subject_initial_for_plot = final_subject_initials_for_ytick{k};
    if iS_original_index <= length(custom_colors)
        current_color = custom_colors{iS_original_index};
    else
        current_color = rand(1,3);
    end
    
    y_pos = subject_y_positions(k);
    
    h_subj = errorbar(ax, actual_per_subject_means(k), y_pos, actual_per_subject_sems(k), 'horizontal', 'o', ...
        'MarkerSize', 5, 'MarkerEdgeColor', current_color, 'MarkerFaceColor', current_color, ...
        'Color', current_color, 'LineWidth', 1, 'CapSize', 10, ...
        'DisplayName', current_subject_initial_for_plot);
    legend_handles(end+1) = h_subj;
    plotted_anything_for_legend = true; 
    ytick_positions(end+1) = y_pos;
    ytick_labels_list{end+1} = current_subject_initial_for_plot;
end

% --- Finalize Plot Styling ---
set(ax, 'FontSize', base_font_size);
if plotted_anything_for_legend
    lgd = legend(ax, legend_handles, 'Location', 'eastoutside', 'FontSize', base_font_size);
    lgd.Box = 'off';
    
    if ~isempty(ytick_positions)
        set(ax, 'YTick', ytick_positions);
        set(ax, 'YTickLabel', ytick_labels_list);
    else
        set(ax, 'YTick', []); 
        set(ax, 'YTickLabel', {});
    end
else
    text(ax, 0.5, 0.5, 'No data available to plot.', ...
         'HorizontalAlignment', 'center', 'Units', 'normalized', 'FontSize', base_font_size, 'Color', 'red');
    warning('No data was plotted. Check input `metrics_mt` and ensure .RT_NumStim field is present and valid.');
    set(ax, 'YTick', []);
    set(ax, 'YTickLabel', {});
end
ylabel(ax, 'Subject', 'FontSize', base_font_size);
xlabel(ax, 'RT Slope (s / stimulus)', 'FontSize', base_font_size);
title(ax, 'CR: Average Slope of Reaction Time', 'FontSize', base_font_size, 'FontWeight', 'bold');
grid(ax, 'off'); 
if ~isempty(ytick_positions)
    axis(ax, 'tight'); 
    xlim(ax, [0, max(xlim)*1.1]); % Start x-axis at 0 for slope
    ylim(ax, [min(ytick_positions) - 0.75*inter_subject_gap, max(ytick_positions) + 0.5*inter_subject_gap]);
    set(ax, 'YDir', 'reverse');
else 
    xlim(ax, [0 1]); 
    ylim(ax, [0 1]);
end

% --- Adjust layout for legend ---
drawnow;
pause(0.1);
if plotted_anything_for_legend && exist('lgd', 'var') && isvalid(lgd)
    try
        if strcmp(lgd.Location, 'eastoutside')
            original_ax_units = get(ax, 'Units');
            original_lgd_units = get(lgd, 'Units');
            set(ax, 'Units', 'normalized');
            set(lgd, 'Units', 'normalized');
            drawnow; pause(0.1);
            ax_pos_norm = get(ax, 'Position');
            lgd_outer_pos_norm = get(lgd, 'OuterPosition');
            max_allowable_ax_width = lgd_outer_pos_norm(1) - ax_pos_norm(1) - 0.05; 
            if ax_pos_norm(3) > max_allowable_ax_width && max_allowable_ax_width > 0.05 
                set(ax, 'Position', [ax_pos_norm(1), ax_pos_norm(2), max_allowable_ax_width, ax_pos_norm(4)]);
            end
            set(ax, 'Units', original_ax_units);
            set(lgd, 'Units', original_lgd_units);
        end
    catch ME_layout
        fprintf('Warning: Could not auto-adjust layout for legend: %s\n', ME_layout.message);
        if exist('original_ax_units','var'); set(ax, 'Units', original_ax_units); end
        if exist('original_lgd_units','var'); set(lgd, 'Units', original_lgd_units); end
    end
end

% --- Saving the Figure and Data ---
if exist('fig', 'var') && isvalid(fig)
    figure(fig);
    base_filename = 'cr_rt_slope_by_subject';
    date_str = datestr(now, 'yyyymmdd_HHMMSS');
    full_base_filename = [base_filename '_' date_str];
    save_folder = 'CR_Figures_RT_Slope';
    if ~exist(save_folder, 'dir')
        try mkdir(save_folder);
        catch ME_mkdir
            fprintf('Error creating save directory ''%s'': %s. Saving to current directory.\n', save_folder, ME_mkdir.message);
            save_folder = '.';
        end
    end
    filepath_base = fullfile(save_folder, full_base_filename);
    
    try
        eps_filename = [filepath_base '.eps'];
        print(fig, eps_filename, '-depsc', '-painters');
        fprintf('Figure saved as: %s\n', eps_filename);
    catch ME_save_eps
        fprintf('Error saving figure as EPS: %s\n', ME_save_eps.message);
    end
    
    try
        png_filename = [filepath_base '.png'];
        saveas(fig, png_filename);
        fprintf('Figure saved as: %s\n', png_filename);
    catch ME_save_png
        fprintf('Error saving figure as PNG: %s\n', ME_save_png.message);
    end
    
    if plotted_anything_for_legend
        data_to_save = {};
        if ~isnan(overall_mean_pooled)
            data_to_save(end+1, :) = {'Overall', overall_mean_pooled, overall_sem_pooled};
        end
        for i = 1:num_valid_subjects_plotted
            data_to_save(end+1, :) = {final_subject_initials_for_ytick{i}, actual_per_subject_means(i), actual_per_subject_sems(i)};
        end
        
        try
            results_table = cell2table(data_to_save, 'VariableNames', {'Label', 'MeanRTSlope', 'SEM'});
            csv_filename = [filepath_base '.csv'];
            writetable(results_table, csv_filename);
            fprintf('Data saved as: %s\n', csv_filename);
        catch ME_save_csv
            fprintf('Error saving data as CSV: %s\n', ME_save_csv.message);
        end
    end
else
    fprintf('Figure handle not valid or figure not created. Figure not saved.\n');
end

hold(ax, 'off');
% End of function
end


function plot_cr_rt_by_subject(metrics_mt)
% plot_cr_rt_by_subject Plots average reaction time for CR task with error bars.
%   - X-axis is 1-20 stimuli. RT for Stimulus Count 1 is NaN.
%   - Y-axis is Reaction Time in seconds.
%   - Font size for plot elements is 14.
%   - Figure size is 800x200.
%   - Legend is on the right, vertically centered, no box.
%   - Saves figure as PNG/EPS and exports data to CSV.
%   .RT_NumStim (input data) is expected to be 19x4 (for stimuli 2-20):
%     Col 1: Mean Reaction Time (for stimuli 2-20)
%     Col 2: Median Reaction Time
%     Col 3: Standard Error (SE) of RT
%     Col 4: Count
%
% INPUT:
%   metrics_mt - Cell array (iS indexes subjects).
%                metrics_mt{iS}(iD) is a structure for one session,
%                containing .RT_NumStim for the CR task.

% --- Hardcoded Subject Information and Colors ---
subject_id_list = {
    'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan'
};
custom_colors = {
[241, 88, 70]/255; [255, 176, 80]/255; [251, 231, 158]/255; 
    [136, 215, 218]/255; [87, 169, 230]/255; [107, 114, 182]/255 
};

% --- X-axis for Plotting ---
x_axis_for_plot = (1:20)'; % X-axis from 1 to 20 stimuli

% --- Expected Input Data Format for RT_NumStim (for stimuli 2-20) ---
expected_input_num_rows = 19; % Corresponds to stimuli 2-20
expected_input_num_cols = 4;  % Assumed [mean, median, SE, count]

% --- Font Size ---
base_font_size = 14;

% --- Input Validation ---
if nargin < 1
    error('Usage: plot_cr_rt_by_subject(metrics_mt)');
end
if ~iscell(metrics_mt)
    error('metrics_mt must be a cell array.');
end

% --- Figure Setup ---
figure_width = 300;
figure_height = 200;
fig = figure('Position', [100, 100, figure_width, figure_height]);
ax = gca;
hold(ax, 'on');

legend_handles = [];
plotted_anything_for_legend = false;
num_subjects_in_data = length(metrics_mt);
all_data_for_csv = table(); % *** NEW: Initialize table for CSV export

% --- Main Loop for Subjects ---
for iS = 1:num_subjects_in_data
    session_rts_for_subject_1_to_20 = {}; % To collect augmented 20-element RT vectors
    
    if iS > length(metrics_mt) || isempty(metrics_mt{iS})
        continue;
    end
    
    num_sessions_for_subject = length(metrics_mt{iS});
    valid_sessions_count = 0;
    for iD = 1:num_sessions_for_subject
        if ~isstruct(metrics_mt{iS}(iD))
            continue;
        end
        session_data = metrics_mt{iS}(iD);
        
        subject_name_for_warning = 'Unknown';
        if iS <= length(subject_id_list); subject_name_for_warning = subject_id_list{iS};
        else; subject_name_for_warning = sprintf('Subject %d', iS); end
        
        if isfield(session_data, 'RT_NumStim') && ~isempty(session_data.RT_NumStim)
            rt_stim_data = session_data.RT_NumStim;
            if size(rt_stim_data, 1) == expected_input_num_rows && size(rt_stim_data, 2) == expected_input_num_cols
                mean_rt_2_to_20 = rt_stim_data(:, 1); 
                full_session_rt_1_to_20 = [NaN; mean_rt_2_to_20];
                session_rts_for_subject_1_to_20{end+1} = full_session_rt_1_to_20;
                valid_sessions_count = valid_sessions_count + 1;
            else
                warning('Subject %s (Index %d), Session %d: RT_NumStim dimensions (%dx%d) do not match expected format %dx%d. Skipping.', ...
                        subject_name_for_warning, iS, iD, size(rt_stim_data,1), size(rt_stim_data,2), expected_input_num_rows, expected_input_num_cols);
            end
        end
    end
    
    if valid_sessions_count > 0
        all_sessions_matrix = cat(2, session_rts_for_subject_1_to_20{:});
        
        % Calculate average RT across sessions
        subject_avg_rt = nanmean(all_sessions_matrix, 2); 
        
        % *** NEW: Calculate Standard Error of the Mean (SEM) across sessions ***
        if valid_sessions_count > 1
            subject_std_dev = nanstd(all_sessions_matrix, 0, 2);
            subject_sem = subject_std_dev / sqrt(valid_sessions_count);
        else
            subject_sem = zeros(size(subject_avg_rt)); % SEM is 0 for a single session
        end
        
        if iS <= length(subject_id_list); current_subject_label = subject_id_list{iS};
        else; current_subject_label = sprintf('Subject %d', iS); end
        
        if iS <= length(custom_colors); current_color = custom_colors{iS};
        else
            current_color = rand(1,3);
            if iS > length(subject_id_list)
                 warning('Not enough custom colors or subject IDs. Using random color for %s.', current_subject_label);
            end
        end
        
        % *** MODIFIED: Plot with error bars ***
        h = errorbar(ax, x_axis_for_plot, subject_avg_rt, subject_sem, ...
                 'LineWidth', 1.5, 'Color', current_color, 'Marker', 'o', ...
                 'MarkerSize', 4, 'DisplayName', current_subject_label, 'CapSize', 3);
                 
        legend_handles(end+1) = h;
        plotted_anything_for_legend = true;
        
        % *** NEW: Collect data for CSV export ***
        subject_label_col = repmat({current_subject_label}, length(x_axis_for_plot), 1);
        temp_table = table(subject_label_col, x_axis_for_plot, subject_avg_rt, subject_sem, ...
            'VariableNames', {'Subject', 'StimulusCount', 'MeanRT', 'SEM'});
        all_data_for_csv = [all_data_for_csv; temp_table];
    end
end

% --- Finalize Plot Styling ---
set(ax, 'FontSize', base_font_size);
if plotted_anything_for_legend
    lgd = legend(ax, legend_handles, 'Location', 'eastoutside', 'FontSize', base_font_size);
    lgd.Box = 'off';
else
    text(ax, 0.5, 0.5, 'No data available to plot based on current assumptions.', ...
         'HorizontalAlignment', 'center', 'Units', 'normalized', 'FontSize', base_font_size, 'Color', 'red');
    warning('No data was plotted. Check input `metrics_mt` and ensure .RT_NumStim field matches expected input format (%dx%d for stimuli 2-20).', expected_input_num_rows, expected_input_num_cols);
end
xlabel(ax, 'Number of Stimuli', 'FontSize', base_font_size);
ylabel(ax, 'Average Reaction Time (s)', 'FontSize', base_font_size);
title(ax, 'CR Task: Average RT vs. Number of Stimuli by Subject', 'FontSize', base_font_size);
grid(ax, 'off');
axis(ax, 'tight');
xlim(ax, [min(x_axis_for_plot)-0.5 max(x_axis_for_plot)+0.5]);

% --- Adjust layout for legend ---
drawnow;
pause(0.1);
if plotted_anything_for_legend && exist('lgd', 'var') && isvalid(lgd)
    try
        if strcmp(lgd.Location, 'eastoutside')
            original_ax_units = get(ax, 'Units');
            original_lgd_units = get(lgd, 'Units');
            set(ax, 'Units', 'normalized');
            set(lgd, 'Units', 'normalized');
            drawnow; pause(0.1);
            ax_pos_norm = get(ax, 'Position');
            lgd_outer_pos_norm = get(lgd, 'OuterPosition');
            max_allowable_ax_width = lgd_outer_pos_norm(1) - ax_pos_norm(1) - 0.03;
            if ax_pos_norm(3) > max_allowable_ax_width && max_allowable_ax_width > 0.1
                set(ax, 'Position', [ax_pos_norm(1), ax_pos_norm(2), max_allowable_ax_width, ax_pos_norm(4)]);
            end
            set(ax, 'Units', original_ax_units);
            set(lgd, 'Units', original_lgd_units);
        end
    catch ME_layout
        fprintf('Warning: Could not auto-adjust layout for legend: %s\n', ME_layout.message);
        if exist('original_ax_units','var'); set(ax, 'Units', original_ax_units); end
        if exist('original_lgd_units','var'); set(lgd, 'Units', original_lgd_units); end
    end
end

% --- Saving Figure and Data ---
if exist('fig', 'var') && isvalid(fig)
    figure(fig);
    base_filename = 'cr_rt_vs_stimuli_by_subject'; 
    date_str = datestr(now, 'yyyymmdd_HHMMSS');
    full_base_filename = [base_filename '_' date_str];
    save_folder = 'CR_Figures_RT';
    if ~exist(save_folder, 'dir')
        try
            mkdir(save_folder);
        catch ME_mkdir
            fprintf('Error creating save directory ''%s'': %s. Saving to current directory.\n', save_folder, ME_mkdir.message);
            save_folder = '.';
        end
    end
    filepath_base = fullfile(save_folder, full_base_filename);
    
    % *** NEW: Save as EPS (vector format) ***
    try
        eps_filename = [filepath_base '.eps'];
        print(fig, eps_filename, '-depsc', '-painters');
        fprintf('Figure saved as: %s\n', eps_filename);
    catch ME_save_eps
        fprintf('Error saving figure as EPS: %s\n', ME_save_eps.message);
    end

    % Save as PNG
    try
        png_filename = [filepath_base '.png'];
        saveas(fig, png_filename);
        fprintf('Figure saved as: %s\n', png_filename);
    catch ME_save_png
        fprintf('Error saving figure as PNG: %s\n', ME_save_png.message);
    end

    % *** NEW: Save data to CSV ***
    if ~isempty(all_data_for_csv)
        try
            csv_filename = [filepath_base '_data.csv'];
            writetable(all_data_for_csv, csv_filename);
            fprintf('Data saved as: %s\n', csv_filename);
        catch ME_save_csv
            fprintf('Error saving data as CSV: %s\n', ME_save_csv.message);
        end
    end
else
    fprintf('Figure handle not valid or figure not created. Figure and data not saved.\n');
end
hold(ax, 'off');
% End of function
end


function plot_cr_accuracy_by_subject(metrics_mt)
% plot_cr_accuracy_by_subject Plots average accuracy for CR task with error bars.
%   - Error bars represent the Standard Error of the Mean (SEM) calculated across sessions.
%   - X-axis is 1-20. Accuracy for Stimulus Count 1 is fixed at 1.
%   - Font size for plot elements is 14.
%   - Figure size is 800x200.
%   - Legend is on the right, vertically centered, no box.
%   .Acc_NumStim (input data) is expected to be 19x4 (for stimuli 2-20):
%     Col 1: Stimulus number (implicitly corresponds to 2:20)
%     Col 2: Mean accuracy (for stimuli 2-20)
%     Col 3: Median accuracy
%     Col 4: Standard Error (SE) (NOTE: This column is no longer used for plotting error bars)
%
% INPUT:
%   metrics_mt - Cell array (iS indexes subjects).
%                metrics_mt{iS}(iD) is a structure for one session,
%                containing .Acc_NumStim for the CR task.
% --- Hardcoded Subject Information and Colors ---
subject_id_list = {
    'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan' 
};
% custom_colors = {
%     [245, 124, 110]/255; [242, 181, 110]/255; [251, 231, 158]/255; % Colors 1-3
%     [132, 195, 183]/255; [136, 215, 218]/255; [113, 184, 237]/255; % Colors 4-6
%     [184, 174, 234]/255; [242, 168, 218]/255 % Colors 7-8
% };
custom_colors = {
    [0.945, 0.345, 0.275]; % #F15846
    [1.0, 0.690, 0.314];   % #FFB050
    [0.984, 0.906, 0.620]; % #FBE79E
    [0.533, 0.843, 0.855]; % #88D7DA
    [0.341, 0.663, 0.902]; % #57A9E6
    [0.420, 0.447, 0.714]  % #6B72B6
};
% --- X-axis for Plotting ---
x_axis_for_plot = (1:20)'; % Changed to 1-20
% --- Expected Input Data Format for Acc_NumStim (for stimuli 2-20) ---
expected_input_num_rows = 19; % Corresponds to stimuli 2-20
expected_input_num_cols = 4;
% --- Font Size ---
base_font_size = 14;
% --- Input Validation ---
if nargin < 1
    error('Usage: plot_cr_accuracy_by_subject(metrics_mt)');
end
if ~iscell(metrics_mt)
    error('metrics_mt must be a cell array.');
end
% --- Figure Setup ---
figure_width = 500;
figure_height = 240;
fig = figure('Position', [100, 100, figure_width, figure_height]);
ax = gca;
hold(ax, 'on');
legend_handles = [];
plotted_anything_for_legend = false;
num_subjects_in_data = length(metrics_mt);
% --- Main Loop for Subjects ---
for iS = 1:num_subjects_in_data
    session_accuracies_for_subject_1_to_20 = {}; % To collect augmented 20-element accuracy vectors
    
    if iS > length(metrics_mt) || isempty(metrics_mt{iS})
        continue;
    end
    
    num_sessions_for_subject = length(metrics_mt{iS});
    valid_sessions_count = 0;
    for iD = 1:num_sessions_for_subject
        if ~isstruct(metrics_mt{iS}(iD))
            continue;
        end
        session_data = metrics_mt{iS}(iD);
        
        subject_name_for_warning = 'Unknown';
        if iS <= length(subject_id_list)
            subject_name_for_warning = subject_id_list{iS};
        else
            subject_name_for_warning = sprintf('Subject %d', iS);
        end
        if isfield(session_data, 'Acc_NumStim') && ~isempty(session_data.Acc_NumStim)
            acc_stim_data = session_data.Acc_NumStim;
            
            if size(acc_stim_data, 1) == expected_input_num_rows && size(acc_stim_data, 2) == expected_input_num_cols
                mean_acc_2_to_20 = acc_stim_data(:, 2);
                full_session_accuracy_1_to_20 = [1; mean_acc_2_to_20]; 
                session_accuracies_for_subject_1_to_20{end+1} = full_session_accuracy_1_to_20; 
                valid_sessions_count = valid_sessions_count + 1;
            else
                warning('Subject %s (Index %d), Session %d: Acc_NumStim dimensions (%dx%d) do not match expected format %dx%d. Skipping.', ...
                        subject_name_for_warning, iS, iD, size(acc_stim_data,1), size(acc_stim_data,2), expected_input_num_rows, expected_input_num_cols);
            end
        end
    end
    if valid_sessions_count > 0
        % Concatenate all session data for the subject
        all_sessions_matrix = cat(2, session_accuracies_for_subject_1_to_20{:});
        
        % Calculate the average accuracy across sessions for each stimulus count
        subject_avg_accuracy = nanmean(all_sessions_matrix, 2); % Will be 20x1
        
        % *** REVISED: Calculate Standard Error of the Mean (SEM) across sessions ***
        % This is the standard deviation of session means divided by the sqrt of the number of sessions.
        if valid_sessions_count > 1
            % Calculate the standard deviation across the session means for each stimulus count.
            subject_std_dev = nanstd(all_sessions_matrix, 0, 2);
            % Calculate the SEM.
            subject_sem = subject_std_dev / sqrt(valid_sessions_count);
        else
            % SEM cannot be calculated with only one session, so error is zero.
            subject_sem = zeros(size(subject_avg_accuracy));
        end

        if iS <= length(subject_id_list)
            current_subject_label = subject_id_list{iS};
        else
            current_subject_label = sprintf('Subject %d', iS);
        end
        
        if iS <= length(custom_colors)
            current_color = custom_colors{iS};
        else
            current_color = rand(1,3); 
            if iS > length(subject_id_list) 
                 warning('Not enough custom colors or subject IDs. Using random color for %s.', current_subject_label);
            end
        end
        
        % Plot with calculated SEM error bars
        h = errorbar(ax, x_axis_for_plot, subject_avg_accuracy, subject_sem, ...
                 'LineWidth', 1.5, 'Color', current_color, 'Marker', 'o', ...
                 'MarkerSize', 4, 'DisplayName', current_subject_label, ...
                 'CapSize', 3);
        
        legend_handles(end+1) = h;
        plotted_anything_for_legend = true;
    end
end
% --- Finalize Plot Styling and Saving (code remains the same) ---
set(ax, 'FontSize', base_font_size);
if plotted_anything_for_legend
    lgd = legend(ax, legend_handles, 'Location', 'eastoutside', 'FontSize', base_font_size);
    lgd.Box = 'off'; 
else
    text(ax, 0.5, 0.5, 'No data available to plot.', 'HorizontalAlignment', 'center', 'Units', 'normalized', 'FontSize', base_font_size, 'Color', 'red');
    warning('No data was plotted. Check input `metrics_mt` and ensure .Acc_NumStim field matches expected format.');
end
xlabel(ax, 'Number of Stimuli', 'FontSize', base_font_size);
ylabel(ax, 'Average Accuracy', 'FontSize', base_font_size);
title(ax, 'CR Task: Average Accuracy vs. Number of Stimuli by Subject', 'FontSize', base_font_size);
grid(ax, 'off'); 
axis(ax, 'tight');
xlim(ax, [min(x_axis_for_plot)-0.5 max(x_axis_for_plot)+0.5]);

drawnow; 
pause(0.1); 
if plotted_anything_for_legend && exist('lgd', 'var') && isvalid(lgd)
    try 
        if strcmp(lgd.Location, 'eastoutside')
            original_ax_units = get(ax, 'Units');
            original_lgd_units = get(lgd, 'Units');
            set(ax, 'Units', 'normalized');
            set(lgd, 'Units', 'normalized');
            drawnow; pause(0.1);
            ax_pos_norm = get(ax, 'Position'); 
            lgd_outer_pos_norm = get(lgd, 'OuterPosition'); 
            max_allowable_ax_width = lgd_outer_pos_norm(1) - ax_pos_norm(1) - 0.03; 
            if ax_pos_norm(3) > max_allowable_ax_width && max_allowable_ax_width > 0.1 
                set(ax, 'Position', [ax_pos_norm(1), ax_pos_norm(2), max_allowable_ax_width, ax_pos_norm(4)]);
            end
            set(ax, 'Units', original_ax_units);
            set(lgd, 'Units', original_lgd_units);
        end
    catch ME_layout
        fprintf('Warning: Could not auto-adjust layout for legend: %s\n', ME_layout.message);
        if exist('original_ax_units','var'); set(ax, 'Units', original_ax_units); end
        if exist('original_lgd_units','var'); set(lgd, 'Units', original_lgd_units); end
    end
end

if exist('fig', 'var') && isvalid(fig)
    figure(fig); 
    base_filename = 'cr_accuracy_vs_stimuli_by_subject'; 
    date_str = datestr(now, 'yyyymmdd_HHMMSS'); 
    full_base_filename = [base_filename '_' date_str];
    save_folder = 'CR_Figures_Accuracy'; 
    if ~exist(save_folder, 'dir')
        try
            mkdir(save_folder);
        catch ME_mkdir
            fprintf('Error creating save directory ''%s'': %s. Saving to current directory.\n', save_folder, ME_mkdir.message);
            save_folder = '.'; 
        end
    end
    filepath_base = fullfile(save_folder, full_base_filename);
    try
        eps_filename = [filepath_base '.eps'];
        print(fig, eps_filename, '-depsc', '-painters');
        fprintf('Figure saved as: %s (vector format)\n', eps_filename);
    catch ME_save_eps
        fprintf('Error saving figure as EPS: %s\n', ME_save_eps.message);
    end
    try
        png_filename = [filepath_base '.png'];
        saveas(fig, png_filename);
        fprintf('Figure saved as: %s\n', png_filename);
    catch ME_save_png
        fprintf('Error saving figure as PNG: %s\n', ME_save_png.message);
    end
    % (The CSV saving and p-value calculation code remains unchanged)
    all_data_for_csv = table();
    all_pvalues_for_csv = table();
    subject_data_cells = cell(0, 3);
    individual_subject_session_accuracies = cell(num_subjects_in_data, 1); 
    for iS_data = 1:num_subjects_in_data
        session_accuracies_for_subject_1_to_20_current = {};
        if iS_data > length(metrics_mt) || isempty(metrics_mt{iS_data})
            continue;
        end
        num_sessions_for_subject = length(metrics_mt{iS_data});
        current_subject_name = 'Unknown';
        if iS_data <= length(subject_id_list)
            current_subject_name = subject_id_list{iS_data};
        else
            current_subject_name = sprintf('Subject %d', iS_data);
        end
        for iD_data = 1:num_sessions_for_subject
            if ~isstruct(metrics_mt{iS_data}(iD_data))
                continue;
            end
            session_data = metrics_mt{iS_data}(iD_data);
            if isfield(session_data, 'Acc_NumStim') && ~isempty(session_data.Acc_NumStim)
                acc_stim_data = session_data.Acc_NumStim;
                if size(acc_stim_data, 1) == expected_input_num_rows && size(acc_stim_data, 2) == expected_input_num_cols
                    mean_acc_2_to_20 = acc_stim_data(:, 2);
                    full_session_accuracy_1_to_20 = [1; mean_acc_2_to_20]; 
                    session_accuracies_for_subject_1_to_20_current{end+1} = full_session_accuracy_1_to_20; 
                end
            end
        end
        if ~isempty(session_accuracies_for_subject_1_to_20_current)
            individual_subject_session_accuracies{iS_data} = cat(2, session_accuracies_for_subject_1_to_20_current{:});
            subject_avg_accuracy = nanmean(individual_subject_session_accuracies{iS_data}, 2);
            for stim_idx = 1:length(x_axis_for_plot)
                subject_data_cells(end+1, :) = {current_subject_name, x_axis_for_plot(stim_idx), subject_avg_accuracy(stim_idx)};
            end
        end
    end
    if ~isempty(subject_data_cells)
        all_data_for_csv = cell2table(subject_data_cells, 'VariableNames', {'Subject', 'StimulusCount', 'AverageAccuracy'});
        data_csv_filename = [filepath_base '_accuracy_data.csv'];
        try
            writetable(all_data_for_csv, data_csv_filename);
            fprintf('Accuracy data saved as: %s\n', data_csv_filename);
        catch ME_save_csv
            fprintf('Error saving accuracy data CSV: %s\n', ME_save_csv.message);
        end
    else
        fprintf('No valid accuracy data to save to CSV.\n');
    end
    if num_subjects_in_data > 1
        pvalue_results_cell = cell(0, 3);
        for current_stim_count = 1:20
            accuracies_for_this_stim = [];
            group_labels_for_this_stim = {};
            for iS_pval = 1:num_subjects_in_data
                if ~isempty(individual_subject_session_accuracies{iS_pval})
                    if size(individual_subject_session_accuracies{iS_pval}, 1) >= current_stim_count
                        subject_accuracies_at_stim = individual_subject_session_accuracies{iS_pval}(current_stim_count, :);
                        subject_accuracies_at_stim = subject_accuracies_at_stim(~isnan(subject_accuracies_at_stim));
                        if ~isempty(subject_accuracies_at_stim)
                            accuracies_for_this_stim = [accuracies_for_this_stim, subject_accuracies_at_stim];
                            if iS_pval <= length(subject_id_list)
                                current_subject_name = subject_id_list{iS_pval};
                            else
                                current_subject_name = sprintf('Subject %d', iS_pval);
                            end
                            group_labels_for_this_stim = [group_labels_for_this_stim, repmat({current_subject_name}, 1, length(subject_accuracies_at_stim))];
                        end
                    end
                end
            end
            unique_groups = unique(group_labels_for_this_stim);
            if length(unique_groups) > 1 && length(accuracies_for_this_stim) >= length(unique_groups)
                try
                    [p_val, ~, stats] = anova1(accuracies_for_this_stim, group_labels_for_this_stim, 'off');
                    pvalue_results_cell(end+1, :) = {current_stim_count, 'ANOVA (Overall)', p_val};
                    if p_val < 0.05
                        c = multcompare(stats, 'Display', 'off');
                        for row_idx = 1:size(c, 1)
                            comparison_p = c(row_idx, 6);
                            comparison_str = sprintf('%s vs. %s', unique_groups{c(row_idx, 1)}, unique_groups{c(row_idx, 2)});
                            pvalue_results_cell(end+1, :) = {current_stim_count, comparison_str, comparison_p};
                        end
                    end
                catch ME_stats
                    fprintf('Warning: Could not perform stats for Stim Count %d: %s\n', current_stim_count, ME_stats.message);
                    pvalue_results_cell(end+1, :) = {current_stim_count, 'Error in Stats', NaN};
                end
            elseif length(unique_groups) <= 1
                pvalue_results_cell(end+1, :) = {current_stim_count, 'Not enough groups for comparison', NaN};
            else
                pvalue_results_cell(end+1, :) = {current_stim_count, 'Not enough data points', NaN};
            end
        end
        if ~isempty(pvalue_results_cell)
            all_pvalues_for_csv = cell2table(pvalue_results_cell, 'VariableNames', {'StimulusCount', 'Comparison', 'PValue'});
            pvalue_csv_filename = [filepath_base '_pvalues.csv'];
            try
                writetable(all_pvalues_for_csv, pvalue_csv_filename);
                fprintf('P-values data saved as: %s\n', pvalue_csv_filename);
            catch ME_save_pval_csv
                fprintf('Error saving p-values data CSV: %s\n', ME_save_pval_csv.message);
            end
        else
            fprintf('No p-value data to save to CSV.\n');
        end
    else
        fprintf('Not enough subjects (%d) for statistical comparison.\n', num_subjects_in_data);
    end
else
    fprintf('Figure handle not valid. Figure and data not saved.\n');
end
hold(ax, 'off');
end


function plot_AS_RT_overall_and_pairs(metrics_mt)
% PLOT_AS_RT_OVERALL_AND_PAIRS Plots Anti-Saccade task RT (in seconds)
% for Overall, Pro, Anti, Congruent, and Incongruent conditions.
%
% Based on plot_AS_accuracy_final, adapted for RT.
% Features:
%   - Y-axis limits are dynamic for RT.
%   - Significance lines for Pro vs Anti RT and Cong vs Incong RT comparisons
%     are at the same y-level.
%   - Grand average RT plotted as black dots with error bars.
%   - Individual subject average RTs plotted as colored dots.
%   - Lines connect subject's Pro-Anti RT dots and Congruent-Incongruent RT dots.
%   - 'Overall' subject RT is a standalone dot.
%   - T-tests on session-level RT differences for significance bars.
%   - Saves plot as PNG and EPS, and exports data to CSV files.
%
% Args:
%   metrics_mt (cell array): Data structure.

% --- Configuration ---
plot_font_size = 14;
figure_width = 450;
figure_height = 350;
subject_id_list_hardcoded = {
    'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan'
};
custom_colors_rgb = {
[241, 88, 70]/255; [255, 176, 80]/255; [251, 231, 158]/255; 
    [136, 215, 218]/255; [87, 169, 230]/255; [107, 114, 182]/255 
};

num_hardcoded_subject_details = length(subject_id_list_hardcoded);
num_defined_custom_colors = length(custom_colors_rgb);
num_subjects_to_process = length(metrics_mt);
if num_subjects_to_process == 0
    disp('Input metrics_mt is empty. No subjects to plot.');
    return;
end

% --- Data Aggregation ---
subject_session_rts.all   = cell(num_subjects_to_process, 1);
subject_session_rts.pro   = cell(num_subjects_to_process, 1);
subject_session_rts.anti  = cell(num_subjects_to_process, 1);
subject_session_rts.cong  = cell(num_subjects_to_process, 1);
subject_session_rts.incong = cell(num_subjects_to_process, 1);
session_rt_differences.pro_anti = [];
session_rt_differences.cong_incong = [];
subject_legend_labels = cell(1, num_subjects_to_process);
valid_subject_indices = [];
for iS = 1:num_subjects_to_process
    if iS <= num_hardcoded_subject_details && ~isempty(subject_id_list_hardcoded{iS})
        subject_name_from_list = subject_id_list_hardcoded{iS};
        subject_legend_labels{iS} = subject_name_from_list;
    else
        subject_legend_labels{iS} = ['S' num2str(iS)];
    end
    if isempty(metrics_mt{iS})
        continue;
    end
    
    current_subject_has_data = false;
    for iD = 1:length(metrics_mt{iS})
        session_data = metrics_mt{iS}(iD);
        
        if ~isfield(session_data, 'RTMeanSE_Pro') || ~isfield(session_data, 'RTMeanSE_Anti') || ...
           ~isfield(session_data, 'RT_Cong_Combined') 
            continue;
        end
        
		if isempty(session_data.RTMeanSE_Pro) || isempty(session_data.RTMeanSE_Anti) || ...
           isempty(session_data.RT_Cong_Combined) 
            continue;
		end
        rt_pro_mean  = session_data.RTMeanSE_Pro(1);
        count_pro    = session_data.RTMeanSE_Pro(4);
        rt_anti_mean = session_data.RTMeanSE_Anti(1);
        count_anti   = session_data.RTMeanSE_Anti(4);
        
        session_rt_all = NaN;
        if (count_pro + count_anti) > 0 && ~isnan(rt_pro_mean) && ~isnan(rt_anti_mean)
            session_rt_all = (rt_pro_mean * count_pro + rt_anti_mean * count_anti) / (count_pro + count_anti);
        elseif ~isnan(rt_pro_mean) && (isnan(rt_anti_mean) || count_anti == 0) && count_pro > 0 
            session_rt_all = rt_pro_mean;
        elseif (isnan(rt_pro_mean) || count_pro == 0) && ~isnan(rt_anti_mean) && count_anti > 0
            session_rt_all = rt_anti_mean;
        end
        
        rt_cong_combined = session_data.RT_Cong_Combined;
        session_rt_cong = NaN; session_rt_incong = NaN;
        if size(rt_cong_combined,1) >= 4 && size(rt_cong_combined,2) >=4
            mean_rt_ll = rt_cong_combined(1,1); count_ll = rt_cong_combined(1,4);
            mean_rt_rr = rt_cong_combined(3,1); count_rr = rt_cong_combined(3,4);
            if (count_ll + count_rr) > 0
                 valid_ll = ~isnan(mean_rt_ll) && count_ll > 0; valid_rr = ~isnan(mean_rt_rr) && count_rr > 0;
                 if valid_ll && valid_rr; session_rt_cong = (mean_rt_ll * count_ll + mean_rt_rr * count_rr) / (count_ll + count_rr);
                 elseif valid_ll; session_rt_cong = mean_rt_ll; elseif valid_rr; session_rt_cong = mean_rt_rr; end
            end
            mean_rt_lr = rt_cong_combined(2,1); count_lr = rt_cong_combined(2,4);
            mean_rt_rl = rt_cong_combined(4,1); count_rl = rt_cong_combined(4,4);
            if (count_lr + count_rl) > 0
                 valid_lr = ~isnan(mean_rt_lr) && count_lr > 0; valid_rl = ~isnan(mean_rt_rl) && count_rl > 0;
                 if valid_lr && valid_rl; session_rt_incong = (mean_rt_lr * count_lr + mean_rt_rl * count_rl) / (count_lr + count_rl);
                 elseif valid_lr; session_rt_incong = mean_rt_lr; elseif valid_rl; session_rt_incong = mean_rt_rl; end
            end
        end
        
        if ~isnan(session_rt_all);  subject_session_rts.all{iS}    = [subject_session_rts.all{iS}, session_rt_all]; end
        if ~isnan(rt_pro_mean);     subject_session_rts.pro{iS}    = [subject_session_rts.pro{iS}, rt_pro_mean]; end
        if ~isnan(rt_anti_mean);    subject_session_rts.anti{iS}   = [subject_session_rts.anti{iS}, rt_anti_mean]; end
        if ~isnan(session_rt_cong); subject_session_rts.cong{iS}   = [subject_session_rts.cong{iS}, session_rt_cong]; end
        if ~isnan(session_rt_incong); subject_session_rts.incong{iS} = [subject_session_rts.incong{iS}, session_rt_incong]; end
        
        if ~isnan(rt_pro_mean) && ~isnan(rt_anti_mean)
            session_rt_differences.pro_anti = [session_rt_differences.pro_anti, rt_anti_mean - rt_pro_mean]; % Anti-Pro for cost
        end
        if ~isnan(session_rt_cong) && ~isnan(session_rt_incong)
            session_rt_differences.cong_incong = [session_rt_differences.cong_incong, session_rt_incong - session_rt_cong]; % Incong-Cong for cost
        end
        
        if any([~isnan(session_rt_all), ~isnan(rt_pro_mean), ~isnan(rt_anti_mean), ~isnan(session_rt_cong), ~isnan(session_rt_incong)])
            current_subject_has_data = true;
        end
    end
    if current_subject_has_data
        valid_subject_indices = [valid_subject_indices, iS];
    end
end
if isempty(valid_subject_indices)
    disp('No subjects with valid AS RT data found. Cannot generate plot.');
    return;
end
num_valid_subjects = length(valid_subject_indices);
subj_avg_rt.all    = NaN(1, num_valid_subjects);
subj_avg_rt.pro    = NaN(1, num_valid_subjects);
subj_avg_rt.anti   = NaN(1, num_valid_subjects);
subj_avg_rt.cong   = NaN(1, num_valid_subjects);
subj_avg_rt.incong = NaN(1, num_valid_subjects);
actual_subject_legend_labels = cell(1, num_valid_subjects);
actual_subject_colors = cell(1, num_valid_subjects);
for i_valid_subj = 1:num_valid_subjects
    iS = valid_subject_indices(i_valid_subj);
    actual_subject_legend_labels{i_valid_subj} = subject_legend_labels{iS};
    color_idx_cycle = mod(i_valid_subj - 1, num_defined_custom_colors) + 1;
    if i_valid_subj > num_defined_custom_colors 
        cc = lines(num_valid_subjects); 
        actual_subject_colors{i_valid_subj} = cc(i_valid_subj,:);
    else
         actual_subject_colors{i_valid_subj} = custom_colors_rgb{color_idx_cycle};
    end
    
    if ~isempty(subject_session_rts.all{iS});    subj_avg_rt.all(i_valid_subj)    = mean(subject_session_rts.all{iS}, 'omitnan'); end
    if ~isempty(subject_session_rts.pro{iS});    subj_avg_rt.pro(i_valid_subj)    = mean(subject_session_rts.pro{iS}, 'omitnan'); end
    if ~isempty(subject_session_rts.anti{iS});   subj_avg_rt.anti(i_valid_subj)   = mean(subject_session_rts.anti{iS}, 'omitnan'); end
    if ~isempty(subject_session_rts.cong{iS});   subj_avg_rt.cong(i_valid_subj)   = mean(subject_session_rts.cong{iS}, 'omitnan'); end
    if ~isempty(subject_session_rts.incong{iS}); subj_avg_rt.incong(i_valid_subj) = mean(subject_session_rts.incong{iS}, 'omitnan'); end
end
all_sessions_flat_rts.all    = horzcat(subject_session_rts.all{valid_subject_indices});
all_sessions_flat_rts.pro    = horzcat(subject_session_rts.pro{valid_subject_indices});
all_sessions_flat_rts.anti   = horzcat(subject_session_rts.anti{valid_subject_indices});
all_sessions_flat_rts.cong   = horzcat(subject_session_rts.cong{valid_subject_indices});
all_sessions_flat_rts.incong = horzcat(subject_session_rts.incong{valid_subject_indices});
grand_mean_rt.all    = mean(all_sessions_flat_rts.all, 'omitnan');
grand_mean_rt.pro    = mean(all_sessions_flat_rts.pro, 'omitnan');
grand_mean_rt.anti   = mean(all_sessions_flat_rts.anti, 'omitnan');
grand_mean_rt.cong   = mean(all_sessions_flat_rts.cong, 'omitnan');
grand_mean_rt.incong = mean(all_sessions_flat_rts.incong, 'omitnan');
grand_sem_rt.all    = std(all_sessions_flat_rts.all, 0, 'omitnan') / sqrt(sum(~isnan(all_sessions_flat_rts.all)));
grand_sem_rt.pro    = std(all_sessions_flat_rts.pro, 0, 'omitnan') / sqrt(sum(~isnan(all_sessions_flat_rts.pro)));
grand_sem_rt.anti   = std(all_sessions_flat_rts.anti, 0, 'omitnan') / sqrt(sum(~isnan(all_sessions_flat_rts.anti)));
grand_sem_rt.cong   = std(all_sessions_flat_rts.cong, 0, 'omitnan') / sqrt(sum(~isnan(all_sessions_flat_rts.cong)));
grand_sem_rt.incong = std(all_sessions_flat_rts.incong, 0, 'omitnan') / sqrt(sum(~isnan(all_sessions_flat_rts.incong)));

% --- Plotting ---
screen_size = get(0, 'ScreenSize');
fig_pos_x = (screen_size(3) - figure_width) / 2;
fig_pos_y = (screen_size(4) - figure_height) / 2;
fig_handle = figure('Position', [fig_pos_x, fig_pos_y, figure_width, figure_height], 'Color', 'w');
ax = gca;
hold(ax, 'on'); 
x_positions = [1, 2.2, 3, 4.4, 5.2]; 
condition_labels = {'Overall', 'Pro', 'Anti', 'Cong.', 'Incong.'};
grand_means_rt_vector = [grand_mean_rt.all, grand_mean_rt.pro, grand_mean_rt.anti, grand_mean_rt.cong, grand_mean_rt.incong];
grand_sems_rt_vector  = [grand_sem_rt.all, grand_sem_rt.pro, grand_sem_rt.anti, grand_sem_rt.cong, grand_sem_rt.incong];

subj_handles = gobjects(1, num_valid_subjects);
marker_size_subj = 5; % MODIFIED: MarkerSize 6 -> 5
line_width_subj = 1;
for i_subj = 1:num_valid_subjects
    subj_color = actual_subject_colors{i_subj};
    current_x_offset = 0; 
    
    x_overall = x_positions(1) + current_x_offset;
    y_overall_rt = subj_avg_rt.all(i_subj);
    
    x_pro_anti_rt = x_positions(2:3) + current_x_offset;
    y_pro_anti_rt = [subj_avg_rt.pro(i_subj), subj_avg_rt.anti(i_subj)];
    
    x_cong_incong_rt = x_positions(4:5) + current_x_offset;
    y_cong_incong_rt = [subj_avg_rt.cong(i_subj), subj_avg_rt.incong(i_subj)];
    if ~isnan(y_overall_rt)
        h_subj_plot = plot(ax, x_overall, y_overall_rt, 'o', ...
            'Color', subj_color, ...
            'MarkerFaceColor', subj_color, ...
            'MarkerEdgeColor', subj_color*0.7, ...
            'LineWidth', line_width_subj, 'MarkerSize', marker_size_subj, ...
            'HandleVisibility', 'on'); 
        subj_handles(i_subj) = h_subj_plot;
    end
    
    if any(~isnan(y_pro_anti_rt))
        plot(ax, x_pro_anti_rt, y_pro_anti_rt, '-o', ...
            'Color', subj_color, ...
            'MarkerFaceColor', subj_color, ...
            'MarkerEdgeColor', subj_color*0.7, ...
            'LineWidth', line_width_subj, 'MarkerSize', marker_size_subj, ...
            'HandleVisibility', 'off'); 
    end
    
    if any(~isnan(y_cong_incong_rt))
        plot(ax, x_cong_incong_rt, y_cong_incong_rt, '-o', ...
            'Color', subj_color, ...
            'MarkerFaceColor', subj_color, ...
            'MarkerEdgeColor', subj_color*0.7, ...
            'LineWidth', line_width_subj, 'MarkerSize', marker_size_subj, ...
            'HandleVisibility', 'off'); 
    end
end
plot(ax, x_positions, grand_means_rt_vector, 'ko', ...
    'MarkerSize', 5, 'MarkerFaceColor', 'k', 'LineWidth', 1.5, 'DisplayName', 'Group Mean'); % MODIFIED: MarkerSize 6 -> 5
errorbar(ax, x_positions, grand_means_rt_vector, grand_sems_rt_vector, ...
    'k.', 'LineWidth', 1.5, 'CapSize', 10, 'HandleVisibility','off');
% --- Aesthetics ---
ax.FontSize = plot_font_size;
ylabel(ax, 'Reaction Time (s)', 'FontSize', plot_font_size);
title(ax, 'Anti-Saccade Task RT', 'FontSize', plot_font_size + 1);
xticks(ax, x_positions);
xticklabels(ax, condition_labels);
xtickangle(ax, 0);
xlim(ax, [x_positions(1)-0.5, x_positions(end)+0.5]);
all_plotted_y_values_rt = []; 
grand_means_rt_no_nan = grand_means_rt_vector(~isnan(grand_means_rt_vector));
grand_sems_rt_no_nan = grand_sems_rt_vector(~isnan(grand_means_rt_vector)); 
if ~isempty(grand_means_rt_no_nan)
    all_plotted_y_values_rt = [all_plotted_y_values_rt; grand_means_rt_no_nan(:) + grand_sems_rt_no_nan(:); grand_means_rt_no_nan(:) - grand_sems_rt_no_nan(:)];
end
fields_rt = fieldnames(subj_avg_rt);
for f_idx = 1:length(fields_rt)
    data_field_rt = subj_avg_rt.(fields_rt{f_idx});
    if ~isempty(data_field_rt(~isnan(data_field_rt))); all_plotted_y_values_rt = [all_plotted_y_values_rt; data_field_rt(~isnan(data_field_rt))']; end
end
if ~isempty(all_plotted_y_values_rt)
    min_y_data_rt = min(all_plotted_y_values_rt(all_plotted_y_values_rt > -Inf & ~isinf(all_plotted_y_values_rt))); 
    max_y_data_rt = max(all_plotted_y_values_rt(all_plotted_y_values_rt < Inf & ~isinf(all_plotted_y_values_rt)));
    if isempty(min_y_data_rt); min_y_data_rt = 0.1; end 
    if isempty(max_y_data_rt); max_y_data_rt = 0.5; end
    y_range_rt = max_y_data_rt - min_y_data_rt;
    if y_range_rt <= 0; y_range_rt = 0.2; end
    y_padding_rt = 0.2 * y_range_rt;
    ylim_bottom_rt = max(0, min_y_data_rt - y_padding_rt);
    ylim_top_rt = max_y_data_rt + y_padding_rt;
    if ylim_bottom_rt >= ylim_top_rt; ylim_top_rt = ylim_bottom_rt + 0.1; end
    ylim(ax, [ylim_bottom_rt ylim_top_rt]);
else
    ylim(ax, [0.1 0.8]);
end
grid(ax, 'off'); 
box(ax, 'off');  
valid_subj_handles = subj_handles(isgraphics(subj_handles));
valid_subj_labels_for_legend = actual_subject_legend_labels(isgraphics(subj_handles));
if ~isempty(valid_subj_handles) && ~isempty(valid_subj_labels_for_legend)
    legend(ax, valid_subj_handles, valid_subj_labels_for_legend, 'Location', 'eastoutside', 'FontSize', plot_font_size-2, 'Box', 'off');
end

% --- Statistical Comparisons ---
y_lim_curr_rt = ylim(ax); 
text_y_offset_rt = 0.015 * diff(y_lim_curr_rt); 
relevant_y_values_for_stat_lines_rt = [];
upper_bounds_fields_rt = {'pro', 'anti', 'cong', 'incong'};
for f_idx = 1:length(upper_bounds_fields_rt)
    field_rt = upper_bounds_fields_rt{f_idx};
    if isfield(grand_mean_rt, field_rt) && isfield(grand_sem_rt, field_rt) && ...
       ~isnan(grand_mean_rt.(field_rt)) && ~isnan(grand_sem_rt.(field_rt))
        relevant_y_values_for_stat_lines_rt(end+1) = grand_mean_rt.(field_rt) + grand_sem_rt.(field_rt);
    end
    if isfield(subj_avg_rt, field_rt)
        subj_data_rt = subj_avg_rt.(field_rt);
        relevant_y_values_for_stat_lines_rt = [relevant_y_values_for_stat_lines_rt, subj_data_rt(~isnan(subj_data_rt))];
    end
end
common_stat_line_y_rt = y_lim_curr_rt(1) + 0.90 * diff(y_lim_curr_rt);
if ~isempty(relevant_y_values_for_stat_lines_rt)
    top_data_boundary_rt = max(relevant_y_values_for_stat_lines_rt(isfinite(relevant_y_values_for_stat_lines_rt))); 
    if ~isempty(top_data_boundary_rt) && common_stat_line_y_rt < top_data_boundary_rt + (0.035 * diff(y_lim_curr_rt))
        common_stat_line_y_rt = top_data_boundary_rt + (0.045 * diff(y_lim_curr_rt)); 
    end
end
common_stat_line_y_rt = min(common_stat_line_y_rt, y_lim_curr_rt(2) - (0.02 * diff(y_lim_curr_rt))); 
if common_stat_line_y_rt <= y_lim_curr_rt(1) 
    common_stat_line_y_rt = y_lim_curr_rt(1) + 0.85 * diff(y_lim_curr_rt); 
end
if length(session_rt_differences.pro_anti) >= 2
    [~, p_rt_pro_vs_anti, ~, stats_rt_pro_anti] = ttest(session_rt_differences.pro_anti); % MODIFIED: Get stats
    fprintf('One-sample t-test on session RT differences (Anti - Pro RT (s)): p = %.4f (N_sessions = %d)\n', p_rt_pro_vs_anti, stats_rt_pro_anti.df + 1);
    stars = 'n.s.';
    if p_rt_pro_vs_anti < 0.001; stars = '***';
    elseif p_rt_pro_vs_anti < 0.01; stars = '**';
    elseif p_rt_pro_vs_anti < 0.05; stars = '*'; end
    
    plot(ax, [x_positions(2), x_positions(3)], [common_stat_line_y_rt, common_stat_line_y_rt], '-k', 'LineWidth', 1, 'HandleVisibility','off');
    text(ax, mean([x_positions(2), x_positions(3)]), common_stat_line_y_rt + text_y_offset_rt, ...
        stars, 'HorizontalAlignment', 'center', 'VerticalAlignment','bottom', 'FontSize', plot_font_size, 'BackgroundColor', 'none');
end
if length(session_rt_differences.cong_incong) >= 2
    [~, p_rt_cong_vs_incong, ~, stats_rt_cong_incong] = ttest(session_rt_differences.cong_incong); % MODIFIED: Get stats
    fprintf('One-sample t-test on session RT differences (Incong - Cong RT (s)): p = %.4f (N_sessions = %d)\n', p_rt_cong_vs_incong, stats_rt_cong_incong.df + 1);
    stars = 'n.s.';
    if p_rt_cong_vs_incong < 0.001; stars = '***';
    elseif p_rt_cong_vs_incong < 0.01; stars = '**';
    elseif p_rt_cong_vs_incong < 0.05; stars = '*'; end
    plot(ax, [x_positions(4), x_positions(5)], [common_stat_line_y_rt, common_stat_line_y_rt], '-k', 'LineWidth', 1, 'HandleVisibility','off');
    text(ax, mean([x_positions(4), x_positions(5)]), common_stat_line_y_rt + text_y_offset_rt, ...
        stars, 'HorizontalAlignment', 'center', 'VerticalAlignment','bottom', 'FontSize', plot_font_size, 'BackgroundColor', 'none');
end
hold(ax, 'off'); 
drawnow;
try 
    lgd = findobj(fig_handle, 'Type', 'Legend');
    if ~isempty(lgd) && strcmp(lgd.Location, 'eastoutside')
        ax_pos_norm = get(ax, 'Position'); 
        lgd_pos_norm = get(lgd, 'OuterPosition'); 
        max_allowable_ax_width = 1 - lgd_pos_norm(3) - ax_pos_norm(1) - 0.05;
        if ax_pos_norm(3) > max_allowable_ax_width && max_allowable_ax_width > 0.2 
            set(ax, 'Position', [ax_pos_norm(1), ax_pos_norm(2), max_allowable_ax_width, ax_pos_norm(4)]);
        end
    end
catch
end

% --- Saving Outputs ---
% MODIFIED: This section now saves figures (PNG, EPS) and data (CSV).
figure(fig_handle);
base_filename = 'as_rt_overall_and_pairs';
date_str = datestr(now, 'yyyymmdd_HHMM');
full_base_filename = [base_filename '_' date_str];
save_folder = 'AS_Figures_RT_Overall';
if ~exist(save_folder, 'dir'); mkdir(save_folder); end
filepath_base = fullfile(save_folder, full_base_filename);

% Save Figure (PNG and EPS)
try
    png_filename = [filepath_base '.png'];
    saveas(fig_handle, png_filename);
    fprintf('Figure saved as: %s\n', png_filename);
    
    eps_filename = [filepath_base '.eps'];
    print(fig_handle, eps_filename, '-depsc'); % Save as vector EPS
    fprintf('Figure saved as: %s\n', eps_filename);
catch ME
    fprintf('Error saving figure: %s\n', ME.message);
end

% Save Data to CSVs
try
    % Grand Averages Data
    grand_avg_table = table(condition_labels', grand_means_rt_vector', grand_sems_rt_vector', ...
        'VariableNames', {'Condition', 'Mean_RT_s', 'SEM_RT_s'});
    csv_grand_avg_filename = [filepath_base '_grand_averages_rt.csv'];
    writetable(grand_avg_table, csv_grand_avg_filename);
    fprintf('Grand average RT data saved to: %s\n', csv_grand_avg_filename);
    
    % Subject Averages Data
    subject_avg_data = [subj_avg_rt.all; subj_avg_rt.pro; subj_avg_rt.anti; subj_avg_rt.cong; subj_avg_rt.incong]';
    subject_avg_table = array2table(subject_avg_data, ...
        'RowNames', actual_subject_legend_labels, ...
        'VariableNames', {'Overall', 'Pro', 'Anti', 'Congruent', 'Incongruent'});
    csv_subj_avg_filename = [filepath_base '_subject_averages_rt.csv'];
    writetable(subject_avg_table, csv_subj_avg_filename, 'WriteRowNames', true);
    fprintf('Subject average RT data saved to: %s\n', csv_subj_avg_filename);
    
    % Statistical Results
    comparisons = {};
    p_values = [];
    t_stats = [];
    dfs = [];
    n_sessions = [];
    
    if exist('p_rt_pro_vs_anti', 'var')
        comparisons{end+1} = 'Anti vs Pro';
        p_values(end+1) = p_rt_pro_vs_anti;
        t_stats(end+1) = stats_rt_pro_anti.tstat;
        dfs(end+1) = stats_rt_pro_anti.df;
        n_sessions(end+1) = stats_rt_pro_anti.df + 1;
    end
    
    if exist('p_rt_cong_vs_incong', 'var')
        comparisons{end+1} = 'Incongruent vs Congruent';
        p_values(end+1) = p_rt_cong_vs_incong;
        t_stats(end+1) = stats_rt_cong_incong.tstat;
        dfs(end+1) = stats_rt_cong_incong.df;
        n_sessions(end+1) = stats_rt_cong_incong.df + 1;
    end
    
    if ~isempty(comparisons)
        stats_table = table(comparisons', p_values', t_stats', dfs', n_sessions', ...
            'VariableNames', {'Comparison', 'PValue', 'TStatistic', 'DF', 'N_Sessions'});
        csv_stats_filename = [filepath_base '_statistics_rt.csv'];
        writetable(stats_table, csv_stats_filename);
        fprintf('Statistical results for RT saved to: %s\n', csv_stats_filename);
    end
    
catch ME_csv
    fprintf('Error saving CSV data: %s\n', ME_csv.message);
end

end

function plot_AS_RT_differences(metrics_mt)
% PLOT_AS_RT_DIFFERENCES Plots RT differences (in seconds) for Pro-Anti
% and Congruent-Incongruent conditions in the Anti-Saccade task.
%
% Based on plot_AS_accuracy_differences, adapted for RT.
% Features:
%   - X-axis: Pro-Anti RT difference, Congruent-Incongruent RT difference.
%   - Y-axis: RT Difference (s). Made dynamic.
%   - Horizontal line at y=0.
%   - Subject-level data: Mean RT difference with SE, colored dots & error bars.
%   - Overall data: Mean RT difference with SE, black dot & error bar.
%   - Significance: One-sample t-test for overall differences against 0.
%   - Figure size: 400x200.
%
% Args:
%   metrics_mt (cell array): Data structure.

% --- Configuration ---
plot_font_size = 12; 
figure_width = 300;
figure_height = 300; % Slightly taller to accommodate dynamic RT y-axis potentially
subject_id_list_hardcoded = {
    'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan'
};
custom_colors_rgb = {
    [245, 124, 110]/255; [242, 181, 110]/255; [251, 231, 158]/255;
    [132, 195, 183]/255; [136, 215, 218]/255; [113, 184, 237]/255;
    [184, 174, 234]/255; [242, 168, 218]/255
};
num_hardcoded_subject_details = length(subject_id_list_hardcoded);
num_defined_custom_colors = length(custom_colors_rgb);
num_subjects_to_process = length(metrics_mt);

if num_subjects_to_process == 0
    disp('Input metrics_mt is empty. No subjects to plot.');
    return;
end

% --- Data Aggregation for RT Differences ---
subject_session_rt_diffs.pro_anti = cell(num_subjects_to_process, 1);
subject_session_rt_diffs.cong_incong = cell(num_subjects_to_process, 1);
subject_legend_labels = cell(1, num_subjects_to_process);
valid_subject_indices = [];

for iS = 1:num_subjects_to_process
    if iS <= num_hardcoded_subject_details && ~isempty(subject_id_list_hardcoded{iS})
        subject_name_from_list = subject_id_list_hardcoded{iS};
        subject_legend_labels{iS} = subject_name_from_list;
    else
        subject_legend_labels{iS} = ['S' num2str(iS)];
    end

    if isempty(metrics_mt{iS})
        continue;
    end
    
    current_subject_has_data_for_diff = false;
    for iD = 1:length(metrics_mt{iS})
        session_data = metrics_mt{iS}(iD);
        
        if ~isfield(session_data, 'RTMeanSE_Pro') || ~isfield(session_data, 'RTMeanSE_Anti') || ...
           ~isfield(session_data, 'RT_Cong_Combined') || isempty(session_data.RTMeanSE_Pro)
            continue;
        end
        
        rt_pro_mean  = session_data.RTMeanSE_Pro(1);
        rt_anti_mean = session_data.RTMeanSE_Anti(1);
        
        if ~isnan(rt_pro_mean) && ~isnan(rt_anti_mean)
            diff_rt_pa = rt_pro_mean - rt_anti_mean;
            subject_session_rt_diffs.pro_anti{iS} = [subject_session_rt_diffs.pro_anti{iS}, diff_rt_pa];
            current_subject_has_data_for_diff = true;
        end
        
        rt_cong_combined = session_data.RT_Cong_Combined;
        session_rt_cong = NaN; session_rt_incong = NaN;
        if size(rt_cong_combined,1) >= 4 && size(rt_cong_combined,2) >=4
            mean_rt_ll = rt_cong_combined(1,1); count_ll = rt_cong_combined(1,4);
            mean_rt_rr = rt_cong_combined(3,1); count_rr = rt_cong_combined(3,4);
            if (count_ll + count_rr) > 0
                 valid_ll = ~isnan(mean_rt_ll) && count_ll > 0; valid_rr = ~isnan(mean_rt_rr) && count_rr > 0;
                 if valid_ll && valid_rr; session_rt_cong = (mean_rt_ll * count_ll + mean_rt_rr * count_rr) / (count_ll + count_rr);
                 elseif valid_ll; session_rt_cong = mean_rt_ll; elseif valid_rr; session_rt_cong = mean_rt_rr; end
            end
            mean_rt_lr = rt_cong_combined(2,1); count_lr = rt_cong_combined(2,4);
            mean_rt_rl = rt_cong_combined(4,1); count_rl = rt_cong_combined(4,4);
            if (count_lr + count_rl) > 0
                 valid_lr = ~isnan(mean_rt_lr) && count_lr > 0; valid_rl = ~isnan(mean_rt_rl) && count_rl > 0;
                 if valid_lr && valid_rl; session_rt_incong = (mean_rt_lr * count_lr + mean_rt_rl * count_rl) / (count_lr + count_rl);
                 elseif valid_lr; session_rt_incong = mean_rt_lr; elseif valid_rl; session_rt_incong = mean_rt_rl; end
            end
        end
        if ~isnan(session_rt_cong) && ~isnan(session_rt_incong)
            diff_rt_ci = session_rt_cong - session_rt_incong; % Congruent RT - Incongruent RT
            subject_session_rt_diffs.cong_incong{iS} = [subject_session_rt_diffs.cong_incong{iS}, diff_rt_ci];
            current_subject_has_data_for_diff = true;
        end
    end
    if current_subject_has_data_for_diff
        valid_subject_indices = [valid_subject_indices, iS];
    end
end

if isempty(valid_subject_indices)
    disp('No subjects with valid session RT differences found. Cannot generate plot.');
    return;
end

num_valid_subjects = length(valid_subject_indices);
actual_subject_legend_labels = cell(1, num_valid_subjects);
actual_subject_colors = cell(1, num_valid_subjects);

subj_mean_rt_diff.pro_anti = NaN(1, num_valid_subjects);
subj_se_rt_diff.pro_anti   = NaN(1, num_valid_subjects);
subj_mean_rt_diff.cong_incong = NaN(1, num_valid_subjects);
subj_se_rt_diff.cong_incong   = NaN(1, num_valid_subjects);

disp('--- Subject RT Differences (Mean +/- SE from sessions) ---');
for i_idx = 1:num_valid_subjects
    iS = valid_subject_indices(i_idx);
    actual_subject_legend_labels{i_idx} = subject_legend_labels{iS};
    color_idx_cycle = mod(i_idx - 1, num_defined_custom_colors) + 1;
    if i_idx > num_defined_custom_colors
        cc = lines(num_valid_subjects); actual_subject_colors{i_idx} = cc(i_idx,:);
    else
        actual_subject_colors{i_idx} = custom_colors_rgb{color_idx_cycle};
    end

    sessions_rt_pa = subject_session_rt_diffs.pro_anti{iS};
    if ~isempty(sessions_rt_pa) && sum(~isnan(sessions_rt_pa)) > 0
        subj_mean_rt_diff.pro_anti(i_idx) = mean(sessions_rt_pa, 'omitnan');
        n_valid_sessions = sum(~isnan(sessions_rt_pa));
        if n_valid_sessions > 1
             subj_se_rt_diff.pro_anti(i_idx) = std(sessions_rt_pa, 0, 'omitnan') / sqrt(n_valid_sessions);
        else 
             subj_se_rt_diff.pro_anti(i_idx) = NaN; % SE is undefined or 0 for N=1
        end
        fprintf('Subject %s, Pro-Anti RT Diff (s): %.3f +/- %.3f (N_sessions=%d)\n', ...
            actual_subject_legend_labels{i_idx}, subj_mean_rt_diff.pro_anti(i_idx), subj_se_rt_diff.pro_anti(i_idx), n_valid_sessions);
    end

    sessions_rt_ci = subject_session_rt_diffs.cong_incong{iS};
    if ~isempty(sessions_rt_ci) && sum(~isnan(sessions_rt_ci)) > 0
        subj_mean_rt_diff.cong_incong(i_idx) = mean(sessions_rt_ci, 'omitnan');
        n_valid_sessions = sum(~isnan(sessions_rt_ci));
        if n_valid_sessions > 1
            subj_se_rt_diff.cong_incong(i_idx) = std(sessions_rt_ci, 0, 'omitnan') / sqrt(n_valid_sessions);
        else
            subj_se_rt_diff.cong_incong(i_idx) = NaN;
        end
        fprintf('Subject %s, Cong-Incong RT Diff (s): %.3f +/- %.3f (N_sessions=%d)\n', ...
            actual_subject_legend_labels{i_idx}, subj_mean_rt_diff.cong_incong(i_idx), subj_se_rt_diff.cong_incong(i_idx), n_valid_sessions);
    end
end
disp('-------------------------------------------------------------');

all_sessions_rt_diff.pro_anti = horzcat(subject_session_rt_diffs.pro_anti{valid_subject_indices});
all_sessions_rt_diff.cong_incong = horzcat(subject_session_rt_diffs.cong_incong{valid_subject_indices});

overall_mean_rt_diff.pro_anti = mean(all_sessions_rt_diff.pro_anti, 'omitnan');
overall_se_rt_diff.pro_anti = std(all_sessions_rt_diff.pro_anti, 0, 'omitnan') / sqrt(sum(~isnan(all_sessions_rt_diff.pro_anti)));
overall_mean_rt_diff.cong_incong = mean(all_sessions_rt_diff.cong_incong, 'omitnan');
overall_se_rt_diff.cong_incong = std(all_sessions_rt_diff.cong_incong, 0, 'omitnan') / sqrt(sum(~isnan(all_sessions_rt_diff.cong_incong)));

p_overall_rt_pro_anti = NaN; N_rt_pa_overall = 0;
if sum(~isnan(all_sessions_rt_diff.pro_anti)) >= 2
    [~, p_overall_rt_pro_anti, ~, stats_rt_pa] = ttest(all_sessions_rt_diff.pro_anti);
    N_rt_pa_overall = stats_rt_pa.df + 1;
end
p_overall_rt_cong_incong = NaN; N_rt_ci_overall = 0;
if sum(~isnan(all_sessions_rt_diff.cong_incong)) >= 2
    [~, p_overall_rt_cong_incong, ~, stats_rt_ci] = ttest(all_sessions_rt_diff.cong_incong);
    N_rt_ci_overall = stats_rt_ci.df + 1;
end

disp('--- Overall RT Differences (Mean +/- SE from all sessions) ---');
fprintf('Pro-Anti Overall RT Diff (s): %.3f +/- %.3f (N_sessions=%d), p(vs 0)=%.4f\n', ...
    overall_mean_rt_diff.pro_anti, overall_se_rt_diff.pro_anti, N_rt_pa_overall, p_overall_rt_pro_anti);
fprintf('Cong-Incong Overall RT Diff (s): %.3f +/- %.3f (N_sessions=%d), p(vs 0)=%.4f\n', ...
    overall_mean_rt_diff.cong_incong, overall_se_rt_diff.cong_incong, N_rt_ci_overall, p_overall_rt_cong_incong);
disp('--------------------------------------------------------------------');

% --- Plotting ---
screen_size = get(0, 'ScreenSize');
fig_pos_x = (screen_size(3) - figure_width) / 2;
fig_pos_y = (screen_size(4) - figure_height) / 2;
fig_handle = figure('Position', [fig_pos_x, fig_pos_y, figure_width, figure_height], 'Color', 'w');
ax = gca;
hold(ax, 'on');

x_conditions = [1, 2];
condition_labels = {'Pro - Anti', 'Cong. - Incong.'};
marker_size_overall = 5; % Adjusted from previous version
marker_size_subj = 4;    % Adjusted
errorbar_capsize = 4;    % Adjusted

yline(ax, 0, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1, 'HandleVisibility','off');

overall_rt_diff_means = [overall_mean_rt_diff.pro_anti, overall_mean_rt_diff.cong_incong];
overall_rt_diff_ses = [overall_se_rt_diff.pro_anti, overall_se_rt_diff.cong_incong];

errorbar(ax, x_conditions, overall_rt_diff_means, overall_rt_diff_ses, ...
    'LineStyle', 'none', 'Marker', 'o', 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k', ...
    'Color', 'k', 'LineWidth', 1.5, 'CapSize', errorbar_capsize, 'MarkerSize', marker_size_overall, ...
    'DisplayName', 'Overall Mean');

total_jitter_width = 0.4; 
if num_valid_subjects > 1
    jitter_values = linspace(-total_jitter_width/2, total_jitter_width/2, num_valid_subjects);
else
    jitter_values = 0; 
end

subj_handles = gobjects(1, num_valid_subjects);
for i_idx = 1:num_valid_subjects
    subj_color = actual_subject_colors{i_idx};
    current_jitter = jitter_values(i_idx);
    
    x_pa_subj = x_conditions(1) + current_jitter;
    y_pa_subj = subj_mean_rt_diff.pro_anti(i_idx);
    se_pa_subj = subj_se_rt_diff.pro_anti(i_idx);
    
    x_ci_subj = x_conditions(2) + current_jitter;
    y_ci_subj = subj_mean_rt_diff.cong_incong(i_idx);
    se_ci_subj = subj_se_rt_diff.cong_incong(i_idx);
    
    display_name_subj = actual_subject_legend_labels{i_idx};
    handle_visibility_subj = 'on';

    if ~isnan(y_pa_subj)
        errorbar(ax, x_pa_subj, y_pa_subj, se_pa_subj, ...
            'LineStyle', 'none', 'Marker', 'o', 'MarkerFaceColor', subj_color, 'MarkerEdgeColor', subj_color*0.7, ...
            'Color', subj_color, 'LineWidth', 1, 'CapSize', errorbar_capsize-2, 'MarkerSize', marker_size_subj, ...
            'HandleVisibility', handle_visibility_subj, 'DisplayName', display_name_subj);
        % Capture handle for the first plotted point of a subject for legend
        if strcmp(handle_visibility_subj,'on')
             temp_handles = findobj(ax, 'Type', 'ErrorBar', 'DisplayName', display_name_subj);
             if ~isempty(temp_handles); subj_handles(i_idx) = temp_handles(1); end
             handle_visibility_subj = 'off'; % Turn off for subsequent points of same subject
        end
    end
    if ~isnan(y_ci_subj)
         errorbar(ax, x_ci_subj, y_ci_subj, se_ci_subj, ...
            'LineStyle', 'none', 'Marker', 'o', 'MarkerFaceColor', subj_color, 'MarkerEdgeColor', subj_color*0.7, ...
            'Color', subj_color, 'LineWidth', 1, 'CapSize', errorbar_capsize-2, 'MarkerSize', marker_size_subj, ...
            'HandleVisibility', 'off'); 
     end
end

% --- Aesthetics ---
ax.FontSize = plot_font_size;
ylabel(ax, 'RT Difference (s)', 'FontSize', plot_font_size);
title(ax, 'RT Differences', 'FontSize', plot_font_size + 1);
xticks(ax, x_conditions);
xticklabels(ax, condition_labels);
xlim(ax, [0.5, 2.5]);

% Dynamic YLim for RT differences
all_plotted_y_vals = [];
if ~any(isnan(overall_rt_diff_means)) && ~any(isnan(overall_rt_diff_ses))
    all_plotted_y_vals = [all_plotted_y_vals; overall_rt_diff_means(:) + overall_rt_diff_ses(:); overall_rt_diff_means(:) - overall_rt_diff_ses(:)];
end
all_plotted_y_vals = [all_plotted_y_vals; subj_mean_rt_diff.pro_anti(:); subj_mean_rt_diff.cong_incong(:)];
% Add subject SEs to the y_vals for limit calculation if they are not NaNs
all_plotted_y_vals = [all_plotted_y_vals; subj_mean_rt_diff.pro_anti(:) + subj_se_rt_diff.pro_anti(:); subj_mean_rt_diff.pro_anti(:) - subj_se_rt_diff.pro_anti(:)];
all_plotted_y_vals = [all_plotted_y_vals; subj_mean_rt_diff.cong_incong(:) + subj_se_rt_diff.cong_incong(:); subj_mean_rt_diff.cong_incong(:) - subj_se_rt_diff.cong_incong(:)];

all_plotted_y_vals = all_plotted_y_vals(~isnan(all_plotted_y_vals) & ~isinf(all_plotted_y_vals));

if ~isempty(all_plotted_y_vals)
    min_y = min(all_plotted_y_vals);
    max_y = max(all_plotted_y_vals);
    y_range = max_y - min_y;
    if y_range == 0; y_range = 0.1; end % Default range if all points are same
    y_padding = 0.15 * y_range; % More padding for RT
    ylim_bottom = min_y - y_padding;
    ylim_top = max_y + y_padding;
    
    % Ensure zero is roughly centered or visible if range is one-sided
    if ylim_bottom > -0.02 && ylim_top > 0; ylim_bottom = min(ylim_bottom, -0.02); end
    if ylim_top < 0.02 && ylim_bottom < 0; ylim_top = max(ylim_top, 0.02); end
    if ylim_bottom == 0 && ylim_top == 0; ylim_bottom = -0.05; ylim_top = 0.05; end


    ylim(ax, [ylim_bottom, ylim_top]);
else
    ylim(ax, [-0.2, 0.2]); % Fallback YLim for RT differences
end


% Significance Stars
y_lim_curr = ylim(ax);
common_stat_line_y = y_lim_curr(2) - 0.1 * diff(y_lim_curr); % Position higher
if common_stat_line_y < max(all_plotted_y_vals(all_plotted_y_vals < Inf)) + 0.02*diff(y_lim_curr) % If it's too close to data
    common_stat_line_y = max(all_plotted_y_vals(all_plotted_y_vals < Inf)) + 0.05*diff(y_lim_curr);
end
common_stat_line_y = min(common_stat_line_y, y_lim_curr(2) - 0.02*diff(y_lim_curr)); % Don't go off chart

text_y_offset_stars = 0.01 * diff(y_lim_curr); 

if ~isnan(p_overall_rt_pro_anti)
    stars = 'n.s.';
    if p_overall_rt_pro_anti < 0.001; stars = '***';
    elseif p_overall_rt_pro_anti < 0.01; stars = '**';
    elseif p_overall_rt_pro_anti < 0.05; stars = '*'; end
    
    plot(ax, [x_conditions(1)-0.1, x_conditions(1)+0.1], [common_stat_line_y, common_stat_line_y], '-k', 'LineWidth', 1, 'HandleVisibility','off');
    text(ax, x_conditions(1), common_stat_line_y + text_y_offset_stars, stars, ...
        'HorizontalAlignment', 'center', 'VerticalAlignment','bottom', 'FontSize', plot_font_size, 'FontWeight', 'normal');
end

if ~isnan(p_overall_rt_cong_incong)
    stars = 'n.s.';
    if p_overall_rt_cong_incong < 0.001; stars = '***';
    elseif p_overall_rt_cong_incong < 0.01; stars = '**';
    elseif p_overall_rt_cong_incong < 0.05; stars = '*'; end
    
    plot(ax, [x_conditions(2)-0.1, x_conditions(2)+0.1], [common_stat_line_y, common_stat_line_y], '-k', 'LineWidth', 1, 'HandleVisibility','off');
    text(ax, x_conditions(2), common_stat_line_y + text_y_offset_stars, stars, ...
        'HorizontalAlignment', 'center', 'VerticalAlignment','bottom', 'FontSize', plot_font_size, 'FontWeight', 'normal');
end
hold(ax, 'off');

valid_legend_handles = subj_handles(isgraphics(subj_handles));
valid_labels_for_legend = actual_subject_legend_labels(isgraphics(subj_handles));
if ~isempty(valid_legend_handles)
    % legend(ax, valid_legend_handles, valid_labels_for_legend, 'Location', 'eastoutside', 'FontSize', plot_font_size-2, 'Box','off');
    % For small figures, legend outside can be problematic. Consider 'best' or manual.
end

% --- Saving Outputs ---
% MODIFIED: This section now saves figures (PNG, EPS) and data (CSV).
figure(fig_handle); 
base_filename = 'as_rt_cost';
date_str = datestr(now, 'yyyymmdd_HHMM');
full_base_filename = [base_filename '_' date_str];
save_folder = 'AS_Figures_RT_Difference'; 
if ~exist(save_folder, 'dir'); mkdir(save_folder); end
filepath_base = fullfile(save_folder, full_base_filename);

% Save Figure (PNG and EPS)
try
    png_filename = [filepath_base '.png'];
    saveas(fig_handle, png_filename);
    fprintf('Figure saved as: %s\n', png_filename);

    eps_filename = [filepath_base '.eps'];
    print(fig_handle, eps_filename, '-depsc', '-vector'); % Save as vector EPS
    fprintf('Figure saved as: %s\n', eps_filename);
catch ME_save
    fprintf('Error saving figure: %s\n', ME_save.message);
end

% Save Data to CSVs
try
    % --- Summary Data CSV ---
    groups = [{'Overall'}; actual_subject_legend_labels(:)];
    pa_means = [overall_mean_rt_diff.pro_anti; subj_mean_rt_diff.pro_anti(:)];
    pa_ses = [overall_se_rt_diff.pro_anti; subj_se_rt_diff.pro_anti(:)];
    ci_means = [overall_mean_rt_diff.cong_incong; subj_mean_rt_diff.cong_incong(:)];
    ci_ses = [overall_se_rt_diff.cong_incong; subj_se_rt_diff.cong_incong(:)];
    
    summary_table = table(groups, pa_means, pa_ses, ci_means, ci_ses, ...
        'VariableNames', {'Group', 'Anti_minus_Pro_Mean', 'Anti_minus_Pro_SE', 'Incong_minus_Cong_Mean', 'Incong_minus_Cong_SE'});

    csv_summary_filename = [filepath_base '_rt_cost_data.csv'];
    writetable(summary_table, csv_summary_filename);
    fprintf('Summary RT cost data saved to: %s\n', csv_summary_filename);

    % --- Statistics CSV ---
    comparisons = {};
    p_values = [];
    t_stats = [];
    dfs = [];
    n_sessions = [];

    if exist('stats_rt_pa', 'var')
        comparisons{end+1} = 'Anti-Pro Cost vs 0';
        p_values(end+1) = p_overall_rt_pro_anti;
        t_stats(end+1) = stats_rt_pa.tstat;
        dfs(end+1) = stats_rt_pa.df;
        n_sessions(end+1) = N_rt_pa_overall;
    end

    if exist('stats_rt_ci', 'var')
        comparisons{end+1} = 'Incong-Cong Cost vs 0';
        p_values(end+1) = p_overall_rt_cong_incong;
        t_stats(end+1) = stats_rt_ci.tstat;
        dfs(end+1) = stats_rt_ci.df;
        n_sessions(end+1) = N_rt_ci_overall;
    end

    if ~isempty(comparisons)
        stats_table = table(comparisons', p_values', t_stats', dfs', n_sessions', ...
            'VariableNames', {'Comparison', 'PValue', 'TStatistic', 'DF', 'N_Sessions'});
        csv_stats_filename = [filepath_base '_rt_cost_stats.csv'];
        writetable(stats_table, csv_stats_filename);
        fprintf('Statistical results for RT cost saved to: %s\n', csv_stats_filename);
    end

catch ME_csv
    fprintf('Error saving CSV data: %s\n', ME_csv.message);
end


end

function plot_AS_RT_diffs_over_sessions_time(metrics_mt)
% PLOT_AS_RT_DIFFS_OVER_SESSIONS_TIME Plots Pro-Anti and Congruent-Incongruent
% RT differences (in seconds) over chronologically ordered sessions.
%
% Based on plot_AS_diffs_over_sessions_time, adapted for RT.
% Features:
%   - Two subplots: Top for Pro-Anti RT diff, Bottom for Cong-Incong RT diff.
%   - X-axis: Session number, ordered by datetime.
%   - Y-axis: RT Difference (s).
%   - Each subject plotted as a colored line (smoothed).
%   - Overall average plotted as a black line (smoothed) with shaded SE.
%   - Overall line LineWidth 2 (no markers), subject lines LineWidth 1.
%   - Adds a linear regression line (red, dashed) for the overall data trend.
%   - Prints detailed regression statistics (R-squared, p-value) to the console.
%   - Figure size 800x800.
%   - Exports figure (PNG, EPS) and data (CSV), including regression stats.
%
% Args:
%   metrics_mt (cell array): Data structure.
%
% Requires: Statistics and Machine Learning Toolbox for fitlm().

% --- Configuration ---
plot_font_size = 14;
figure_width = 800;
figure_height = 800; 
smoothing_window_size = 5; 
subject_id_list_hardcoded = { 
    'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan' 
};
num_hardcoded_subject_details = length(subject_id_list_hardcoded);
custom_colors_rgb = {
    [245, 124, 110]/255; [242, 181, 110]/255; [251, 231, 158]/255;
    [132, 195, 183]/255; [136, 215, 218]/255; [113, 184, 237]/255;
    [184, 174, 234]/255; [242, 168, 218]/255
};
num_defined_custom_colors = length(custom_colors_rgb);
num_subjects_to_process = length(metrics_mt);
if num_subjects_to_process == 0
    disp('Input metrics_mt is empty. No subjects to plot.');
    return;
end

% --- Data Aggregation ---
all_subjects_session_info_rt_pro_anti = cell(num_subjects_to_process, 1);
all_subjects_session_info_rt_cong_incong = cell(num_subjects_to_process, 1);
subject_legend_labels = cell(1, num_subjects_to_process);
max_sessions_for_any_subject = 0;
for iS = 1:num_subjects_to_process
    if iS <= num_hardcoded_subject_details && ~isempty(subject_id_list_hardcoded{iS})
        subject_name_from_list = subject_id_list_hardcoded{iS};
        subject_legend_labels{iS} = subject_name_from_list; 
    else
        subject_legend_labels{iS} = ['Subject ' num2str(iS)];
    end
    subject_sessions_temp_rt_pa = []; 
    subject_sessions_temp_rt_ci = [];
    if isempty(metrics_mt{iS})
        fprintf('Data for Subject %s (Index %d) is empty, skipping.\n', subject_legend_labels{iS}, iS);
        continue;
    end
    num_sessions_this_subject = 0;
    for iD = 1:length(metrics_mt{iS}) 
        session_data = metrics_mt{iS}(iD);
        
        if ~isfield(session_data, 'RTMeanSE_Pro') || ~isfield(session_data, 'RTMeanSE_Anti') || ...
           ~isfield(session_data, 'RT_Cong_Combined') || ~isfield(session_data, 'dataset')
            continue;
        end
        if isempty(session_data.dataset)
			continue;
		end
        
        % Calculate Pro-Anti RT difference
        rt_pro_mean  = session_data.RTMeanSE_Pro(1);
        rt_anti_mean = session_data.RTMeanSE_Anti(1);
        session_diff_rt_pa = NaN;
        if ~isnan(rt_pro_mean) && ~isnan(rt_anti_mean)
            session_diff_rt_pa = rt_pro_mean - rt_anti_mean;
        end
        
        % Calculate Congruent-Incongruent RT difference
        rt_cong_combined = session_data.RT_Cong_Combined;
        session_rt_cong = NaN; session_rt_incong = NaN;
        session_diff_rt_ci = NaN;
        if size(rt_cong_combined,1) >= 4 && size(rt_cong_combined,2) >=4
            mean_rt_ll = rt_cong_combined(1,1); count_ll = rt_cong_combined(1,4);
            mean_rt_rr = rt_cong_combined(3,1); count_rr = rt_cong_combined(3,4);
            if (count_ll + count_rr) > 0
                 valid_ll = ~isnan(mean_rt_ll) && count_ll > 0; valid_rr = ~isnan(mean_rt_rr) && count_rr > 0;
                 if valid_ll && valid_rr; session_rt_cong = (mean_rt_ll * count_ll + mean_rt_rr * count_rr) / (count_ll + count_rr);
                 elseif valid_ll; session_rt_cong = mean_rt_ll; elseif valid_rr; session_rt_cong = mean_rt_rr; end
            end
            mean_rt_lr = rt_cong_combined(2,1); count_lr = rt_cong_combined(2,4);
            mean_rt_rl = rt_cong_combined(4,1); count_rl = rt_cong_combined(4,4);
            if (count_lr + count_rl) > 0
                 valid_lr = ~isnan(mean_rt_lr) && count_lr > 0; valid_rl = ~isnan(mean_rt_rl) && count_rl > 0;
                 if valid_lr && valid_rl; session_rt_incong = (mean_rt_lr * count_lr + mean_rt_rl * count_rl) / (count_lr + count_rl);
                 elseif valid_lr; session_rt_incong = mean_rt_lr; elseif valid_rl; session_rt_incong = mean_rt_rl; end
            end
        end
        if ~isnan(session_rt_cong) && ~isnan(session_rt_incong)
            session_diff_rt_ci = session_rt_cong - session_rt_incong;
        end
        
        session_datetime = NaT; 
        datetime_str_match = regexp(session_data.dataset, '(\d{2}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2})', 'once', 'match');
        if ~isempty(datetime_str_match)
            try 
                session_datetime = datetime(datetime_str_match, 'InputFormat', 'MM_dd_yy__HH_mm_ss', 'PivotYear', year(now)-50);
            catch ME_date
                fprintf('Warning: Could not parse datetime string "%s" for S%d Sess%d. Error: %s. Using index.\n', datetime_str_match, iS, iD, ME_date.message);
                session_datetime = datetime(2000,1,1) + days(num_sessions_this_subject); 
            end
        else
             session_datetime = datetime(2000,1,1) + days(num_sessions_this_subject); 
        end
        
        if ~isnat(session_datetime)
            num_sessions_this_subject = num_sessions_this_subject + 1; 
            if ~isnan(session_diff_rt_pa)
                subject_sessions_temp_rt_pa = [subject_sessions_temp_rt_pa; struct('datetime', session_datetime, 'value', session_diff_rt_pa, 'original_iD', iD)];
            end
            if ~isnan(session_diff_rt_ci)
                subject_sessions_temp_rt_ci = [subject_sessions_temp_rt_ci; struct('datetime', session_datetime, 'value', session_diff_rt_ci, 'original_iD', iD)];
            end
        end
    end 
    
    current_max_sessions = 0;
    if ~isempty(subject_sessions_temp_rt_pa)
        [~, sort_idx_pa] = sort([subject_sessions_temp_rt_pa.datetime]);
        all_subjects_session_info_rt_pro_anti{iS} = subject_sessions_temp_rt_pa(sort_idx_pa);
        current_max_sessions = max(current_max_sessions, length(subject_sessions_temp_rt_pa));
    end
    if ~isempty(subject_sessions_temp_rt_ci)
        [~, sort_idx_ci] = sort([subject_sessions_temp_rt_ci.datetime]);
        all_subjects_session_info_rt_cong_incong{iS} = subject_sessions_temp_rt_ci(sort_idx_ci);
         current_max_sessions = max(current_max_sessions, length(subject_sessions_temp_rt_ci));
    end
    if current_max_sessions > max_sessions_for_any_subject
        max_sessions_for_any_subject = current_max_sessions;
    end
end 
if max_sessions_for_any_subject == 0
    disp('No valid session data with RT differences found across all subjects.');
    return;
end
% --- Align Data into Matrices and Smooth ---
% Pro-Anti RT Differences
subject_rt_pa_diff_matrix_aligned = nan(num_subjects_to_process, max_sessions_for_any_subject);
for iS = 1:num_subjects_to_process
    if ~isempty(all_subjects_session_info_rt_pro_anti{iS})
        for k_sess = 1:length(all_subjects_session_info_rt_pro_anti{iS})
            subject_rt_pa_diff_matrix_aligned(iS, k_sess) = all_subjects_session_info_rt_pro_anti{iS}(k_sess).value;
        end
    end
end
subject_rt_pa_diff_matrix_smoothed = smoothdata(subject_rt_pa_diff_matrix_aligned, 2, 'movmean', smoothing_window_size, 'omitnan');
overall_avg_rt_pa_raw = nanmean(subject_rt_pa_diff_matrix_aligned, 1);
overall_se_rt_pa_raw  = nanstd(subject_rt_pa_diff_matrix_aligned, 0, 1) ./ sqrt(sum(~isnan(subject_rt_pa_diff_matrix_aligned), 1));
overall_avg_rt_pa_smoothed = smoothdata(overall_avg_rt_pa_raw, 'movmean', smoothing_window_size, 'omitnan');
% Congruent-Incongruent RT Differences
subject_rt_ci_diff_matrix_aligned = nan(num_subjects_to_process, max_sessions_for_any_subject);
for iS = 1:num_subjects_to_process
    if ~isempty(all_subjects_session_info_rt_cong_incong{iS})
        for k_sess = 1:length(all_subjects_session_info_rt_cong_incong{iS})
            subject_rt_ci_diff_matrix_aligned(iS, k_sess) = all_subjects_session_info_rt_cong_incong{iS}(k_sess).value;
        end
    end
end
subject_rt_ci_diff_matrix_smoothed = smoothdata(subject_rt_ci_diff_matrix_aligned, 2, 'movmean', smoothing_window_size, 'omitnan');
overall_avg_rt_ci_raw = nanmean(subject_rt_ci_diff_matrix_aligned, 1);
overall_se_rt_ci_raw  = nanstd(subject_rt_ci_diff_matrix_aligned, 0, 1) ./ sqrt(sum(~isnan(subject_rt_ci_diff_matrix_aligned), 1));
overall_avg_rt_ci_smoothed = smoothdata(overall_avg_rt_ci_raw, 'movmean', smoothing_window_size, 'omitnan');

% --- NEW: Regression Analysis ---
% Initialize variables to hold regression model results
mdl_pa = []; slope_pa = NaN; intercept_pa = NaN;
mdl_ci = []; slope_ci = NaN; intercept_ci = NaN;
x_vals_reg = (1:max_sessions_for_any_subject)'; % Use column vector for fitlm

if max_sessions_for_any_subject < 2
    disp('Skipping regression analysis: not enough session data points.');
else
    % --- Pro-Anti Overall Regression ---
    y_overall_pa = overall_avg_rt_pa_raw(:); % Ensure column
    valid_idx_pa = ~isnan(y_overall_pa);
    
        try
            mdl_pa = fitlm(x_vals_reg(valid_idx_pa), y_overall_pa(valid_idx_pa));
            coeffs_pa = mdl_pa.Coefficients.Estimate;
            intercept_pa = coeffs_pa(1);
            slope_pa = coeffs_pa(2);
            
            fprintf('\n--- Overall Pro-Anti RT Difference Regression ---\n');
            fprintf('  y = %.4f * x + %.4f\n', slope_pa, intercept_pa);
            fprintf('  R-squared: %.4f\n', mdl_pa.Rsquared.Ordinary);
            fprintf('  p-value (for slope): %.4f\n', mdl_pa.Coefficients.pValue(2));
            if mdl_pa.Coefficients.pValue(2) < 0.05
                fprintf('  -> The trend is statistically significant.\n');
            else
                fprintf('  -> The trend is not statistically significant.\n');
            end
        catch ME_fitlm
            fprintf('Could not perform linear regression for Pro-Anti: %s\n', ME_fitlm.message);
        end
    

    % --- Cong-Incong Overall Regression ---
    y_overall_ci = overall_avg_rt_ci_raw(:); % Ensure column
    valid_idx_ci = ~isnan(y_overall_ci);
    
        try
            mdl_ci = fitlm(x_vals_reg(valid_idx_ci), y_overall_ci(valid_idx_ci));
            coeffs_ci = mdl_ci.Coefficients.Estimate;
            intercept_ci = coeffs_ci(1);
            slope_ci = coeffs_ci(2);

            fprintf('\n--- Overall Congruent-Incongruent RT Difference Regression ---\n');
            fprintf('  y = %.4f * x + %.4f\n', slope_ci, intercept_ci);
            fprintf('  R-squared: %.4f\n', mdl_ci.Rsquared.Ordinary);
            fprintf('  p-value (for slope): %.4f\n', mdl_ci.Coefficients.pValue(2));
            if mdl_ci.Coefficients.pValue(2) < 0.05
                fprintf('  -> The trend is statistically significant.\n');
            else
                fprintf('  -> The trend is not statistically significant.\n');
            end
        catch ME_fitlm
            fprintf('Could not perform linear regression for Cong-Incong: %s\n', ME_fitlm.message);
        end
    
end
fprintf('\n'); % Add a newline for cleaner separation before plot messages

% --- Plotting ---
screen_size = get(0, 'ScreenSize');
fig_pos_x = (screen_size(3) - figure_width) / 2;
fig_pos_y = (screen_size(4) - figure_height) / 2;
fig_handle = figure('Position', [fig_pos_x, fig_pos_y, figure_width, figure_height], 'Color', 'w');
x_axis_values = 1:max_sessions_for_any_subject;
subject_legend_handles = gobjects(num_subjects_to_process, 1);
subject_has_legend_entry = false(num_subjects_to_process, 1);
% Subplot 1: Pro-Anti RT Difference
ax1 = subplot(2, 1, 1);
hold(ax1, 'on');
yline(ax1, 0, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5, 'HandleVisibility','off'); % Zero line
upper_bound_rt_pa = overall_avg_rt_pa_smoothed + overall_se_rt_pa_raw;
lower_bound_rt_pa = overall_avg_rt_pa_smoothed - overall_se_rt_pa_raw;
valid_fill_rt_pa = ~isnan(overall_avg_rt_pa_smoothed) & ~isnan(upper_bound_rt_pa) & ~isnan(lower_bound_rt_pa);
if any(valid_fill_rt_pa)
    fill_x_pa = x_axis_values(valid_fill_rt_pa);
    fill([fill_x_pa, fliplr(fill_x_pa)], [upper_bound_rt_pa(valid_fill_rt_pa), fliplr(lower_bound_rt_pa(valid_fill_rt_pa))], ...
         [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'Parent', ax1, 'DisplayName', 'Overall SE');
end
for iS = 1:num_subjects_to_process
    if ~isempty(all_subjects_session_info_rt_pro_anti{iS}) 
        subj_data_to_plot = subject_rt_pa_diff_matrix_smoothed(iS, :);
        valid_data_indices = ~isnan(subj_data_to_plot);
        if any(valid_data_indices)
            subj_line_color = [];
            if iS <= num_defined_custom_colors; subj_line_color = custom_colors_rgb{iS};
            else 
                cc = lines(num_subjects_to_process);
                color_idx_cycle = mod(iS - 1 - num_defined_custom_colors, size(cc,1)) + 1;
                subj_line_color = cc(color_idx_cycle,:);
            end
            h_line = plot(ax1, x_axis_values(valid_data_indices), subj_data_to_plot(valid_data_indices), ...
                          '-', 'LineWidth', 1, 'Color', subj_line_color); 
            if ~subject_has_legend_entry(iS) 
                subject_legend_handles(iS) = h_line;
                subject_has_legend_entry(iS) = true;
            end
        end
    end
end
if any(valid_fill_rt_pa) 
    plot(ax1, x_axis_values(valid_fill_rt_pa), overall_avg_rt_pa_smoothed(valid_fill_rt_pa), ...
         'k', 'LineWidth', 2, 'DisplayName', 'Overall Avg (Smoothed)'); 
end
% NEW: Plot regression line
if ~isempty(mdl_pa)
    reg_line_y = predict(mdl_pa, x_axis_values');
    plot(ax1, x_axis_values, reg_line_y, '--r', 'LineWidth', 2, 'DisplayName', 'Overall Regression');
end
hold(ax1, 'off'); 
title(ax1, 'Pro-Saccade vs. Anti-Saccade RT Difference', 'FontSize', plot_font_size + 1);
ylabel(ax1, 'RT Diff. (s) (Pro - Anti)', 'FontSize', plot_font_size);
set(ax1, 'FontSize', plot_font_size);

hold on
% Subplot 2: Congruent-Incongruent RT Difference
ax2 = subplot(2, 1, 2);
hold(ax2, 'on');
yline(ax2, 0, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5, 'HandleVisibility','off'); % Zero line
upper_bound_rt_ci = overall_avg_rt_ci_smoothed + overall_se_rt_ci_raw;
lower_bound_rt_ci = overall_avg_rt_ci_smoothed - overall_se_rt_ci_raw;
valid_fill_rt_ci = ~isnan(overall_avg_rt_ci_smoothed) & ~isnan(upper_bound_rt_ci) & ~isnan(lower_bound_rt_ci);
if any(valid_fill_rt_ci)
    fill_x_ci = x_axis_values(valid_fill_rt_ci);
    fill([fill_x_ci, fliplr(fill_x_ci)], [upper_bound_rt_ci(valid_fill_rt_ci), fliplr(lower_bound_rt_ci(valid_fill_rt_ci))], ...
         [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'Parent', ax2);
end
for iS = 1:num_subjects_to_process
    if ~isempty(all_subjects_session_info_rt_cong_incong{iS}) 
        subj_data_to_plot = subject_rt_ci_diff_matrix_smoothed(iS, :);
        valid_data_indices = ~isnan(subj_data_to_plot);
        if any(valid_data_indices)
            subj_line_color = []; 
            if iS <= num_defined_custom_colors; subj_line_color = custom_colors_rgb{iS};
            else
                cc = lines(num_subjects_to_process);
                color_idx_cycle = mod(iS - 1 - num_defined_custom_colors, size(cc,1)) + 1;
                subj_line_color = cc(color_idx_cycle,:);
            end
            plot(ax2, x_axis_values(valid_data_indices), subj_data_to_plot(valid_data_indices), ...
                 '-', 'LineWidth', 1, 'Color', subj_line_color); 
        end
    end
end
if any(valid_fill_rt_ci) 
    plot(ax2, x_axis_values(valid_fill_rt_ci), overall_avg_rt_ci_smoothed(valid_fill_rt_ci), ...
         'k', 'LineWidth', 2); 
end
% NEW: Plot regression line
if ~isempty(mdl_ci)
    reg_line_y = predict(mdl_ci, x_axis_values');
    plot(ax2, x_axis_values, reg_line_y, '--r', 'LineWidth', 2);
end
hold(ax2, 'off'); 
title(ax2, 'Congruent vs. Incongruent Simon RT Difference', 'FontSize', plot_font_size + 1);
ylabel(ax2, 'RT Diff. (s) (Cong. - Incong.)', 'FontSize', plot_font_size);
xlabel(ax2, 'Session Number (Chronological)', 'FontSize', plot_font_size); 
set(ax2, 'FontSize', plot_font_size);

% --- Common Aesthetics for X-axis and Y-limits ---
all_axes = [ax1, ax2];
common_y_min = Inf; common_y_max = -Inf;
for i=1:length(all_axes)
    current_ax = all_axes(i);
    plotted_elements = get(current_ax, 'Children');
    for k=1:length(plotted_elements)
        if isprop(plotted_elements(k), 'YData')
            y_data = get(plotted_elements(k), 'YData');
            if ~isempty(y_data) && isnumeric(y_data)
                y_data_finite = y_data(~isinf(y_data) & ~isnan(y_data));
                if ~isempty(y_data_finite)
                    common_y_min = min(common_y_min, min(y_data_finite));
                    common_y_max = max(common_y_max, max(y_data_finite));
                end
            end
        end
    end
end
if isinf(common_y_min) && isinf(common_y_max); common_y_min = -0.1; common_y_max = 0.1; end
y_padding = (common_y_max - common_y_min) * 0.10;
if y_padding < 0.02 || isnan(y_padding) || y_padding == 0; y_padding = 0.05; end 
final_ylim_bottom = common_y_min - y_padding;
final_ylim_top = common_y_max + y_padding;
for i=1:length(all_axes)
    current_ax = all_axes(i);
    if max_sessions_for_any_subject > 0
        custom_ticks = 1:5:max_sessions_for_any_subject;
        if ~ismember(1, custom_ticks); custom_ticks = [1 custom_ticks]; end
        if ~ismember(max_sessions_for_any_subject, custom_ticks); custom_ticks = [custom_ticks max_sessions_for_any_subject]; end
        set(current_ax, 'XTick', unique(custom_ticks));
        xlim(current_ax, [0.5 max_sessions_for_any_subject + 0.5]);
    else 
        set(current_ax, 'XTick', [1]); xlim(current_ax, [0.5 1.5]); 
    end
    ylim(current_ax, [final_ylim_bottom, final_ylim_top]); 
    grid(current_ax, 'off'); 
    if i == length(all_axes) && max_sessions_for_any_subject > 1
        current_plot_ylim = get(current_ax, 'YLim');
        text_y_pos = current_plot_ylim(1) - 0.12 * diff(current_plot_ylim); 
        text(current_ax, 1, text_y_pos, 'Early', 'HorizontalAlignment', 'center');
        text(current_ax, max_sessions_for_any_subject, text_y_pos, 'Late', 'HorizontalAlignment', 'center');
    end
end

% --- Legend ---
handles_for_actual_legend = subject_legend_handles(subject_has_legend_entry);
labels_for_actual_legend = subject_legend_labels(subject_has_legend_entry);
if ~isempty(handles_for_actual_legend)
    % Add dummy plots for overall average and regression to add them to the legend
    h_overall_dummy = plot(ax1, NaN, NaN, 'k', 'LineWidth', 2, 'DisplayName', 'Overall Avg.'); 
    h_regr_dummy = plot(ax1, NaN, NaN, '--r', 'LineWidth', 2, 'DisplayName', 'Overall Regression');
    
    handles_for_legend = [h_overall_dummy; h_regr_dummy; handles_for_actual_legend(:)];
    labels_for_legend = [{'Overall Avg.'; 'Overall Regression'}; labels_for_actual_legend(:)];

    % lgd = legend(ax1, handles_for_legend, labels_for_legend, 'Location', 'eastoutside', 'FontSize', plot_font_size - 2, 'Box', 'off');
    
    % This legend will only be visible on the top plot, which is common practice.
    % To have it affect both, we would need a single legend for the figure.
    
    delete([h_overall_dummy, h_regr_dummy]); % Clean up dummy handles
end

% --- Saving Outputs ---
base_filename = 'as_rt_diffs_over_sessions'; 
date_str = datestr(now, 'yyyymmdd_HHMM');
full_base_filename = [base_filename '_' date_str];
save_folder = 'AS_Figures_RT_TimeSeries'; 
if ~exist(save_folder, 'dir'); mkdir(save_folder); end
filepath_base = fullfile(save_folder, full_base_filename);
% Save Figure
try
    print(fig_handle, [filepath_base '.png'], '-dpng', '-r300');
    fprintf('Figure saved as: %s.png\n', filepath_base);
    print(fig_handle, [filepath_base '.eps'], '-depsc', '-vector');
    fprintf('Figure saved as: %s.eps\n', filepath_base);
catch ME_save
    fprintf('Error saving figure: %s\n', ME_save.message);
end

% Save Data to CSVs
try
    num_sessions = max_sessions_for_any_subject;
    valid_subjects_idx = find(subject_has_legend_entry);
    valid_subject_labels = subject_legend_labels(valid_subjects_idx);
    session_headers = arrayfun(@(x) sprintf('Session_%d', x), 1:num_sessions, 'UniformOutput', false);

    if any(~isnan(subject_rt_pa_diff_matrix_aligned(:)))
        T_pa = array2table([overall_avg_rt_pa_smoothed; overall_se_rt_pa_raw; subject_rt_pa_diff_matrix_smoothed(valid_subjects_idx, :)], ...
            'VariableNames', session_headers, 'RowNames', [{'Overall_Average_Smoothed'; 'Overall_SE_Raw'}; valid_subject_labels(:)]);
        writetable(T_pa, [filepath_base '_rt_pro_anti_diff.csv'], 'WriteRowNames', true);
        fprintf('Pro-Anti RT difference data saved to: %s_rt_pro_anti_diff.csv\n', full_base_filename);
    end

    if any(~isnan(subject_rt_ci_diff_matrix_aligned(:)))
        T_ci = array2table([overall_avg_rt_ci_smoothed; overall_se_rt_ci_raw; subject_rt_ci_diff_matrix_smoothed(valid_subjects_idx, :)], ...
             'VariableNames', session_headers, 'RowNames', [{'Overall_Average_Smoothed'; 'Overall_SE_Raw'}; valid_subject_labels(:)]);
        writetable(T_ci, [filepath_base '_rt_cong_incong_diff.csv'], 'WriteRowNames', true);
        fprintf('Cong-Incong RT difference data saved to: %s_rt_cong_incong_diff.csv\n', full_base_filename);
    end

    % --- MODIFIED: Save Regression Data to CSV with new stats ---
    if num_sessions >= 2
        regression_results = cell(0, 6); % Expanded for R2 and p-value
        % Overall Pro-Anti
        if ~isempty(mdl_pa)
            regression_results = [regression_results; {'Pro-Anti_RT_Diff', 'Overall_Average', slope_pa, intercept_pa, mdl_pa.Rsquared.Ordinary, mdl_pa.Coefficients.pValue(2)}];
        end
        % Overall Cong-Incong
        if ~isempty(mdl_ci)
            regression_results = [regression_results; {'Cong-Incong_RT_Diff', 'Overall_Average', slope_ci, intercept_ci, mdl_ci.Rsquared.Ordinary, mdl_ci.Coefficients.pValue(2)}];
        end
        % Individual Subjects (with NaN for detailed stats)
        x_vals_subj = 1:num_sessions;
        for i = 1:length(valid_subjects_idx)
            iS = valid_subjects_idx(i);
            % Pro-Anti
            y_subj_pa = subject_rt_pa_diff_matrix_aligned(iS, :);
            valid_idx_subj_pa = ~isnan(y_subj_pa);
            if sum(valid_idx_subj_pa) >= 2
                p = polyfit(x_vals_subj(valid_idx_subj_pa), y_subj_pa(valid_idx_subj_pa), 1);
                regression_results = [regression_results; {'Pro-Anti_RT_Diff', subject_legend_labels{iS}, p(1), p(2), NaN, NaN}];
            end
            % Cong-Incong
            y_subj_ci = subject_rt_ci_diff_matrix_aligned(iS, :);
            valid_idx_subj_ci = ~isnan(y_subj_ci);
            if sum(valid_idx_subj_ci) >= 2
                p = polyfit(x_vals_subj(valid_idx_subj_ci), y_subj_ci(valid_idx_subj_ci), 1);
                regression_results = [regression_results; {'Cong-Incong_RT_Diff', subject_legend_labels{iS}, p(1), p(2), NaN, NaN}];
            end
        end

        if ~isempty(regression_results)
            regression_table = cell2table(regression_results, 'VariableNames', {'AnalysisType', 'Series', 'Slope', 'Intercept', 'R_Squared', 'P_Value'});
            writetable(regression_table, [filepath_base '_regression_stats.csv']);
            fprintf('Regression statistics saved to: %s_regression_stats.csv\n', full_base_filename);
        end
    end
catch ME_csv
    fprintf('Error saving CSV data: %s\n', ME_csv.message);
end

end

function plot_AS_diffs_over_sessions_time(metrics_mt)
% PLOT_AS_DIFFS_OVER_SESSIONS_TIME Plots Pro-Anti and Congruent-Incongruent
% accuracy differences over chronologically ordered sessions.
%
% V3 Changes:
%   - Adds a linear regression line (red, dashed) for the overall data trend.
%   - Prints detailed regression statistics (R-squared, p-value) to the console.
%   - Exports regression statistics to a CSV file.
%
% Features:
%   - Two subplots: Top for Pro-Anti diff, Bottom for Cong-Incong diff.
%   - X-axis: Session number, ordered by datetime.
%   - Y-axis: Accuracy Difference.
%   - Each subject plotted as a colored line (smoothed).
%   - Overall average plotted as a black line (smoothed) with shaded SE.
%   - Figure size 800x800.
%
% Args:
%   metrics_mt (cell array): Data structure.
%
% Requires: Statistics and Machine Learning Toolbox for fitlm().

% --- Configuration ---
plot_font_size = 14;
figure_width = 800;
figure_height = 800; 
smoothing_window_size = 5; 
subject_id_list_hardcoded = { 
    'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan' 
};
num_hardcoded_subject_details = length(subject_id_list_hardcoded);
custom_colors_rgb = {
    [245, 124, 110]/255; [242, 181, 110]/255; [251, 231, 158]/255;
    [132, 195, 183]/255; [136, 215, 218]/255; [113, 184, 237]/255;
    [184, 174, 234]/255; [242, 168, 218]/255
};
num_defined_custom_colors = length(custom_colors_rgb);
num_subjects_to_process = length(metrics_mt);
if num_subjects_to_process == 0
    disp('Input metrics_mt is empty. No subjects to plot.');
    return;
end
% --- Data Aggregation ---
all_subjects_session_info_pro_anti = cell(num_subjects_to_process, 1);
all_subjects_session_info_cong_incong = cell(num_subjects_to_process, 1);
subject_legend_labels = cell(1, num_subjects_to_process);
max_sessions_for_any_subject = 0;
for iS = 1:num_subjects_to_process
    if iS <= num_hardcoded_subject_details && ~isempty(subject_id_list_hardcoded{iS})
        subject_name_from_list = subject_id_list_hardcoded{iS};
        subject_legend_labels{iS} = subject_name_from_list; 
    else
        subject_legend_labels{iS} = ['Subject ' num2str(iS)];
    end
    subject_sessions_temp_pa = []; 
    subject_sessions_temp_ci = [];
    if isempty(metrics_mt{iS})
        fprintf('Data for Subject %s (Index %d) is empty, skipping.\n', subject_legend_labels{iS}, iS);
        continue;
    end
    num_sessions_this_subject = 0;
    for iD = 1:length(metrics_mt{iS}) 
        session_data = metrics_mt{iS}(iD);
        
        if ~isfield(session_data, 'AccuracyMeanSE_Pro') || ~isfield(session_data, 'AccuracyMeanSE_Anti') || ...
           ~isfield(session_data, 'Accuracy_Cong_Combined') || ~isfield(session_data, 'dataset')
            continue;
        end
        if isempty(session_data.dataset)
			continue;
		end
        
        acc_pro_mean  = session_data.AccuracyMeanSE_Pro(1);
        acc_anti_mean = session_data.AccuracyMeanSE_Anti(1);
        session_diff_pa = NaN;
        if ~isnan(acc_pro_mean) && ~isnan(acc_anti_mean)
            session_diff_pa = acc_pro_mean - acc_anti_mean;
        end
        acc_cong_combined = session_data.Accuracy_Cong_Combined;
        session_acc_cong = NaN; session_acc_incong = NaN;
        session_diff_ci = NaN;
        if size(acc_cong_combined,1) >= 4 && size(acc_cong_combined,2) >=4
            mean_ll = acc_cong_combined(1,1); count_ll = acc_cong_combined(1,4);
            mean_rr = acc_cong_combined(3,1); count_rr = acc_cong_combined(3,4);
            if (count_ll + count_rr) > 0
                 valid_ll = ~isnan(mean_ll) && count_ll > 0; valid_rr = ~isnan(mean_rr) && count_rr > 0;
                 if valid_ll && valid_rr; session_acc_cong = (mean_ll * count_ll + mean_rr * count_rr) / (count_ll + count_rr);
                 elseif valid_ll; session_acc_cong = mean_ll; elseif valid_rr; session_acc_cong = mean_rr; end
            end
            mean_lr = acc_cong_combined(2,1); count_lr = acc_cong_combined(2,4);
            mean_rl = acc_cong_combined(4,1); count_rl = acc_cong_combined(4,4);
            if (count_lr + count_rl) > 0
                 valid_lr = ~isnan(mean_lr) && count_lr > 0; valid_rl = ~isnan(mean_rl) && count_rl > 0;
                 if valid_lr && valid_rl; session_acc_incong = (mean_lr * count_lr + mean_rl * count_rl) / (count_lr + count_rl);
                 elseif valid_lr; session_acc_incong = mean_lr; elseif valid_rl; session_acc_incong = mean_rl; end
            end
        end
        if ~isnan(session_acc_cong) && ~isnan(session_acc_incong)
            session_diff_ci = session_acc_cong - session_acc_incong;
        end
        
        session_datetime = NaT; 
        datetime_str_match = regexp(session_data.dataset, '(\d{2}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2})', 'once', 'match');
        if ~isempty(datetime_str_match)
            try 
                session_datetime = datetime(datetime_str_match, 'InputFormat', 'MM_dd_yy__HH_mm_ss', 'PivotYear', year(now)-50);
            catch ME_date
                fprintf('Warning: Could not parse datetime string "%s" for S%d Sess%d. Error: %s. Using index.\n', datetime_str_match, iS, iD, ME_date.message);
                session_datetime = datetime(2000,1,1) + days(num_sessions_this_subject); 
            end
        else
             session_datetime = datetime(2000,1,1) + days(num_sessions_this_subject); 
        end
        
        if ~isnat(session_datetime)
            num_sessions_this_subject = num_sessions_this_subject + 1; 
            if ~isnan(session_diff_pa)
                subject_sessions_temp_pa = [subject_sessions_temp_pa; struct('datetime', session_datetime, 'value', session_diff_pa, 'original_iD', iD)];
            end
            if ~isnan(session_diff_ci)
                subject_sessions_temp_ci = [subject_sessions_temp_ci; struct('datetime', session_datetime, 'value', session_diff_ci, 'original_iD', iD)];
            end
        end
    end 
    
    current_max_sessions = 0;
    if ~isempty(subject_sessions_temp_pa)
        [~, sort_idx_pa] = sort([subject_sessions_temp_pa.datetime]);
        all_subjects_session_info_pro_anti{iS} = subject_sessions_temp_pa(sort_idx_pa);
        current_max_sessions = max(current_max_sessions, length(subject_sessions_temp_pa));
    end
    if ~isempty(subject_sessions_temp_ci)
        [~, sort_idx_ci] = sort([subject_sessions_temp_ci.datetime]);
        all_subjects_session_info_cong_incong{iS} = subject_sessions_temp_ci(sort_idx_ci);
         current_max_sessions = max(current_max_sessions, length(subject_sessions_temp_ci));
    end
    if current_max_sessions > max_sessions_for_any_subject
        max_sessions_for_any_subject = current_max_sessions;
    end
end 
if max_sessions_for_any_subject == 0
    disp('No valid session data with accuracy differences found across all subjects.');
    return;
end
% --- Align Data into Matrices and Smooth ---
subject_pa_diff_matrix_aligned = nan(num_subjects_to_process, max_sessions_for_any_subject);
for iS = 1:num_subjects_to_process
    if ~isempty(all_subjects_session_info_pro_anti{iS})
        for k_sess = 1:length(all_subjects_session_info_pro_anti{iS})
            subject_pa_diff_matrix_aligned(iS, k_sess) = all_subjects_session_info_pro_anti{iS}(k_sess).value;
        end
    end
end
subject_pa_diff_matrix_smoothed = smoothdata(subject_pa_diff_matrix_aligned, 2, 'movmean', smoothing_window_size, 'omitnan');
overall_avg_pa_raw = nanmean(subject_pa_diff_matrix_aligned, 1);
overall_se_pa_raw  = nanstd(subject_pa_diff_matrix_aligned, 0, 1) ./ sqrt(sum(~isnan(subject_pa_diff_matrix_aligned), 1));
overall_avg_pa_smoothed = smoothdata(overall_avg_pa_raw, 'movmean', smoothing_window_size, 'omitnan');
subject_ci_diff_matrix_aligned = nan(num_subjects_to_process, max_sessions_for_any_subject);
for iS = 1:num_subjects_to_process
    if ~isempty(all_subjects_session_info_cong_incong{iS})
        for k_sess = 1:length(all_subjects_session_info_cong_incong{iS})
            subject_ci_diff_matrix_aligned(iS, k_sess) = all_subjects_session_info_cong_incong{iS}(k_sess).value;
        end
    end
end
subject_ci_diff_matrix_smoothed = smoothdata(subject_ci_diff_matrix_aligned, 2, 'movmean', smoothing_window_size, 'omitnan');
overall_avg_ci_raw = nanmean(subject_ci_diff_matrix_aligned, 1);
overall_se_ci_raw  = nanstd(subject_ci_diff_matrix_aligned, 0, 1) ./ sqrt(sum(~isnan(subject_ci_diff_matrix_aligned), 1));
overall_avg_ci_smoothed = smoothdata(overall_avg_ci_raw, 'movmean', smoothing_window_size, 'omitnan');

% --- NEW: Regression Analysis ---
mdl_pa = []; slope_pa = NaN; intercept_pa = NaN;
mdl_ci = []; slope_ci = NaN; intercept_ci = NaN;
x_vals_reg = (1:max_sessions_for_any_subject)'; 

if max_sessions_for_any_subject < 2
    disp('Skipping regression analysis: not enough session data points.');
else
    % Pro-Anti Overall Regression
    y_overall_pa = overall_avg_pa_raw(:);
    valid_idx_pa = ~isnan(y_overall_pa);
    if sum(valid_idx_pa) >= 2
        try
            mdl_pa = fitlm(x_vals_reg(valid_idx_pa), y_overall_pa(valid_idx_pa));
            coeffs_pa = mdl_pa.Coefficients.Estimate;
            intercept_pa = coeffs_pa(1);
            slope_pa = coeffs_pa(2);
            fprintf('\n--- Overall Pro-Anti Accuracy Difference Regression ---\n');
            fprintf('  y = %.4f * x + %.4f\n', slope_pa, intercept_pa);
            fprintf('  R-squared: %.4f\n', mdl_pa.Rsquared.Ordinary);
            fprintf('  p-value (for slope): %.4f\n', mdl_pa.Coefficients.pValue(2));
            if mdl_pa.Coefficients.pValue(2) < 0.05
                fprintf('  -> The trend is statistically significant.\n');
            else
                fprintf('  -> The trend is not statistically significant.\n');
            end
        catch ME_fitlm
            fprintf('Could not perform linear regression for Pro-Anti: %s\n', ME_fitlm.message);
        end
    end

    % Cong-Incong Overall Regression
    y_overall_ci = overall_avg_ci_raw(:);
    valid_idx_ci = ~isnan(y_overall_ci);
    if sum(valid_idx_ci) >= 2
        try
            mdl_ci = fitlm(x_vals_reg(valid_idx_ci), y_overall_ci(valid_idx_ci));
            coeffs_ci = mdl_ci.Coefficients.Estimate;
            intercept_ci = coeffs_ci(1);
            slope_ci = coeffs_ci(2);
            fprintf('\n--- Overall Congruent-Incongruent Accuracy Difference Regression ---\n');
            fprintf('  y = %.4f * x + %.4f\n', slope_ci, intercept_ci);
            fprintf('  R-squared: %.4f\n', mdl_ci.Rsquared.Ordinary);
            fprintf('  p-value (for slope): %.4f\n', mdl_ci.Coefficients.pValue(2));
            if mdl_ci.Coefficients.pValue(2) < 0.05
                fprintf('  -> The trend is statistically significant.\n');
            else
                fprintf('  -> The trend is not statistically significant.\n');
            end
        catch ME_fitlm
            fprintf('Could not perform linear regression for Cong-Incong: %s\n', ME_fitlm.message);
        end
    end
end
fprintf('\n'); % Formatting space

% --- Plotting ---
screen_size = get(0, 'ScreenSize');
fig_pos_x = (screen_size(3) - figure_width) / 2;
fig_pos_y = (screen_size(4) - figure_height) / 2;
fig_handle = figure('Position', [fig_pos_x, fig_pos_y, figure_width, figure_height], 'Color', 'w');
x_axis_values = 1:max_sessions_for_any_subject;
subject_legend_handles = gobjects(num_subjects_to_process, 1);
subject_has_legend_entry = false(num_subjects_to_process, 1);

% Subplot 1: Pro-Anti Difference
ax1 = subplot(2, 1, 1);
hold(ax1, 'on');
yline(ax1, 0, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5, 'HandleVisibility','off');
upper_bound_pa = overall_avg_pa_smoothed + overall_se_pa_raw;
lower_bound_pa = overall_avg_pa_smoothed - overall_se_pa_raw;
valid_fill_pa = ~isnan(overall_avg_pa_smoothed) & ~isnan(upper_bound_pa) & ~isnan(lower_bound_pa);
if any(valid_fill_pa)
    fill_x_pa = x_axis_values(valid_fill_pa);
    fill([fill_x_pa, fliplr(fill_x_pa)], [upper_bound_pa(valid_fill_pa), fliplr(lower_bound_pa(valid_fill_pa))], ...
         [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'Parent', ax1);
end
for iS = 1:num_subjects_to_process
    if ~isempty(all_subjects_session_info_pro_anti{iS}) 
        subj_data_to_plot = subject_pa_diff_matrix_smoothed(iS, :);
        valid_data_indices = ~isnan(subj_data_to_plot);
        if any(valid_data_indices)
            subj_line_color = [];
            if iS <= num_defined_custom_colors; subj_line_color = custom_colors_rgb{iS};
            else 
                cc = lines(num_subjects_to_process);
                color_idx_cycle = mod(iS - 1 - num_defined_custom_colors, size(cc,1)) + 1;
                subj_line_color = cc(color_idx_cycle,:);
            end
            h_line = plot(ax1, x_axis_values(valid_data_indices), subj_data_to_plot(valid_data_indices), ...
                          '-', 'LineWidth', 1, 'Color', subj_line_color); 
            if ~subject_has_legend_entry(iS) 
                subject_legend_handles(iS) = h_line;
                subject_has_legend_entry(iS) = true;
            end
        end
    end
end
if any(valid_fill_pa)
    plot(ax1, x_axis_values(valid_fill_pa), overall_avg_pa_smoothed(valid_fill_pa), 'k', 'LineWidth', 2);
end
% NEW: Plot regression line
if ~isempty(mdl_pa)
    reg_line_y = predict(mdl_pa, x_axis_values');
    plot(ax1, x_axis_values, reg_line_y, '--r', 'LineWidth', 2);
end
hold(ax1, 'off');
title(ax1, 'Pro-Saccade vs. Anti-Saccade Accuracy Difference', 'FontSize', plot_font_size + 1);
ylabel(ax1, 'Accuracy Diff. (Pro - Anti)', 'FontSize', plot_font_size);
set(ax1, 'FontSize', plot_font_size);
hold on
% Subplot 2: Congruent-Incongruent Difference
ax2 = subplot(2, 1, 2);
hold(ax2, 'on');
yline(ax2, 0, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5, 'HandleVisibility','off');
upper_bound_ci = overall_avg_ci_smoothed + overall_se_ci_raw;
lower_bound_ci = overall_avg_ci_smoothed - overall_se_ci_raw;
valid_fill_ci = ~isnan(overall_avg_ci_smoothed) & ~isnan(upper_bound_ci) & ~isnan(lower_bound_ci);
if any(valid_fill_ci)
    fill_x_ci = x_axis_values(valid_fill_ci);
    fill([fill_x_ci, fliplr(fill_x_ci)], [upper_bound_ci(valid_fill_ci), fliplr(lower_bound_ci(valid_fill_ci))], ...
         [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'Parent', ax2);
end
for iS = 1:num_subjects_to_process
    if ~isempty(all_subjects_session_info_cong_incong{iS}) 
        subj_data_to_plot = subject_ci_diff_matrix_smoothed(iS, :);
        valid_data_indices = ~isnan(subj_data_to_plot);
        if any(valid_data_indices)
            subj_line_color = []; 
            if iS <= num_defined_custom_colors; subj_line_color = custom_colors_rgb{iS};
            else
                cc = lines(num_subjects_to_process);
                color_idx_cycle = mod(iS - 1 - num_defined_custom_colors, size(cc,1)) + 1;
                subj_line_color = cc(color_idx_cycle,:);
            end
            plot(ax2, x_axis_values(valid_data_indices), subj_data_to_plot(valid_data_indices), ...
                 '-', 'LineWidth', 1, 'Color', subj_line_color);
        end
    end
end
if any(valid_fill_ci)
    plot(ax2, x_axis_values(valid_fill_ci), overall_avg_ci_smoothed(valid_fill_ci), 'k', 'LineWidth', 2);
end
% NEW: Plot regression line
if ~isempty(mdl_ci)
    reg_line_y = predict(mdl_ci, x_axis_values');
    plot(ax2, x_axis_values, reg_line_y, '--r', 'LineWidth', 2);
end
hold(ax2, 'off');
title(ax2, 'Congruent vs. Incongruent Simon Accuracy Difference', 'FontSize', plot_font_size + 1);
ylabel(ax2, 'Accuracy Diff. (Cong. - Incong.)', 'FontSize', plot_font_size);
xlabel(ax2, 'Session Number (Chronological)', 'FontSize', plot_font_size);
set(ax2, 'FontSize', plot_font_size);

% --- Common Aesthetics ---
all_axes = [ax1, ax2];
common_y_min = Inf; common_y_max = -Inf;
for i=1:length(all_axes)
    y_lims = get(all_axes(i), 'YLim');
    common_y_min = min(common_y_min, y_lims(1));
    common_y_max = max(common_y_max, y_lims(2));
end
y_padding = (common_y_max - common_y_min) * 0.10;
final_ylim = [common_y_min - y_padding, common_y_max + y_padding];
set(all_axes, 'YLim', final_ylim);
for i=1:length(all_axes)
    if max_sessions_for_any_subject > 0
        custom_ticks = 1:5:max_sessions_for_any_subject;
        if ~ismember(1, custom_ticks); custom_ticks = [1 custom_ticks]; end
        if ~ismember(max_sessions_for_any_subject, custom_ticks); custom_ticks = [custom_ticks max_sessions_for_any_subject]; end
        set(all_axes(i), 'XTick', unique(custom_ticks), 'XLim', [0.5 max_sessions_for_any_subject + 0.5]);
    end
    if i == length(all_axes) && max_sessions_for_any_subject > 1
        current_plot_ylim = get(all_axes(i), 'YLim');
        text_y_pos = current_plot_ylim(1) - 0.12 * diff(current_plot_ylim); 
        text(all_axes(i), 1, text_y_pos, 'Early', 'HorizontalAlignment', 'center');
        text(all_axes(i), max_sessions_for_any_subject, text_y_pos, 'Late', 'HorizontalAlignment', 'center');
    end
end

% --- Legend ---
handles_for_actual_legend = subject_legend_handles(subject_has_legend_entry);
labels_for_actual_legend = subject_legend_labels(subject_has_legend_entry);
if ~isempty(handles_for_actual_legend)
    h_overall_dummy = plot(ax1, NaN, NaN, 'k', 'LineWidth', 2, 'DisplayName', 'Overall Avg.');
    h_regr_dummy = plot(ax1, NaN, NaN, '--r', 'LineWidth', 2, 'DisplayName', 'Overall Regression');
    handles_for_legend = [h_overall_dummy; h_regr_dummy; handles_for_actual_legend(:)];
    labels_for_legend = [{'Overall Avg.'; 'Overall Regression'}; labels_for_actual_legend(:)];
    % lgd = legend(ax1, handles_for_legend, labels_for_legend, 'Location', 'eastoutside', 'FontSize', plot_font_size - 2, 'Box', 'off');
    delete([h_overall_dummy, h_regr_dummy]);
end

% --- Saving Outputs ---
base_filename = 'as_accuracy_diffs_over_sessions';
date_str = datestr(now, 'yyyymmdd_HHMM');
full_base_filename = [base_filename '_' date_str];
save_folder = 'AS_Figures_TimeSeries';
if ~exist(save_folder, 'dir'); mkdir(save_folder); end
filepath_base = fullfile(save_folder, full_base_filename);
% Save Figure
try
    print(fig_handle, [filepath_base '.png'], '-dpng', '-r300');
    fprintf('Figure saved as: %s.png\n', filepath_base);
    print(fig_handle, [filepath_base '.eps'], '-depsc', '-vector');
    fprintf('Figure saved as: %s.eps\n', filepath_base);
catch ME_save
    fprintf('Error saving figure: %s\n', ME_save.message);
end
% Save Data to CSVs
try
    num_sessions = max_sessions_for_any_subject;
    valid_subjects_idx = find(subject_has_legend_entry);
    valid_subject_labels = subject_legend_labels(valid_subjects_idx);
    session_headers = arrayfun(@(x) sprintf('Session_%d', x), 1:num_sessions, 'UniformOutput', false);

    if any(~isnan(subject_pa_diff_matrix_aligned(:)))
        T_pa = array2table([overall_avg_pa_smoothed; overall_se_pa_raw; subject_pa_diff_matrix_smoothed(valid_subjects_idx, :)], ...
            'VariableNames', session_headers, 'RowNames', [{'Overall_Average_Smoothed'; 'Overall_SE_Raw'}; valid_subject_labels(:)]);
        writetable(T_pa, [filepath_base '_pro_anti_diff.csv'], 'WriteRowNames', true);
        fprintf('Pro-Anti accuracy difference data saved to: %s_pro_anti_diff.csv\n', full_base_filename);
    end

    if any(~isnan(subject_ci_diff_matrix_aligned(:)))
        T_ci = array2table([overall_avg_ci_smoothed; overall_se_ci_raw; subject_ci_diff_matrix_smoothed(valid_subjects_idx, :)], ...
             'VariableNames', session_headers, 'RowNames', [{'Overall_Average_Smoothed'; 'Overall_SE_Raw'}; valid_subject_labels(:)]);
        writetable(T_ci, [filepath_base '_cong_incong_diff.csv'], 'WriteRowNames', true);
        fprintf('Cong-Incong accuracy difference data saved to: %s_cong_incong_diff.csv\n', full_base_filename);
    end

    % --- NEW: Save Regression Data to CSV ---
    if num_sessions >= 2
        regression_results = cell(0, 6);
        if ~isempty(mdl_pa)
            regression_results = [regression_results; {'Pro-Anti_Diff', 'Overall_Average', slope_pa, intercept_pa, mdl_pa.Rsquared.Ordinary, mdl_pa.Coefficients.pValue(2)}];
        end
        if ~isempty(mdl_ci)
            regression_results = [regression_results; {'Cong-Incong_Diff', 'Overall_Average', slope_ci, intercept_ci, mdl_ci.Rsquared.Ordinary, mdl_ci.Coefficients.pValue(2)}];
        end
        
        x_vals_subj = 1:num_sessions;
        for i = 1:length(valid_subjects_idx)
            iS = valid_subjects_idx(i);
            y_subj_pa = subject_pa_diff_matrix_aligned(iS, :);
            if sum(~isnan(y_subj_pa)) >= 2
                p = polyfit(x_vals_subj(~isnan(y_subj_pa)), y_subj_pa(~isnan(y_subj_pa)), 1);
                regression_results = [regression_results; {'Pro-Anti_Diff', subject_legend_labels{iS}, p(1), p(2), NaN, NaN}];
            end
            y_subj_ci = subject_ci_diff_matrix_aligned(iS, :);
            if sum(~isnan(y_subj_ci)) >= 2
                p = polyfit(x_vals_subj(~isnan(y_subj_ci)), y_subj_ci(~isnan(y_subj_ci)), 1);
                regression_results = [regression_results; {'Cong-Incong_Diff', subject_legend_labels{iS}, p(1), p(2), NaN, NaN}];
            end
        end

        if ~isempty(regression_results)
            regression_table = cell2table(regression_results, 'VariableNames', {'AnalysisType', 'Series', 'Slope', 'Intercept', 'R_Squared', 'P_Value'});
            writetable(regression_table, [filepath_base '_regression_stats.csv']);
            fprintf('Regression statistics saved to: %s_regression_stats.csv\n', full_base_filename);
        end
    end
catch ME_csv
    fprintf('Error saving CSV data: %s\n', ME_csv.message);
end

end




function plot_AS_accuracy_differences(metrics_mt)
% PLOT_AS_ACCURACY_DIFFERENCES Plots accuracy differences for Pro-Anti
% and Congruent-Incongruent conditions in the Anti-Saccade task.
%
% Features:
%   - Subject-level and overall data with error bars.
%   - One-sample t-test for overall differences against 0.
%   - Saves plot as PNG and EPS, and exports data to CSV files.
%
% Args:
%   metrics_mt (cell array): Data structure from previous examples.

% --- Configuration ---
plot_font_size = 12;
figure_width = 300;   % MODIFIED: Figure size changed
figure_height = 300;  % MODIFIED: Figure size changed
subject_id_list_hardcoded = {
    'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan'
};
custom_colors_rgb = {
    [245, 124, 110]/255; [242, 181, 110]/255; [251, 231, 158]/255;
    [132, 195, 183]/255; [136, 215, 218]/255; [113, 184, 237]/255;
    [184, 174, 234]/255; [242, 168, 218]/255
};
num_hardcoded_subject_details = length(subject_id_list_hardcoded);
num_defined_custom_colors = length(custom_colors_rgb);
num_subjects_to_process = length(metrics_mt);
if num_subjects_to_process == 0
    disp('Input metrics_mt is empty. No subjects to plot.');
    return;
end

% --- Data Aggregation for Differences ---
subject_session_diffs.pro_anti = cell(num_subjects_to_process, 1);
subject_session_diffs.cong_incong = cell(num_subjects_to_process, 1);
subject_legend_labels = cell(1, num_subjects_to_process);
valid_subject_indices = [];
for iS = 1:num_subjects_to_process
    if iS <= num_hardcoded_subject_details && ~isempty(subject_id_list_hardcoded{iS})
        subject_legend_labels{iS} = subject_id_list_hardcoded{iS};
    else
        subject_legend_labels{iS} = ['S' num2str(iS)];
    end
    if isempty(metrics_mt{iS}); continue; end
    
    current_subject_has_data_for_diff = false;
    for iD = 1:length(metrics_mt{iS})
        session_data = metrics_mt{iS}(iD);
        
        if ~isfield(session_data, 'AccuracyMeanSE_Pro') || isempty(session_data.AccuracyMeanSE_Pro)
            continue;
        end
        
        acc_pro_mean  = session_data.AccuracyMeanSE_Pro(1);
        acc_anti_mean = session_data.AccuracyMeanSE_Anti(1);
        
        if ~isnan(acc_pro_mean) && ~isnan(acc_anti_mean)
            diff_pa = acc_pro_mean - acc_anti_mean;
            subject_session_diffs.pro_anti{iS} = [subject_session_diffs.pro_anti{iS}, diff_pa];
            current_subject_has_data_for_diff = true;
        end
        
        acc_cong_combined = session_data.Accuracy_Cong_Combined;
        session_acc_cong = NaN; session_acc_incong = NaN;
        if size(acc_cong_combined,1) >= 4 && size(acc_cong_combined,2) >=4
            mean_ll = acc_cong_combined(1,1); count_ll = acc_cong_combined(1,4);
            mean_rr = acc_cong_combined(3,1); count_rr = acc_cong_combined(3,4);
            if (count_ll + count_rr) > 0
                 valid_ll = ~isnan(mean_ll) && count_ll > 0; valid_rr = ~isnan(mean_rr) && count_rr > 0;
                 if valid_ll && valid_rr; session_acc_cong = (mean_ll * count_ll + mean_rr * count_rr) / (count_ll + count_rr);
                 elseif valid_ll; session_acc_cong = mean_ll; elseif valid_rr; session_acc_cong = mean_rr; end
            end
            mean_lr = acc_cong_combined(2,1); count_lr = acc_cong_combined(2,4);
            mean_rl = acc_cong_combined(4,1); count_rl = acc_cong_combined(4,4);
            if (count_lr + count_rl) > 0
                 valid_lr = ~isnan(mean_lr) && count_lr > 0; valid_rl = ~isnan(mean_rl) && count_rl > 0;
                 if valid_lr && valid_rl; session_acc_incong = (mean_lr * count_lr + mean_rl * count_rl) / (count_lr + count_rl);
                 elseif valid_lr; session_acc_incong = mean_lr; elseif valid_rl; session_acc_incong = mean_rl; end
            end
        end
        if ~isnan(session_acc_cong) && ~isnan(session_acc_incong)
            diff_ci = session_acc_cong - session_acc_incong;
            subject_session_diffs.cong_incong{iS} = [subject_session_diffs.cong_incong{iS}, diff_ci];
            current_subject_has_data_for_diff = true;
        end
    end
    if current_subject_has_data_for_diff
        valid_subject_indices = [valid_subject_indices, iS];
    end
end
if isempty(valid_subject_indices)
    disp('No subjects with valid session differences found. Cannot generate plot.');
    return;
end

num_valid_subjects = length(valid_subject_indices);
actual_subject_legend_labels = cell(1, num_valid_subjects);
actual_subject_colors = cell(1, num_valid_subjects);
subj_mean_diff.pro_anti = NaN(1, num_valid_subjects);
subj_se_diff.pro_anti   = NaN(1, num_valid_subjects);
subj_mean_diff.cong_incong = NaN(1, num_valid_subjects);
subj_se_diff.cong_incong   = NaN(1, num_valid_subjects);

for i_idx = 1:num_valid_subjects
    iS = valid_subject_indices(i_idx);
    actual_subject_legend_labels{i_idx} = subject_legend_labels{iS};
    actual_subject_colors{i_idx} = custom_colors_rgb{mod(i_idx-1, num_defined_custom_colors) + 1};
    
    sessions_pa = subject_session_diffs.pro_anti{iS};
    if ~isempty(sessions_pa) && sum(~isnan(sessions_pa)) > 0
        subj_mean_diff.pro_anti(i_idx) = mean(sessions_pa, 'omitnan');
        if sum(~isnan(sessions_pa)) > 1
             subj_se_diff.pro_anti(i_idx) = std(sessions_pa, 0, 'omitnan') / sqrt(sum(~isnan(sessions_pa)));
        else
             subj_se_diff.pro_anti(i_idx) = 0;
        end
    end
    sessions_ci = subject_session_diffs.cong_incong{iS};
    if ~isempty(sessions_ci) && sum(~isnan(sessions_ci)) > 0
        subj_mean_diff.cong_incong(i_idx) = mean(sessions_ci, 'omitnan');
         if sum(~isnan(sessions_ci)) > 1
            subj_se_diff.cong_incong(i_idx) = std(sessions_ci, 0, 'omitnan') / sqrt(sum(~isnan(sessions_ci)));
         else
            subj_se_diff.cong_incong(i_idx) = 0;
         end
    end
end

all_sessions_diff.pro_anti = horzcat(subject_session_diffs.pro_anti{valid_subject_indices});
all_sessions_diff.cong_incong = horzcat(subject_session_diffs.cong_incong{valid_subject_indices});
overall_mean_diff.pro_anti = mean(all_sessions_diff.pro_anti, 'omitnan');
overall_se_diff.pro_anti = std(all_sessions_diff.pro_anti, 0, 'omitnan') / sqrt(sum(~isnan(all_sessions_diff.pro_anti)));
overall_mean_diff.cong_incong = mean(all_sessions_diff.cong_incong, 'omitnan');
overall_se_diff.cong_incong = std(all_sessions_diff.cong_incong, 0, 'omitnan') / sqrt(sum(~isnan(all_sessions_diff.cong_incong)));

p_overall_pro_anti = NaN; N_pa_overall = 0;
if sum(~isnan(all_sessions_diff.pro_anti)) >= 2
    [~, p_overall_pro_anti, ~, stats_pa] = ttest(all_sessions_diff.pro_anti);
    N_pa_overall = stats_pa.df + 1;
end
p_overall_cong_incong = NaN; N_ci_overall = 0;
if sum(~isnan(all_sessions_diff.cong_incong)) >= 2
    [~, p_overall_cong_incong, ~, stats_ci] = ttest(all_sessions_diff.cong_incong);
    N_ci_overall = stats_ci.df + 1;
end

% --- Plotting ---
screen_size = get(0, 'ScreenSize');
fig_pos_x = (screen_size(3) - figure_width) / 2;
fig_pos_y = (screen_size(4) - figure_height) / 2;
fig_handle = figure('Position', [fig_pos_x, fig_pos_y, figure_width, figure_height], 'Color', 'w');
hold on;
x_conditions = [1, 2];
condition_labels = {'Pro - Anti', 'Cong. - Incong.'};
marker_size = 5;      % MODIFIED: Dot size set to 5
errorbar_capsize = 6;

plot([0.5, 2.5], [0, 0], 'k-', 'LineWidth', 1, 'HandleVisibility','off');

overall_diff_means = [overall_mean_diff.pro_anti, overall_mean_diff.cong_incong];
overall_diff_ses = [overall_se_diff.pro_anti, overall_se_diff.cong_incong];
errorbar(x_conditions, overall_diff_means, overall_diff_ses, ...
    'LineStyle', 'none', 'Marker', 'o', 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k', ...
    'Color', 'k', 'LineWidth', 1.5, 'CapSize', errorbar_capsize, 'MarkerSize', marker_size, ...
    'DisplayName', 'Overall Mean');

total_jitter_width = 0.4;
jitter_values = linspace(-total_jitter_width/2, total_jitter_width/2, num_valid_subjects);
if num_valid_subjects <= 1; jitter_values = 0; end

for i_idx = 1:num_valid_subjects
    current_jitter = jitter_values(i_idx);
    errorbar(x_conditions(1) + current_jitter, subj_mean_diff.pro_anti(i_idx), subj_se_diff.pro_anti(i_idx), ...
        'LineStyle', 'none', 'Marker', 'o', 'MarkerFaceColor', actual_subject_colors{i_idx}, 'MarkerEdgeColor', actual_subject_colors{i_idx}*0.7, ...
        'Color', actual_subject_colors{i_idx}, 'LineWidth', 1, 'CapSize', errorbar_capsize-2, 'MarkerSize', marker_size);
    errorbar(x_conditions(2) + current_jitter, subj_mean_diff.cong_incong(i_idx), subj_se_diff.cong_incong(i_idx), ...
        'LineStyle', 'none', 'Marker', 'o', 'MarkerFaceColor', actual_subject_colors{i_idx}, 'MarkerEdgeColor', actual_subject_colors{i_idx}*0.7, ...
        'Color', actual_subject_colors{i_idx}, 'LineWidth', 1, 'CapSize', errorbar_capsize-2, 'MarkerSize', marker_size);
end

% --- Aesthetics ---
ax = gca;
ax.FontSize = plot_font_size;
ylabel('Accuracy Difference', 'FontSize', plot_font_size);
title('Accuracy Differences', 'FontSize', plot_font_size + 1);
xticks(x_conditions);
xticklabels(condition_labels);
xlim([0.5, 2.5]);
ylim([-0.1, 0.15]);

common_stat_line_y = 0.09;
text_y_offset_stars = 0.005;
if ~isnan(p_overall_pro_anti)
    stars = 'n.s.';
    if p_overall_pro_anti < 0.001; stars = '***';
    elseif p_overall_pro_anti < 0.01; stars = '**';
    elseif p_overall_pro_anti < 0.05; stars = '*'; end
    plot([x_conditions(1)-0.1, x_conditions(1)+0.1], [common_stat_line_y, common_stat_line_y], '-k', 'HandleVisibility','off');
    text(x_conditions(1), common_stat_line_y + text_y_offset_stars, stars, 'HorizontalAlignment', 'center', 'VerticalAlignment','bottom');
end
if ~isnan(p_overall_cong_incong)
    stars = 'n.s.';
    if p_overall_cong_incong < 0.001; stars = '***';
    elseif p_overall_cong_incong < 0.01; stars = '**';
    elseif p_overall_cong_incong < 0.05; stars = '*'; end
    plot([x_conditions(2)-0.1, x_conditions(2)+0.1], [common_stat_line_y, common_stat_line_y], '-k', 'HandleVisibility','off');
    text(x_conditions(2), common_stat_line_y + text_y_offset_stars, stars, 'HorizontalAlignment', 'center', 'VerticalAlignment','bottom');
end
hold off;

% --- Saving Outputs ---
% MODIFIED: This section now saves figures (PNG, EPS) and data (CSV).
figure(fig_handle);
base_filename = 'antisaccade_accuracy_differences';
date_str = datestr(now, 'yyyymmdd_HHMM');
full_base_filename = [base_filename '_' date_str];
save_folder = 'AS_Figures_Differences';
if ~exist(save_folder, 'dir'); mkdir(save_folder); end
filepath_base = fullfile(save_folder, full_base_filename);

% Save Figure (PNG and EPS)
try
    png_filename = [filepath_base '.png'];
    saveas(fig_handle, png_filename);
    fprintf('Figure saved as: %s\n', png_filename);

    eps_filename = [filepath_base '.eps'];
    print(fig_handle, eps_filename, '-depsc');
    fprintf('Figure saved as: %s\n', eps_filename);
catch ME_save
    fprintf('Error saving figure: %s\n', ME_save.message);
end

% Save Data to CSVs
try
    % --- Summary Data CSV ---
    groups = [{'Overall'}; actual_subject_legend_labels(:)];
    pa_means = [overall_mean_diff.pro_anti; subj_mean_diff.pro_anti(:)];
    pa_ses = [overall_se_diff.pro_anti; subj_se_diff.pro_anti(:)];
    ci_means = [overall_mean_diff.cong_incong; subj_mean_diff.cong_incong(:)];
    ci_ses = [overall_se_diff.cong_incong; subj_se_diff.cong_incong(:)];
    
    summary_table = table(groups, pa_means, pa_ses, ci_means, ci_ses, ...
        'VariableNames', {'Group', 'Pro_Anti_Diff_Mean', 'Pro_Anti_Diff_SE', 'Cong_Incong_Diff_Mean', 'Cong_Incong_Diff_SE'});

    csv_summary_filename = [filepath_base '_accuracy_differences_data.csv'];
    writetable(summary_table, csv_summary_filename);
    fprintf('Summary accuracy difference data saved to: %s\n', csv_summary_filename);

    % --- Statistics CSV ---
    comparisons = {};
    p_values = [];
    t_stats = [];
    dfs = [];
    n_sessions = [];

    if exist('stats_pa', 'var')
        comparisons{end+1} = 'Pro-Anti vs 0';
        p_values(end+1) = p_overall_pro_anti;
        t_stats(end+1) = stats_pa.tstat;
        dfs(end+1) = stats_pa.df;
        n_sessions(end+1) = N_pa_overall;
    end

    if exist('stats_ci', 'var')
        comparisons{end+1} = 'Cong-Incong vs 0';
        p_values(end+1) = p_overall_cong_incong;
        t_stats(end+1) = stats_ci.tstat;
        dfs(end+1) = stats_ci.df;
        n_sessions(end+1) = N_ci_overall;
    end

    if ~isempty(comparisons)
        stats_table = table(comparisons', p_values', t_stats', dfs', n_sessions', ...
            'VariableNames', {'Comparison', 'PValue', 'TStatistic', 'DF', 'N_Sessions'});
        csv_stats_filename = [filepath_base '_accuracy_differences_stats.csv'];
        writetable(stats_table, csv_stats_filename);
        fprintf('Statistical results for accuracy differences saved to: %s\n', csv_stats_filename);
    end

catch ME_csv
    fprintf('Error saving CSV data: %s\n', ME_csv.message);
end

end

function plot_AS_accuracy(metrics_mt)
% PLOT_AS_ACCURACY_FINAL Generates a plot for Anti-Saccade task accuracy.
%
% Features:
%   - Y-axis maximum is fixed at 1.0.
%   - Significance lines for comparisons are at the same y-level.
%   - Grand average plotted as black dots with error bars.
%   - Individual subject accuracies plotted as colored dots.
%   - Lines connect subject's Pro-Anti dots and Congruent-Incongruent dots.
%   - 'Overall' subject accuracy is a standalone dot.
%   - T-tests performed on session-level differences (N = session count).
%   - Saves plot as PNG and EPS, and exports data to CSV files.
%
% Args:
%   metrics_mt (cell array): Data structure where metrics_mt{iS}(iD)
%                            contains metrics for subject iS, session iD.

% --- Configuration ---
plot_font_size = 14;
figure_width = 400;
figure_height = 300;
subject_id_list_hardcoded = {
    'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan'
};
custom_colors_rgb = { 
    [241, 88, 70]/255; [255, 176, 80]/255; [251, 231, 158]/255; 
    [136, 215, 218]/255; [87, 169, 230]/255; [107, 114, 182]/255 
};
num_hardcoded_subject_details = length(subject_id_list_hardcoded);
num_defined_custom_colors = length(custom_colors_rgb);
num_subjects_to_process = length(metrics_mt);
if num_subjects_to_process == 0
    disp('Input metrics_mt is empty. No subjects to plot.');
    return;
end

% --- Data Aggregation ---
subject_session_means.all   = cell(num_subjects_to_process, 1);
subject_session_means.pro   = cell(num_subjects_to_process, 1);
subject_session_means.anti  = cell(num_subjects_to_process, 1);
subject_session_means.cong  = cell(num_subjects_to_process, 1);
subject_session_means.incong = cell(num_subjects_to_process, 1);
session_differences.pro_anti = [];
session_differences.cong_incong = [];
subject_legend_labels = cell(1, num_subjects_to_process);
valid_subject_indices = [];
for iS = 1:num_subjects_to_process
    if iS <= num_hardcoded_subject_details && ~isempty(subject_id_list_hardcoded{iS})
        subject_name_from_list = subject_id_list_hardcoded{iS};
        subject_legend_labels{iS} = subject_name_from_list;
    else
        subject_legend_labels{iS} = ['S' num2str(iS)];
    end
    if isempty(metrics_mt{iS})
        continue;
    end
    
    current_subject_has_data = false;
    for iD = 1:length(metrics_mt{iS})
        session_data = metrics_mt{iS}(iD);
        
        if ~isfield(session_data, 'AccuracyMeanSE_Pro') || ~isfield(session_data, 'AccuracyMeanSE_Anti') || ...
           ~isfield(session_data, 'Accuracy_Cong_Combined') || isempty(session_data.AccuracyMeanSE_Pro)
            continue;
        end
        
        acc_pro_mean  = session_data.AccuracyMeanSE_Pro(1);
        count_pro     = session_data.AccuracyMeanSE_Pro(4);
        acc_anti_mean = session_data.AccuracyMeanSE_Anti(1);
        count_anti    = session_data.AccuracyMeanSE_Anti(4);
        
        session_acc_all = NaN;
        if (count_pro + count_anti) > 0 && ~isnan(acc_pro_mean) && ~isnan(acc_anti_mean)
            session_acc_all = (acc_pro_mean * count_pro + acc_anti_mean * count_anti) / (count_pro + count_anti);
        elseif ~isnan(acc_pro_mean) && (isnan(acc_anti_mean) || count_anti == 0) && count_pro > 0 
            session_acc_all = acc_pro_mean;
        elseif (isnan(acc_pro_mean) || count_pro == 0) && ~isnan(acc_anti_mean) && count_anti > 0
            session_acc_all = acc_anti_mean;
        end
        
        acc_cong_combined = session_data.Accuracy_Cong_Combined;
        session_acc_cong = NaN; session_acc_incong = NaN;
        if size(acc_cong_combined,1) >= 4 && size(acc_cong_combined,2) >=4
            mean_ll = acc_cong_combined(1,1); count_ll = acc_cong_combined(1,4);
            mean_rr = acc_cong_combined(3,1); count_rr = acc_cong_combined(3,4);
            if (count_ll + count_rr) > 0
                 valid_ll = ~isnan(mean_ll) && count_ll > 0;
                 valid_rr = ~isnan(mean_rr) && count_rr > 0;
                 if valid_ll && valid_rr
                    session_acc_cong = (mean_ll * count_ll + mean_rr * count_rr) / (count_ll + count_rr);
                 elseif valid_ll
                    session_acc_cong = mean_ll;
                 elseif valid_rr
                    session_acc_cong = mean_rr;
                 end
            end
            mean_lr = acc_cong_combined(2,1); count_lr = acc_cong_combined(2,4);
            mean_rl = acc_cong_combined(4,1); count_rl = acc_cong_combined(4,4);
            if (count_lr + count_rl) > 0
                 valid_lr = ~isnan(mean_lr) && count_lr > 0;
                 valid_rl = ~isnan(mean_rl) && count_rl > 0;
                 if valid_lr && valid_rl
                    session_acc_incong = (mean_lr * count_lr + mean_rl * count_rl) / (count_lr + count_rl);
                 elseif valid_lr
                    session_acc_incong = mean_lr;
                 elseif valid_rl
                    session_acc_incong = mean_rl;
                 end
            end
        end
        
        if ~isnan(session_acc_all);  subject_session_means.all{iS}    = [subject_session_means.all{iS}, session_acc_all]; end
        if ~isnan(acc_pro_mean);     subject_session_means.pro{iS}    = [subject_session_means.pro{iS}, acc_pro_mean]; end
        if ~isnan(acc_anti_mean);    subject_session_means.anti{iS}   = [subject_session_means.anti{iS}, acc_anti_mean]; end
        if ~isnan(session_acc_cong); subject_session_means.cong{iS}   = [subject_session_means.cong{iS}, session_acc_cong]; end
        if ~isnan(session_acc_incong); subject_session_means.incong{iS} = [subject_session_means.incong{iS}, session_acc_incong]; end
        
        if ~isnan(acc_pro_mean) && ~isnan(acc_anti_mean)
            session_differences.pro_anti = [session_differences.pro_anti, acc_pro_mean - acc_anti_mean];
        end
        if ~isnan(session_acc_cong) && ~isnan(session_acc_incong)
            session_differences.cong_incong = [session_differences.cong_incong, session_acc_cong - session_acc_incong];
        end
        
        if any([~isnan(session_acc_all), ~isnan(acc_pro_mean), ~isnan(acc_anti_mean), ~isnan(session_acc_cong), ~isnan(session_acc_incong)])
            current_subject_has_data = true;
        end
    end
    if current_subject_has_data
        valid_subject_indices = [valid_subject_indices, iS];
    end
end
if isempty(valid_subject_indices)
    disp('No subjects with valid AS data found. Cannot generate plot.');
    return;
end
num_valid_subjects = length(valid_subject_indices);
subj_avg_acc.all    = NaN(1, num_valid_subjects);
subj_avg_acc.pro    = NaN(1, num_valid_subjects);
subj_avg_acc.anti   = NaN(1, num_valid_subjects);
subj_avg_acc.cong   = NaN(1, num_valid_subjects);
subj_avg_acc.incong = NaN(1, num_valid_subjects);
actual_subject_legend_labels = cell(1, num_valid_subjects);
actual_subject_colors = cell(1, num_valid_subjects);
for i_valid_subj = 1:num_valid_subjects
    iS = valid_subject_indices(i_valid_subj);
    actual_subject_legend_labels{i_valid_subj} = subject_legend_labels{iS};
    color_idx_cycle = mod(i_valid_subj - 1, num_defined_custom_colors) + 1;
    if i_valid_subj > num_defined_custom_colors 
        cc = lines(num_valid_subjects); 
        actual_subject_colors{i_valid_subj} = cc(i_valid_subj,:);
    else
         actual_subject_colors{i_valid_subj} = custom_colors_rgb{color_idx_cycle};
    end
    
    if ~isempty(subject_session_means.all{iS});    subj_avg_acc.all(i_valid_subj)    = mean(subject_session_means.all{iS}, 'omitnan'); end
    if ~isempty(subject_session_means.pro{iS});    subj_avg_acc.pro(i_valid_subj)    = mean(subject_session_means.pro{iS}, 'omitnan'); end
    if ~isempty(subject_session_means.anti{iS});   subj_avg_acc.anti(i_valid_subj)   = mean(subject_session_means.anti{iS}, 'omitnan'); end
    if ~isempty(subject_session_means.cong{iS});   subj_avg_acc.cong(i_valid_subj)   = mean(subject_session_means.cong{iS}, 'omitnan'); end
    if ~isempty(subject_session_means.incong{iS}); subj_avg_acc.incong(i_valid_subj) = mean(subject_session_means.incong{iS}, 'omitnan'); end
end
all_sessions_flat.all    = horzcat(subject_session_means.all{valid_subject_indices});
all_sessions_flat.pro    = horzcat(subject_session_means.pro{valid_subject_indices});
all_sessions_flat.anti   = horzcat(subject_session_means.anti{valid_subject_indices});
all_sessions_flat.cong   = horzcat(subject_session_means.cong{valid_subject_indices});
all_sessions_flat.incong = horzcat(subject_session_means.incong{valid_subject_indices});
grand_mean.all    = mean(all_sessions_flat.all, 'omitnan');
grand_mean.pro    = mean(all_sessions_flat.pro, 'omitnan');
grand_mean.anti   = mean(all_sessions_flat.anti, 'omitnan');
grand_mean.cong   = mean(all_sessions_flat.cong, 'omitnan');
grand_mean.incong = mean(all_sessions_flat.incong, 'omitnan');
grand_sem.all    = std(all_sessions_flat.all, 0, 'omitnan') / sqrt(sum(~isnan(all_sessions_flat.all)));
grand_sem.pro    = std(all_sessions_flat.pro, 0, 'omitnan') / sqrt(sum(~isnan(all_sessions_flat.pro)));
grand_sem.anti   = std(all_sessions_flat.anti, 0, 'omitnan') / sqrt(sum(~isnan(all_sessions_flat.anti)));
grand_sem.cong   = std(all_sessions_flat.cong, 0, 'omitnan') / sqrt(sum(~isnan(all_sessions_flat.cong)));
grand_sem.incong = std(all_sessions_flat.incong, 0, 'omitnan') / sqrt(sum(~isnan(all_sessions_flat.incong)));

% --- Plotting ---
screen_size = get(0, 'ScreenSize');
fig_pos_x = (screen_size(3) - figure_width) / 2;
fig_pos_y = (screen_size(4) - figure_height) / 2;
figure('Position', [fig_pos_x, fig_pos_y, figure_width, figure_height], 'Color', 'w');
hold on; 
x_positions = [1, 2.2, 3, 4.4, 5.2]; 
condition_labels = {'Overall', 'Pro', 'Anti', 'Cong.', 'Incong.'};
grand_means_vector = [grand_mean.all, grand_mean.pro, grand_mean.anti, grand_mean.cong, grand_mean.incong];
grand_sems_vector  = [grand_sem.all, grand_sem.pro, grand_sem.anti, grand_sem.cong, grand_sem.incong];
plot(x_positions, grand_means_vector, 'ko', ...
    'MarkerSize', 5, 'MarkerFaceColor', 'k', 'LineWidth', 1.5, 'DisplayName', 'Group Mean'); % MODIFIED: MarkerSize 6 -> 5
errorbar(x_positions, grand_means_vector, grand_sems_vector, ...
    'k.', 'LineWidth', 1.5, 'CapSize', 10, 'HandleVisibility','off');
subj_handles = gobjects(1, num_valid_subjects);
marker_size_subj = 5; % MODIFIED: MarkerSize 6 -> 5
line_width_subj = 1;
for i_subj = 1:num_valid_subjects
    subj_color = actual_subject_colors{i_subj};
    current_x_offset = 0; 
    
    x_overall = x_positions(1) + current_x_offset;
    y_overall = subj_avg_acc.all(i_subj);
    
    x_pro_anti = x_positions(2:3) + current_x_offset;
    y_pro_anti = [subj_avg_acc.pro(i_subj), subj_avg_acc.anti(i_subj)];
    
    x_cong_incong = x_positions(4:5) + current_x_offset;
    y_cong_incong = [subj_avg_acc.cong(i_subj), subj_avg_acc.incong(i_subj)];
    if ~isnan(y_overall)
        h_subj_plot = plot(x_overall, y_overall, 'o', ...
            'Color', subj_color, ...
            'MarkerFaceColor', subj_color, ...
            'MarkerEdgeColor', subj_color*0.7, ...
            'LineWidth', line_width_subj, 'MarkerSize', marker_size_subj, ...
            'HandleVisibility', 'on'); 
        subj_handles(i_subj) = h_subj_plot;
    end
    
    if any(~isnan(y_pro_anti))
        plot(x_pro_anti, y_pro_anti, '-o', ...
            'Color', subj_color, ...
            'MarkerFaceColor', subj_color, ...
            'MarkerEdgeColor', subj_color*0.7, ...
            'LineWidth', line_width_subj, 'MarkerSize', marker_size_subj, ...
            'HandleVisibility', 'off'); 
    end
    
    if any(~isnan(y_cong_incong))
        plot(x_cong_incong, y_cong_incong, '-o', ...
            'Color', subj_color, ...
            'MarkerFaceColor', subj_color, ...
            'MarkerEdgeColor', subj_color*0.7, ...
            'LineWidth', line_width_subj, 'MarkerSize', marker_size_subj, ...
            'HandleVisibility', 'off'); 
    end
end

% --- Aesthetics ---
ax = gca;
ax.FontSize = plot_font_size;
ylabel('Accuracy', 'FontSize', plot_font_size);
title('Anti-Saccade Task Accuracy', 'FontSize', plot_font_size + 1);
xticks(x_positions);
xticklabels(condition_labels);
xtickangle(0);
xlim([x_positions(1)-0.5, x_positions(end)+0.5]);
all_plotted_y_values = [];
grand_means_no_nan = grand_means_vector(~isnan(grand_means_vector));
grand_sems_no_nan = grand_sems_vector(~isnan(grand_means_vector));
if ~isempty(grand_means_no_nan)
    all_plotted_y_values = [all_plotted_y_values; grand_means_no_nan(:) + grand_sems_no_nan(:); grand_means_no_nan(:) - grand_sems_no_nan(:)];
end
fields = fieldnames(subj_avg_acc);
for f_idx = 1:length(fields)
    data_field = subj_avg_acc.(fields{f_idx});
    if ~isempty(data_field(~isnan(data_field))); all_plotted_y_values = [all_plotted_y_values; data_field(~isnan(data_field))']; end
end
ylim_top = 1.0;
if ~isempty(all_plotted_y_values)
    min_y_data = min(all_plotted_y_values(all_plotted_y_values > -Inf));
    if isempty(min_y_data); min_y_data = 0; end
    
    y_range_for_padding = ylim_top - min_y_data;
    if y_range_for_padding <= 0; y_range_for_padding = 0.1; end
    y_padding = 0.1 * y_range_for_padding;
    ylim_bottom = max(0, min_y_data - y_padding);
    if ylim_bottom >= ylim_top; ylim_bottom = ylim_top - 0.1; end
    if ylim_bottom < 0; ylim_bottom = 0; end
else
    ylim_bottom = 0;
end
ylim([ylim_bottom ylim_top]);
grid off; 
box off;  
valid_subj_handles = subj_handles(isgraphics(subj_handles));
valid_subj_labels_for_legend = actual_subject_legend_labels(isgraphics(subj_handles));
if ~isempty(valid_subj_handles) && ~isempty(valid_subj_labels_for_legend)
    legend(valid_subj_handles, valid_subj_labels_for_legend, 'Location', 'eastoutside', 'FontSize', plot_font_size-2, 'Box', 'off');
end

% --- Statistical Comparisons ---
y_lim_curr = ylim; 
text_y_offset = 0.015 * (y_lim_curr(2)-y_lim_curr(1));
relevant_y_values_for_stat_lines = [];
upper_bounds_fields = {'pro', 'anti', 'cong', 'incong'};
for f_idx = 1:length(upper_bounds_fields)
    field = upper_bounds_fields{f_idx};
    if ~isnan(grand_mean.(field)) && ~isnan(grand_sem.(field))
        relevant_y_values_for_stat_lines(end+1) = grand_mean.(field) + grand_sem.(field);
    end
    subj_data = subj_avg_acc.(field);
    relevant_y_values_for_stat_lines = [relevant_y_values_for_stat_lines, subj_data(~isnan(subj_data))];
end
common_stat_line_y = 0.95;
if ~isempty(relevant_y_values_for_stat_lines)
    top_data_boundary = max(relevant_y_values_for_stat_lines(isfinite(relevant_y_values_for_stat_lines))); 
    if ~isempty(top_data_boundary) && common_stat_line_y < top_data_boundary + 0.035 
        common_stat_line_y = top_data_boundary + 0.045; 
    end
end
common_stat_line_y = min(common_stat_line_y, 0.98);
if common_stat_line_y <= ylim_bottom
    common_stat_line_y = (ylim_bottom + ylim_top) * 0.9;
end
if length(session_differences.pro_anti) >= 2
    [~, p_pro_vs_anti, ~, stats_pro_anti] = ttest(session_differences.pro_anti);
    fprintf('One-sample t-test on session differences (Pro - Anti Accuracy): p = %.4f (N_sessions = %d)\n', p_pro_vs_anti, stats_pro_anti.df + 1);
    stars = 'n.s.';
    if p_pro_vs_anti < 0.001; stars = '***';
    elseif p_pro_vs_anti < 0.01; stars = '**';
    elseif p_pro_vs_anti < 0.05; stars = '*'; end
    
    plot([x_positions(2), x_positions(3)], [common_stat_line_y, common_stat_line_y], '-k', 'LineWidth', 1, 'HandleVisibility','off');
    text(mean([x_positions(2), x_positions(3)]), common_stat_line_y + text_y_offset, ...
        stars, 'HorizontalAlignment', 'center', 'VerticalAlignment','bottom', 'FontSize', plot_font_size, 'BackgroundColor', 'none');
end
if length(session_differences.cong_incong) >= 2
    [~, p_cong_vs_incong, ~, stats_cong_incong] = ttest(session_differences.cong_incong);
    fprintf('One-sample t-test on session differences (Congruent - Incongruent Accuracy): p = %.4f (N_sessions = %d)\n', p_cong_vs_incong, stats_cong_incong.df + 1);
    stars = 'n.s.';
    if p_cong_vs_incong < 0.001; stars = '***';
    elseif p_cong_vs_incong < 0.01; stars = '**';
    elseif p_cong_vs_incong < 0.05; stars = '*'; end
    plot([x_positions(4), x_positions(5)], [common_stat_line_y, common_stat_line_y], '-k', 'LineWidth', 1, 'HandleVisibility','off');
    text(mean([x_positions(4), x_positions(5)]), common_stat_line_y + text_y_offset, ...
        stars, 'HorizontalAlignment', 'center', 'VerticalAlignment','bottom', 'FontSize', plot_font_size, 'BackgroundColor', 'none');
end
hold off; 
drawnow;
try 
    lgd = findobj(gcf, 'Type', 'Legend');
    if ~isempty(lgd) && strcmp(lgd.Location, 'eastoutside')
        ax_pos = get(ax, 'Position'); 
        lgd_pos = get(lgd, 'OuterPosition'); 
        max_allowable_ax_width = 1 - lgd_pos(3) - ax_pos(1) - 0.03; 
        if ax_pos(3) > max_allowable_ax_width && max_allowable_ax_width > 0.2 
            set(ax, 'Position', [ax_pos(1), ax_pos(2), max_allowable_ax_width, ax_pos(4)]);
        end
    end
catch
end

% --- Saving Outputs ---
% MODIFIED: This section now saves figures (PNG, EPS) and data (CSV).
figure(gcf); 
base_filename = 'antisaccade_accuracy_final'; 
date_str = datestr(now, 'yyyymmdd_HHMM');
full_base_filename = [base_filename '_' date_str];
save_folder = 'AS_Figures_Final'; 
if ~exist(save_folder, 'dir'); mkdir(save_folder); end
filepath_base = fullfile(save_folder, full_base_filename);

% Save Figure (PNG and EPS)
try
    png_filename = [filepath_base '.png'];
    saveas(gcf, png_filename);
    fprintf('Figure saved as: %s\n', png_filename);
    
    eps_filename = [filepath_base '.eps'];
    print(gcf, eps_filename, '-depsc'); % Save as vector EPS
    fprintf('Figure saved as: %s\n', eps_filename);
catch ME
    fprintf('Error saving figure: %s\n', ME.message);
end

% Save Data to CSVs
try
    % Grand Averages Data
    grand_avg_table = table(condition_labels', grand_means_vector', grand_sems_vector', ...
        'VariableNames', {'Condition', 'Mean', 'SEM'});
    csv_grand_avg_filename = [filepath_base '_grand_averages.csv'];
    writetable(grand_avg_table, csv_grand_avg_filename);
    fprintf('Grand average data saved to: %s\n', csv_grand_avg_filename);
    
    % Subject Averages Data
    subject_avg_data = [subj_avg_acc.all; subj_avg_acc.pro; subj_avg_acc.anti; subj_avg_acc.cong; subj_avg_acc.incong]';
    subject_avg_table = array2table(subject_avg_data, ...
        'RowNames', actual_subject_legend_labels, ...
        'VariableNames', {'Overall', 'Pro', 'Anti', 'Congruent', 'Incongruent'});
    csv_subj_avg_filename = [filepath_base '_subject_averages.csv'];
    writetable(subject_avg_table, csv_subj_avg_filename, 'WriteRowNames', true);
    fprintf('Subject average data saved to: %s\n', csv_subj_avg_filename);
    
    % Statistical Results
    comparisons = {};
    p_values = [];
    t_stats = [];
    dfs = [];
    n_sessions = [];
    
    if exist('p_pro_vs_anti', 'var')
        comparisons{end+1} = 'Pro vs Anti';
        p_values(end+1) = p_pro_vs_anti;
        t_stats(end+1) = stats_pro_anti.tstat;
        dfs(end+1) = stats_pro_anti.df;
        n_sessions(end+1) = stats_pro_anti.df + 1;
    end
    
    if exist('p_cong_vs_incong', 'var')
        comparisons{end+1} = 'Congruent vs Incongruent';
        p_values(end+1) = p_cong_vs_incong;
        t_stats(end+1) = stats_cong_incong.tstat;
        dfs(end+1) = stats_cong_incong.df;
        n_sessions(end+1) = stats_cong_incong.df + 1;
    end
    
    if ~isempty(comparisons)
        stats_table = table(comparisons', p_values', t_stats', dfs', n_sessions', ...
            'VariableNames', {'Comparison', 'PValue', 'TStatistic', 'DF', 'N_Sessions'});
        csv_stats_filename = [filepath_base '_statistics.csv'];
        writetable(stats_table, csv_stats_filename);
        fprintf('Statistical results saved to: %s\n', csv_stats_filename);
    end
    
catch ME_csv
    fprintf('Error saving CSV data: %s\n', ME_csv.message);
end

end

function plot_CEn_proportion_over_sessions(metrics_mt)
% PLOT_CEN_PROPORTION_OVER_SESSIONS Plots CEn proportion over chronologically
% ordered sessions for each subject and an overall session-wised average.
%
% MODIFIED: This function now includes a shaded standard error region around
% the overall average line and saves data to CSV files.
%
% Args:
%   metrics_mt (cell array): Learning data structure.

% --- Configuration ---
plot_font_size = 14;
figure_width = 800;
figure_height = 400;
smoothing_window_size = 5; 
subject_id_list_hardcoded = { 
    'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan'
};
num_hardcoded_subject_details = length(subject_id_list_hardcoded);
custom_colors_rgb = {
    [245, 124, 110]/255; [242, 181, 110]/255; [251, 231, 158]/255;
    [132, 195, 183]/255; [136, 215, 218]/255; [113, 184, 237]/255;
    [184, 174, 234]/255; [242, 168, 218]/255
};
num_defined_custom_colors = length(custom_colors_rgb);
num_subjects_to_process = length(metrics_mt);
if num_subjects_to_process == 0
    disp('Input metrics_mt is empty. No subjects to plot.');
    return;
end

% --- Data Aggregation ---
all_subjects_session_info = cell(num_subjects_to_process, 1);
subject_legend_labels = cell(1, num_subjects_to_process);
max_sessions_for_any_subject = 0;

for iS = 1:num_subjects_to_process
    if iS <= num_hardcoded_subject_details
        subject_name_from_list = subject_id_list_hardcoded{iS};
        name_parts = strsplit(subject_name_from_list, '_'); subject_name = name_parts{1};
        if ~isempty(subject_name); subject_initial = upper(subject_name(1)); subject_legend_labels{iS} = sprintf('Subject %s', subject_initial);
        else; subject_legend_labels{iS} = ['Subject ' num2str(iS)]; end
    else; subject_legend_labels{iS} = ['Subject ' num2str(iS)]; end

    subject_sessions_temp = []; 
    if isempty(metrics_mt{iS})
        fprintf('Data for Subject %s (Index %d) is empty, skipping.\n', subject_legend_labels{iS}, iS);
        all_subjects_session_info{iS} = [];
        continue;
    end

    for iD = 1:length(metrics_mt{iS}) 
        session_data = metrics_mt{iS}(iD);
        if ~isfield(session_data, 'perseverationsN_fl') || ~isfield(session_data, 'dataset')
            fprintf('Warning: Missing perseverationsN_fl or dataset field for S%d Sess%d. Skipping session.\n',iS,iD);
            continue;
        end

        total_CEn_in_session = 0; total_valid_trials_in_session = 0;
        num_blocks_in_session = size(session_data.perseverationsN_fl, 1);
        if num_blocks_in_session == 0; continue; end

        for j = 1:num_blocks_in_session
             if size(session_data.perseverationsN_fl, 2) < 7
                fprintf('Warning: perseverationsN_fl has fewer than 7 columns for S%d Sess%d Blk%d. Skipping block.\n',iS,iD,j);
                continue;
            end
            total_CEn_in_session = total_CEn_in_session + session_data.perseverationsN_fl(j,2);
            total_valid_trials_in_session = total_valid_trials_in_session + session_data.perseverationsN_fl(j,7);
        end

        session_cen_proportion = NaN;
        if total_valid_trials_in_session > 0
            session_cen_proportion = total_CEn_in_session / total_valid_trials_in_session;
        end

        session_datetime = NaT; 
        datetime_str_match = regexp(session_data.dataset, '(\d{2}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2})', 'once', 'match');
        if ~isempty(datetime_str_match)
            try
                session_datetime = datetime(datetime_str_match, 'InputFormat', 'MM_dd_yy__HH_mm_ss', 'PivotYear', year(now)-50);
            catch ME_date
                fprintf('Warning: Could not parse datetime string "%s" for S%d Sess%d. Error: %s\n', datetime_str_match, iS, iD, ME_date.message);
            end
        else
             fprintf('Warning: Datetime string not found in dataset for S%d Sess%d: "%s". Using session index for ordering.\n', iS, iD, session_data.dataset);
             session_datetime = datetime(2000,1,1) + days(iD-1); 
        end

        if ~isnan(session_cen_proportion) && ~isnat(session_datetime)
            subject_sessions_temp = [subject_sessions_temp; struct('datetime', session_datetime, 'cen_prop', session_cen_proportion, 'original_iD', iD)];
        end
    end 
    
    if ~isempty(subject_sessions_temp)
        [~, sort_idx] = sort([subject_sessions_temp.datetime]);
        all_subjects_session_info{iS} = subject_sessions_temp(sort_idx);
        if length(subject_sessions_temp) > max_sessions_for_any_subject
            max_sessions_for_any_subject = length(subject_sessions_temp);
        end
    else
        all_subjects_session_info{iS} = [];
    end
end 

if max_sessions_for_any_subject == 0
    disp('No valid session data with CEn proportions found across all subjects.');
    return;
end

subject_cen_prop_matrix_aligned = nan(num_subjects_to_process, max_sessions_for_any_subject);
for iS = 1:num_subjects_to_process
    if ~isempty(all_subjects_session_info{iS})
        num_subj_sessions = length(all_subjects_session_info{iS});
        for k_sess = 1:num_subj_sessions
            subject_cen_prop_matrix_aligned(iS, k_sess) = all_subjects_session_info{iS}(k_sess).cen_prop;
        end
    end
end

subject_cen_prop_matrix_smoothed = nan(size(subject_cen_prop_matrix_aligned));
for iS = 1:num_subjects_to_process
    subject_cen_prop_matrix_smoothed(iS, :) = smoothdata(subject_cen_prop_matrix_aligned(iS, :), ...
                                                          'movmean', smoothing_window_size, 'omitnan');
end

% MODIFIED: Calculate raw average, SE, and smoothed bounds
overall_session_avg_cen_prop_raw = nanmean(subject_cen_prop_matrix_aligned, 1);
num_subjects_per_session = sum(~isnan(subject_cen_prop_matrix_aligned), 1);
overall_session_se_raw = nanstd(subject_cen_prop_matrix_aligned, 0, 1) ./ sqrt(num_subjects_per_session);
overall_upper_bound_raw = overall_session_avg_cen_prop_raw + overall_session_se_raw;
overall_lower_bound_raw = overall_session_avg_cen_prop_raw - overall_session_se_raw;

overall_session_avg_cen_prop_smoothed = smoothdata(overall_session_avg_cen_prop_raw, 'movmean', smoothing_window_size, 'omitnan');
overall_upper_bound_smoothed = smoothdata(overall_upper_bound_raw, 'movmean', smoothing_window_size, 'omitnan');
overall_lower_bound_smoothed = smoothdata(overall_lower_bound_raw, 'movmean', smoothing_window_size, 'omitnan');

% --- Plotting ---
screen_size = get(0, 'ScreenSize');
fig_pos_x = (screen_size(3) - figure_width) / 2;
fig_pos_y = (screen_size(4) - figure_height) / 2;
figure('Position', [fig_pos_x, fig_pos_y, figure_width, figure_height]);
hold on;

x_axis_values = 1:max_sessions_for_any_subject;
subject_legend_handles = gobjects(num_subjects_to_process, 1);
subject_has_legend_entry = false(num_subjects_to_process, 1);

for iS = 1:num_subjects_to_process
    if ~isempty(all_subjects_session_info{iS}) 
        subj_cen_props_to_plot = subject_cen_prop_matrix_smoothed(iS, :); 
        valid_data_indices = ~isnan(subj_cen_props_to_plot);
        if any(valid_data_indices) 
            subj_line_color = custom_colors_rgb{mod(iS - 1, num_defined_custom_colors) + 1};
            h_line = plot(x_axis_values(valid_data_indices), subj_cen_props_to_plot(valid_data_indices), ...
                          '-o', 'LineWidth', 1, 'Color', subj_line_color, ...
                          'MarkerFaceColor', subj_line_color, 'MarkerSize', 1);
            if ~subject_has_legend_entry(iS) 
                subject_legend_handles(iS) = h_line;
                subject_has_legend_entry(iS) = true;
            end
        end
    end
end

% MODIFIED: Plot shaded SE area for the overall average
valid_overall_avg_indices = ~isnan(overall_session_avg_cen_prop_smoothed) & ...
                              ~isnan(overall_upper_bound_smoothed) & ...
                              ~isnan(overall_lower_bound_smoothed);

if any(valid_overall_avg_indices)
    x_fill = x_axis_values(valid_overall_avg_indices);
    y_upper = overall_upper_bound_smoothed(valid_overall_avg_indices);
    y_lower = overall_lower_bound_smoothed(valid_overall_avg_indices);
    
    y_lower(y_lower < 0) = 0; % Ensure SE doesn't go below 0 for proportion

    fill_x_coords = [x_fill, fliplr(x_fill)];
    fill_y_coords = [y_upper, fliplr(y_lower)];
    
    % Plot the shaded gray area with 0.5 alpha
    fill(fill_x_coords, fill_y_coords, [0.5 0.5 0.5], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    
    % Plot the average line on top
    plot(x_axis_values(valid_overall_avg_indices), overall_session_avg_cen_prop_smoothed(valid_overall_avg_indices), ...
         '--ko', 'LineWidth', 2, 'MarkerFaceColor', 'k', 'MarkerSize', 1);
end
hold off;

% --- Aesthetics ---
ax = gca;
xlabel_str = 'Session Number';
if max_sessions_for_any_subject > 0
    custom_ticks = [1, 5:5:max_sessions_for_any_subject];
    if max_sessions_for_any_subject > 1; custom_ticks = [custom_ticks, max_sessions_for_any_subject]; end
    new_xticks = unique(sort(custom_ticks(custom_ticks <= max_sessions_for_any_subject & custom_ticks >= 1)));
    if isempty(new_xticks); new_xticks = 1; end
    set(ax, 'XTick', new_xticks);
    xlim([0.5 max_sessions_for_any_subject + 0.5]);
else 
    set(ax, 'XTick', [1]);
    xlim([0.5 1.5]); 
end

xlabel(xlabel_str, 'FontSize', plot_font_size);
ylabel('Smoothed Avg. CEn Proportion (per session)', 'FontSize', plot_font_size);
title('Smoothed CEn Proportion Across Sessions', 'FontSize', plot_font_size + 1);

all_plotted_values_for_ylim = subject_cen_prop_matrix_smoothed(~isnan(subject_cen_prop_matrix_smoothed));
if ~isempty(overall_session_avg_cen_prop_smoothed(~isnan(overall_session_avg_cen_prop_smoothed)))
    all_plotted_values_for_ylim = [all_plotted_values_for_ylim(:); overall_session_avg_cen_prop_smoothed(~isnan(overall_session_avg_cen_prop_smoothed))'];
end
if ~isempty(all_plotted_values_for_ylim)
    min_y_val = min(all_plotted_values_for_ylim); 
    max_y_val = max(all_plotted_values_for_ylim);
    y_padding = (max_y_val - min_y_val) * 0.005;
    if y_padding < 0.005 || isnan(y_padding); y_padding = 0.02; end 
    ylim_bottom = max(0, min_y_val - y_padding); 
    ylim_top = min(1.0, max_y_val + y_padding);
    if ylim_bottom >= ylim_top; ylim_top = ylim_bottom + 0.005; end
    ylim([ylim_bottom ylim_top]);
else
    ylim([0 0.5]); 
end
ax.FontSize = plot_font_size;
grid off;

if max_sessions_for_any_subject > 0
    current_plot_ylim = get(ax, 'YLim');
    text_y_pos = current_plot_ylim(1) - 0.08 * diff(current_plot_ylim);
    text(1, text_y_pos, 'Early', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
         'FontSize', plot_font_size - 2, 'Color', [0.3 0.3 0.3]);
    if max_sessions_for_any_subject > 1
        text(max_sessions_for_any_subject, text_y_pos, 'Late', ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
             'FontSize', plot_font_size - 2, 'Color', [0.3 0.3 0.3]);
    end
    drawnow;
    current_pos = get(ax, 'Position');
    if current_pos(2) > 0.15
         original_bottom = current_pos(2);
         current_pos(2) = original_bottom + 0.05;
         current_pos(4) = current_pos(4) - 0.05;
         if current_pos(4) > 0.1; set(ax, 'Position', current_pos); end
    end
end

handles_for_actual_legend = subject_legend_handles(subject_has_legend_entry);
labels_for_actual_legend = subject_legend_labels(subject_has_legend_entry);
if ~isempty(handles_for_actual_legend)
    legend(handles_for_actual_legend, labels_for_actual_legend, ...
           'Location', 'eastoutside', 'FontSize', plot_font_size - 2);
else
    disp('No subject data plotted; legend not created.');
end

% --- Saving Files ---
figure(gcf); 
base_filename = 'flexlearning_CEn_proportion_over_sessions_smoothed'; 
date_str = datestr(now, 'yyyymmdd_HHMM');
full_base_filename = [base_filename '_' date_str];
save_folder = 'FL_CEn_SessionSmoothed'; 
if ~exist(save_folder, 'dir'); mkdir(save_folder); end
filepath_base = fullfile(save_folder, full_base_filename);

try
    session_headers = "Session_" + (1:max_sessions_for_any_subject);
    
    row_labels_smoothed = [subject_legend_labels, {'Overall_Average'}]';
    data_to_save_smoothed = [subject_cen_prop_matrix_smoothed; overall_session_avg_cen_prop_smoothed];
    T_smoothed = array2table(data_to_save_smoothed, 'VariableNames', session_headers);
    T_smoothed = [table(row_labels_smoothed, 'VariableNames', {'Subject'}), T_smoothed];
    writetable(T_smoothed, [filepath_base '_smoothed_data.csv']);
    fprintf('Smoothed data saved as: %s\n', [filepath_base '_smoothed_data.csv']);
    
    row_labels_raw = [subject_legend_labels, {'Overall_Average'}]';
    data_to_save_raw = [subject_cen_prop_matrix_aligned; overall_session_avg_cen_prop_raw];
    T_raw = array2table(data_to_save_raw, 'VariableNames', session_headers);
    T_raw = [table(row_labels_raw, 'VariableNames', {'Subject'}), T_raw];
    writetable(T_raw, [filepath_base '_raw_data.csv']);
    fprintf('Raw data saved as: %s\n', [filepath_base '_raw_data.csv']);
    
    raw_data_log = {};
    for iS = 1:num_subjects_to_process
        for iSession = 1:max_sessions_for_any_subject
            val = subject_cen_prop_matrix_aligned(iS, iSession);
            if ~isnan(val)
                raw_data_log = [raw_data_log; {subject_legend_labels{iS}, iSession, val}];
            end
        end
    end
    
    if ~isempty(raw_data_log)
        T_raw_long = cell2table(raw_data_log, 'VariableNames', {'Subject', 'SessionNumber', 'CEn_Proportion'});
        writetable(T_raw_long, [filepath_base '_raw_data_long_format.csv']);
        fprintf('Long-format raw data saved as: %s\n', [filepath_base '_raw_data_long_format.csv']);
    end
    
catch ME
    fprintf('Error saving CSV data: %s\n', ME.message);
end

try
    saveas(gcf, [filepath_base '.png']);
    fprintf('Figure saved as: %s\n', [filepath_base '.png']);
    print(gcf, [filepath_base '.eps'], '-depsc', '-painters');
    fprintf('Figure saved as: %s\n', [filepath_base '.eps']);
catch ME
    fprintf('Error saving figure: %s\n', ME.message);
end

end

function plot_CEn_proportion_by_condition(metrics_mt)
% PLOT_CEN_PROPORTION_BY_CONDITION Generates a plot showing average
% CEn proportion for various conditions, with pooled block-level
% error bars for the overall mean, and pairwise statistical comparisons on subject means.
% X-axis conditions are grouped visually.
%
% MODIFIED: This function's CSV saving logic has been updated to match
% the two-file format (_data.csv and _stats.csv).
%
% Args:
%   metrics_mt (cell array): Learning data structure. metrics_mt{iS}
%                            is expected to correspond to the i-th subject
%                            in the hardcoded list.
% --- Configuration ---
plot_font_size = 14;
figure_width = 500;
figure_height = 300;
condition_labels = {'All', 'ID', 'ED', 'SAME', 'NEW', 'L-3', 'L-1', 'G2', 'G4'};
num_conditions = length(condition_labels);
x_group_spacing = 1.2; 
x_within_pair_spacing = 0.8;
x_positions = zeros(1, num_conditions);
current_x = 1;
x_positions(1) = current_x; current_x = current_x + x_group_spacing; % All
x_positions(2) = current_x; x_positions(3) = current_x + x_within_pair_spacing; current_x = current_x + x_within_pair_spacing + x_group_spacing; % ID/ED
x_positions(4) = current_x; x_positions(5) = current_x + x_within_pair_spacing; current_x = current_x + x_within_pair_spacing + x_group_spacing; % SAME/NEW
x_positions(6) = current_x; x_positions(7) = current_x + x_within_pair_spacing; current_x = current_x + x_within_pair_spacing + x_group_spacing; % L-3/L-1
x_positions(8) = current_x; x_positions(9) = current_x + x_within_pair_spacing; % G2/G4
subject_id_list_hardcoded = {
    'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan'
};
num_hardcoded_subject_details = length(subject_id_list_hardcoded);
custom_colors_rgb = {
    [245, 124, 110]/255; [242, 181, 110]/255; [251, 231, 158]/255;
    [132, 195, 183]/255; [136, 215, 218]/255; [113, 184, 237]/255;
    [184, 174, 234]/255; [242, 168, 218]/255
};
num_defined_custom_colors = length(custom_colors_rgb);
num_subjects_to_process = length(metrics_mt);
if num_subjects_to_process == 0
    disp('Input metrics_mt is empty. No subjects to plot.');
    return;
end
% --- Data Aggregation ---
subject_condition_CEn_prop_means = nan(num_subjects_to_process, num_conditions);
subject_legend_labels = cell(1, num_subjects_to_process);
pooled_block_CEn_prop_by_condition = cell(1, num_conditions);
for k=1:num_conditions; pooled_block_CEn_prop_by_condition{k} = []; end

for iS = 1:num_subjects_to_process
    if iS <= num_hardcoded_subject_details
        subject_name_from_list = subject_id_list_hardcoded{iS};
        name_parts = strsplit(subject_name_from_list, '_'); subject_name = name_parts{1};
        if ~isempty(subject_name); subject_initial = upper(subject_name(1)); subject_legend_labels{iS} = sprintf('Subject %s', subject_initial);
        else; subject_legend_labels{iS} = ['Subject ' num2str(iS)]; end
    else; subject_legend_labels{iS} = ['Subject ' num2str(iS)]; end
    
    block_CEn_prop_for_this_subject_conditions = cell(1, num_conditions);
    for k=1:num_conditions; block_CEn_prop_for_this_subject_conditions{k} = []; end
    
    if isempty(metrics_mt{iS})
        fprintf('Data for Subject %s (Index %d) is empty, skipping.\n', subject_legend_labels{iS}, iS);
        continue;
    end
    
    for iD = 1:length(metrics_mt{iS})
        session_data = metrics_mt{iS}(iD);
        if ~isfield(session_data, 'perseverationsN_fl') || ...
           ~isfield(session_data, 'blockID_fl') || ...
           ~isfield(session_data, 'tokenCondition_fl')
            fprintf('Warning: Missing required fields for session %d, subject %s. Skipping session.\n', iD, subject_legend_labels{iS});
            continue;
        end
        num_blocks = size(session_data.perseverationsN_fl, 1);
        for j = 1:num_blocks
            if size(session_data.perseverationsN_fl, 2) < 7
                fprintf('Warning: perseverationsN_fl has fewer than 7 columns for S%d Sess%d Blk%d. Skipping block.\n',iS,iD,j);
                continue;
            end
            CEn_count = session_data.perseverationsN_fl(j,2);
            total_valid_trials = session_data.perseverationsN_fl(j,7);
            CEn_proportion_value = NaN;
            if total_valid_trials > 0
                CEn_proportion_value = CEn_count / total_valid_trials;
            end
            
            if isnan(CEn_proportion_value); continue; end
            
            if j > length(session_data.blockID_fl) || isempty(session_data.blockID_fl{j}) || ...
               j > length(session_data.tokenCondition_fl) || isempty(session_data.tokenCondition_fl{j})
                continue;
            end
            block_id_str = session_data.blockID_fl{j};
            token_cond_str = session_data.tokenCondition_fl{j};
            
            conditions_met_indices = [];
            conditions_met_indices = [conditions_met_indices, 1];
            if strncmp(block_id_str, 'ID', 2); conditions_met_indices = [conditions_met_indices, 2]; end
            if strncmp(block_id_str, 'ED', 2); conditions_met_indices = [conditions_met_indices, 3]; end
            if (strncmp(block_id_str,'IDSame',6) || strncmp(block_id_str,'EDSame',6)); conditions_met_indices = [conditions_met_indices, 4]; end
            if (strncmp(block_id_str,'IDNew',5) || strncmp(block_id_str,'EDNew',5)); conditions_met_indices = [conditions_met_indices, 5]; end
            if (strcmp(token_cond_str,'L-3G2') || strcmp(token_cond_str,'L-3G4')); conditions_met_indices = [conditions_met_indices, 6]; end
            if (strcmp(token_cond_str,'L-1G2') || strcmp(token_cond_str,'L-1G4')); conditions_met_indices = [conditions_met_indices, 7]; end
            if (strcmp(token_cond_str,'L-1G2') || strcmp(token_cond_str,'L-3G2')); conditions_met_indices = [conditions_met_indices, 8]; end
            if (strcmp(token_cond_str,'L-1G4') || strcmp(token_cond_str,'L-3G4')); conditions_met_indices = [conditions_met_indices, 9]; end
            
            for k_idx = unique(conditions_met_indices)
                block_CEn_prop_for_this_subject_conditions{k_idx} = [block_CEn_prop_for_this_subject_conditions{k_idx}, CEn_proportion_value];
                pooled_block_CEn_prop_by_condition{k_idx} = [pooled_block_CEn_prop_by_condition{k_idx}, CEn_proportion_value];
            end
        end
    end
    for k_cond = 1:num_conditions
        if ~isempty(block_CEn_prop_for_this_subject_conditions{k_cond})
            subject_condition_CEn_prop_means(iS, k_cond) = nanmean(block_CEn_prop_for_this_subject_conditions{k_cond});
        end
    end
end
overall_pooled_mean_CEn_prop = nan(1, num_conditions);
overall_pooled_se_CEn_prop   = nan(1, num_conditions);
for k_cond = 1:num_conditions
    current_condition_pooled_blocks = pooled_block_CEn_prop_by_condition{k_cond};
    if ~isempty(current_condition_pooled_blocks)
        overall_pooled_mean_CEn_prop(k_cond) = nanmean(current_condition_pooled_blocks);
        if length(current_condition_pooled_blocks) > 1
            overall_pooled_se_CEn_prop(k_cond) = nanstd(current_condition_pooled_blocks) / sqrt(length(current_condition_pooled_blocks));
        else
            overall_pooled_se_CEn_prop(k_cond) = 0; 
        end
    end
end
% --- Plotting ---
screen_size = get(0, 'ScreenSize');
fig_pos_x = (screen_size(3) - figure_width) / 2;
fig_pos_y = (screen_size(4) - figure_height) / 2;
figure('Position', [fig_pos_x, fig_pos_y, figure_width, figure_height]);
hold on;
jitter_range = 0.3 / x_within_pair_spacing;
subject_legend_handles = gobjects(num_subjects_to_process, 1);
subject_has_legend_entry = false(num_subjects_to_process, 1);
for k_cond = 1:num_conditions
    current_x_base = x_positions(k_cond);
    for iS = 1:num_subjects_to_process
        if ~isnan(subject_condition_CEn_prop_means(iS, k_cond))
            subj_dot_color = [];
            if iS <= num_defined_custom_colors; subj_dot_color = custom_colors_rgb{iS};
            else
                color_idx_cycle = mod(iS - 1, num_defined_custom_colors) + 1;
                subj_dot_color = custom_colors_rgb{color_idx_cycle};
            end
            x_jittered = current_x_base + (rand - 0.5) * jitter_range;
            h_dot = plot(x_jittered, subject_condition_CEn_prop_means(iS, k_cond), 'o', ...
                 'MarkerFaceColor', subj_dot_color, 'MarkerEdgeColor', subj_dot_color*0.8, 'MarkerSize', 5);
            if ~subject_has_legend_entry(iS)
                subject_legend_handles(iS) = h_dot; subject_has_legend_entry(iS) = true;
            end
        end
    end
end
for k_cond = 1:num_conditions
    current_x_base = x_positions(k_cond);
    if ~isnan(overall_pooled_mean_CEn_prop(k_cond))
        errorbar(current_x_base, overall_pooled_mean_CEn_prop(k_cond), overall_pooled_se_CEn_prop(k_cond), ...
                 'k', 'LineWidth', 1.5, 'CapSize', 6, 'LineStyle','none');
        plot(current_x_base, overall_pooled_mean_CEn_prop(k_cond), 'o', ...
             'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k', 'MarkerSize', 5);
    end
end
% --- Determine Y-level for Stats Bars (Dynamically) ---
temp_ax_for_ylim = gca; 
current_y_limits_after_data = get(temp_ax_for_ylim, 'YLim');
max_data_y_on_plot = current_y_limits_after_data(2);
all_overall_means_with_se = overall_pooled_mean_CEn_prop + overall_pooled_se_CEn_prop;
max_overall_mean_plus_se = max(all_overall_means_with_se(~isnan(all_overall_means_with_se)));
if ~isempty(max_overall_mean_plus_se)
    max_data_y_on_plot = max(max_data_y_on_plot, max_overall_mean_plus_se);
end
max_subj_dot_y = max(subject_condition_CEn_prop_means(:));
if ~isempty(max_subj_dot_y) && ~isnan(max_subj_dot_y)
    max_data_y_on_plot = max(max_data_y_on_plot, max_subj_dot_y);
end
if isempty(max_data_y_on_plot) || isnan(max_data_y_on_plot) || max_data_y_on_plot == -inf || ~isfinite(max_data_y_on_plot)
     max_data_y_on_plot = 0.1; % Fallback
end
min_data_y_on_plot = current_y_limits_after_data(1);
if isnan(min_data_y_on_plot) || ~isfinite(min_data_y_on_plot); min_data_y_on_plot = 0; end
y_span_data = max_data_y_on_plot - min_data_y_on_plot;
if y_span_data <=0; y_span_data = max_data_y_on_plot; end 
if y_span_data == 0 && max_data_y_on_plot == 0; y_span_data = 0.1; end 
stat_bar_y_level_start_offset = 0.08 * y_span_data; 
stat_bar_y_level = max_data_y_on_plot + stat_bar_y_level_start_offset;
star_text_y_offset = 0.02 * y_span_data; 
cap_height_on_bar = 0.01 * y_span_data;
min_stat_bar_y_level = max_data_y_on_plot + 0.05 * max(0.01,y_span_data); 
stat_bar_y_level = max(stat_bar_y_level, min_stat_bar_y_level);
% --- Statistical Tests and Significance Bars ---
stat_pairs_indices = {[2,3], [4,5], [6,7], [8,9]}; 
for p_idx = 1:length(stat_pairs_indices)
    pair = stat_pairs_indices{p_idx}; idx1 = pair(1); idx2 = pair(2);
    cond1_subject_means = subject_condition_CEn_prop_means(:, idx1);
    cond2_subject_means = subject_condition_CEn_prop_means(:, idx2);
    valid_pairs_mask = ~isnan(cond1_subject_means) & ~isnan(cond2_subject_means);
    data1_paired = cond1_subject_means(valid_pairs_mask);
    data2_paired = cond2_subject_means(valid_pairs_mask);
    if length(data1_paired) >= 2 
        p_value = signrank(data1_paired, data2_paired);
        star_str = 'n.s.'; text_font_weight = 'normal';
        p_val_thresholds = [0.001, 0.01, 0.05]; star_levels = {'***', '**', '*'};
        for s_idx = 1:length(p_val_thresholds)
            if p_value < p_val_thresholds(s_idx)
                star_str = star_levels{s_idx}; text_font_weight = 'bold'; break; 
            end
        end
        x1_pos = x_positions(idx1); x2_pos = x_positions(idx2);
        plot([x1_pos, x2_pos], [stat_bar_y_level, stat_bar_y_level], '-k', 'LineWidth', 1);
        plot([x1_pos, x1_pos], [stat_bar_y_level - cap_height_on_bar, stat_bar_y_level], '-k', 'LineWidth', 1);
        plot([x2_pos, x2_pos], [stat_bar_y_level - cap_height_on_bar, stat_bar_y_level], '-k', 'LineWidth', 1);
        text(mean([x1_pos,x2_pos]), stat_bar_y_level + star_text_y_offset, star_str, ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
             'FontSize', plot_font_size -1, 'FontWeight', text_font_weight);
    end
end
hold off;
% --- Aesthetics ---
ax = gca;
set(ax, 'XTick', x_positions); set(ax, 'XTickLabel', condition_labels);
xtickangle(ax, 30);
xlim([min(x_positions) - 0.75, max(x_positions) + 0.75]);
xlabel('Condition', 'FontSize', plot_font_size);
ylabel('Average CEn Proportion (CEn / Valid Trials)', 'FontSize', plot_font_size);
title('CEn Proportion by Condition', 'FontSize', plot_font_size + 1);
ax.FontSize = plot_font_size; grid off;
final_y_lim_top = stat_bar_y_level + star_text_y_offset + (0.03 * y_span_data);
current_plot_min_y = min_data_y_on_plot;
if isnan(current_plot_min_y) || ~isfinite(current_plot_min_y) || current_plot_min_y < 0; current_plot_min_y = 0; end
ylim_bottom = floor(current_plot_min_y * 100)/100;
ylim_top = ceil(min(1.0, final_y_lim_top) * 100)/100;
if ylim_bottom >= ylim_top; ylim_top = ylim_bottom + 0.05; end
if ylim_top > 1.0; ylim_top = 1.0; end
if ylim_bottom < 0; ylim_bottom = 0; end
ylim([ylim_bottom ylim_top]);
handles_for_actual_legend = subject_legend_handles(subject_has_legend_entry);
labels_for_actual_legend = subject_legend_labels(subject_has_legend_entry);
if ~isempty(handles_for_actual_legend) && ~isempty(labels_for_actual_legend)
    legend(handles_for_actual_legend, labels_for_actual_legend, ...
           'Location', 'eastoutside', 'FontSize', plot_font_size - 2);
else; disp('No subject data plotted; legend not created.'); end
% --- Saving Files ---
figure(gcf); 
base_filename = 'flexlearning_CEn_proportion_by_condition_grouped';
date_str = datestr(now, 'yyyymmdd_HHMM');
full_base_filename = [base_filename '_' date_str];
save_folder = 'FL_CEn_Proportion'; 
if ~exist(save_folder, 'dir'); mkdir(save_folder); end
filepath_base = fullfile(save_folder, full_base_filename);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%% --- START: MODIFIED SECTION FOR SAVING CSV DATA --- %%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
try
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %%% --- NEW CSV SAVING LOGIC (DATA) --- %%%
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % 1. Create Data Table
    % Start with condition labels
    T_data = table(condition_labels(:), 'VariableNames', {'Condition'});
    
    % Add Overall Mean and SE
    T_data.Overall_Mean = overall_pooled_mean_CEn_prop(:);
    T_data.Overall_SE_Pooled = overall_pooled_se_CEn_prop(:);
    
    % Add Subject-wise Mean data
    for iS = 1:num_subjects_to_process
        % Clean up legend label for use as a variable name
        safe_label = strrep(subject_legend_labels{iS}, ' ', '_');
        var_name = sprintf('%s_Mean', safe_label);
        T_data.(var_name) = subject_condition_CEn_prop_means(iS, :)';
    end
    
    % Save Data CSV
    data_csv_filename = [filepath_base '_data.csv'];
    writetable(T_data, data_csv_filename);
    fprintf('Data saved as: %s\n', data_csv_filename);
    
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %%% --- NEW CSV SAVING LOGIC (STATS) --- %%%
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % 1. Initialize Stats Cell Array
    stats_data_to_save = {'Comparison', 'P_Value', 'Z_Value', 'N_Pairs'};
    
    % 2. Recalculate p-values and get stats
    for p_idx = 1:length(stat_pairs_indices)
        pair = stat_pairs_indices{p_idx};
        idx1 = pair(1);
        idx2 = pair(2);
        comparison_label = sprintf('%s_vs_%s', condition_labels{idx1}, condition_labels{idx2});
        
        cond1_subject_means = subject_condition_CEn_prop_means(:, idx1);
        cond2_subject_means = subject_condition_CEn_prop_means(:, idx2);
        valid_pairs_mask = ~isnan(cond1_subject_means) & ~isnan(cond2_subject_means);
        data1_paired = cond1_subject_means(valid_pairs_mask);
        data2_paired = cond2_subject_means(valid_pairs_mask);
        
        n_pairs = length(data1_paired);
        p_val = NaN;
        z_val = NaN; % z-value from signrank
        
        if n_pairs >= 2 
            % Use [p, h, stats] = signrank(...) to get zval
            [p_val, ~, stats] = signrank(data1_paired, data2_paired);
            
            % Check if 'zval' field exists (it should for signrank)
            if isfield(stats, 'zval')
                z_val = stats.zval;
            elseif isfield(stats, 'signedrank') % Fallback
                z_val = stats.signedrank;
            end
        end
        
        % Add row to cell array
        stats_data_to_save(end+1, :) = {comparison_label, p_val, z_val, n_pairs};
    end
    
    % 3. Convert cell to table and save
    stats_csv_filename = [filepath_base '_stats.csv'];
    T_stats = cell2table(stats_data_to_save(2:end,:), 'VariableNames', stats_data_to_save(1,:));
    writetable(T_stats, stats_csv_filename);
    fprintf('Stats saved as: %s\n', stats_csv_filename);
    
catch ME
    fprintf('Error saving CSV data: %s\n', ME.message);
end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%% --- END: MODIFIED SECTION FOR SAVING CSV DATA --- %%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Save Figure (PNG and EPS)
try
    png_filename = [filepath_base '.png']; saveas(gcf, png_filename);
    fprintf('Figure saved as: %s\n', png_filename);
    eps_filename = [filepath_base '.eps']; print(gcf, eps_filename, '-depsc', '-painters');
    fprintf('Figure saved as: %s\n', eps_filename);
catch ME
    fprintf('Error saving figure: %s\n', ME.message);
end
end

function plot_plateau_accuracy_by_condition(metrics_mt)
% PLOT_PLATEAU_ACCURACY_BY_CONDITION Generates a plot showing average
% plateau accuracy for various conditions, with pooled block-level error bars
% for the overall mean, and pairwise statistical comparisons on subject means.
% X-axis conditions are grouped visually.
%
% MODIFIED: This function's CSV saving logic has been updated to match
% the two-file format (_data.csv and _stats.csv).
%
% Args:
%   metrics_mt (cell array): Learning data structure. metrics_mt{iS}
%                            is expected to correspond to the i-th subject
%                            in the hardcoded list.
% --- Configuration ---
plot_font_size = 14;
figure_width = 500;
figure_height = 300;
condition_labels = {'All', 'ID', 'ED', 'SAME', 'NEW', 'L-3', 'L-1', 'G2', 'G4'};
num_conditions = length(condition_labels);
x_group_spacing = 1.2; 
x_within_pair_spacing = 0.8; 
x_positions = zeros(1, num_conditions);
current_x = 1;
x_positions(1) = current_x;
current_x = current_x + x_group_spacing;
x_positions(2) = current_x;
x_positions(3) = current_x + x_within_pair_spacing;
current_x = current_x + x_within_pair_spacing + x_group_spacing;
x_positions(4) = current_x;
x_positions(5) = current_x + x_within_pair_spacing;
current_x = current_x + x_within_pair_spacing + x_group_spacing;
x_positions(6) = current_x;
x_positions(7) = current_x + x_within_pair_spacing;
current_x = current_x + x_within_pair_spacing + x_group_spacing;
x_positions(8) = current_x;
x_positions(9) = current_x + x_within_pair_spacing;
subject_id_list_hardcoded = {
    'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan'
};
num_hardcoded_subject_details = length(subject_id_list_hardcoded);
custom_colors_rgb = {
    [245, 124, 110]/255; [242, 181, 110]/255; [251, 231, 158]/255;
    [132, 195, 183]/255; [136, 215, 218]/255; [113, 184, 237]/255;
    [184, 174, 234]/255; [242, 168, 218]/255
};
num_defined_custom_colors = length(custom_colors_rgb);
num_subjects_to_process = length(metrics_mt);
if num_subjects_to_process == 0
    disp('Input metrics_mt is empty. No subjects to plot.');
    return;
end
% --- Data Aggregation ---
subject_condition_accuracy_means = nan(num_subjects_to_process, num_conditions);
subject_legend_labels = cell(1, num_subjects_to_process);
pooled_block_accuracies_by_condition = cell(1, num_conditions);
for k=1:num_conditions; pooled_block_accuracies_by_condition{k} = []; end

for iS = 1:num_subjects_to_process
    if iS <= num_hardcoded_subject_details
        subject_name_from_list = subject_id_list_hardcoded{iS};
        name_parts = strsplit(subject_name_from_list, '_');
        subject_name = name_parts{1};
        if ~isempty(subject_name)
            subject_initial = upper(subject_name(1));
            subject_legend_labels{iS} = sprintf('Subject %s', subject_initial);
        else
            subject_legend_labels{iS} = ['Subject ' num2str(iS)];
        end
    else
        subject_legend_labels{iS} = ['Subject ' num2str(iS)];
    end
    block_accuracies_for_this_subject_conditions = cell(1, num_conditions);
    for k=1:num_conditions; block_accuracies_for_this_subject_conditions{k} = []; end
    if isempty(metrics_mt{iS})
        fprintf('Data for Subject %s (Index %d) is empty, skipping.\n', subject_legend_labels{iS}, iS);
        continue;
    end
    for iD = 1:length(metrics_mt{iS})
        session_data = metrics_mt{iS}(iD);
        if ~isfield(session_data, 'plateauAccuracy_fl') || ...
           ~isfield(session_data, 'blockID_fl') || ...
           ~isfield(session_data, 'tokenCondition_fl')
            fprintf('Warning: Missing required fields for session %d, subject %s. Skipping session.\n', iD, subject_legend_labels{iS});
            continue;
        end
        num_blocks = size(session_data.plateauAccuracy_fl, 1);
        for j = 1:num_blocks
            accuracy_value = session_data.plateauAccuracy_fl(j,1);
            if isnan(accuracy_value); continue; end
            if j > length(session_data.blockID_fl) || isempty(session_data.blockID_fl{j}) || ...
               j > length(session_data.tokenCondition_fl) || isempty(session_data.tokenCondition_fl{j})
                continue;
            end
            block_id_str = session_data.blockID_fl{j};
            token_cond_str = session_data.tokenCondition_fl{j};
            
            conditions_met_indices = [];
            conditions_met_indices = [conditions_met_indices, 1]; % All
            if strncmp(block_id_str, 'ID', 2); conditions_met_indices = [conditions_met_indices, 2]; end
            if strncmp(block_id_str, 'ED', 2); conditions_met_indices = [conditions_met_indices, 3]; end
            if (strncmp(block_id_str,'IDSame',6) || strncmp(block_id_str,'EDSame',6)); conditions_met_indices = [conditions_met_indices, 4]; end
            if (strncmp(block_id_str,'IDNew',5) || strncmp(block_id_str,'EDNew',5)); conditions_met_indices = [conditions_met_indices, 5]; end
            if (strcmp(token_cond_str,'L-3G2') || strcmp(token_cond_str,'L-3G4')); conditions_met_indices = [conditions_met_indices, 6]; end
            if (strcmp(token_cond_str,'L-1G2') || strcmp(token_cond_str,'L-1G4')); conditions_met_indices = [conditions_met_indices, 7]; end
            if (strcmp(token_cond_str,'L-1G2') || strcmp(token_cond_str,'L-3G2')); conditions_met_indices = [conditions_met_indices, 8]; end
            if (strcmp(token_cond_str,'L-1G4') || strcmp(token_cond_str,'L-3G4')); conditions_met_indices = [conditions_met_indices, 9]; end
            
            for k_idx = unique(conditions_met_indices)
                block_accuracies_for_this_subject_conditions{k_idx} = [block_accuracies_for_this_subject_conditions{k_idx}, accuracy_value];
                pooled_block_accuracies_by_condition{k_idx} = [pooled_block_accuracies_by_condition{k_idx}, accuracy_value];
            end
        end
    end
    for k_cond = 1:num_conditions
        if ~isempty(block_accuracies_for_this_subject_conditions{k_cond})
            subject_condition_accuracy_means(iS, k_cond) = nanmean(block_accuracies_for_this_subject_conditions{k_cond});
        end
    end
end
overall_pooled_mean_accuracy = nan(1, num_conditions);
overall_pooled_se_accuracy   = nan(1, num_conditions);
for k_cond = 1:num_conditions
    current_condition_pooled_blocks = pooled_block_accuracies_by_condition{k_cond};
    if ~isempty(current_condition_pooled_blocks)
        overall_pooled_mean_accuracy(k_cond) = nanmean(current_condition_pooled_blocks);
        if length(current_condition_pooled_blocks) > 1
            overall_pooled_se_accuracy(k_cond) = nanstd(current_condition_pooled_blocks) / sqrt(length(current_condition_pooled_blocks));
        else
            overall_pooled_se_accuracy(k_cond) = 0; 
        end
    end
end
% --- Plotting ---
screen_size = get(0, 'ScreenSize');
fig_pos_x = (screen_size(3) - figure_width) / 2;
fig_pos_y = (screen_size(4) - figure_height) / 2;
figure('Position', [fig_pos_x, fig_pos_y, figure_width, figure_height]);
hold on;
jitter_range = 0.3 / x_within_pair_spacing;
subject_legend_handles = gobjects(num_subjects_to_process, 1);
subject_has_legend_entry = false(num_subjects_to_process, 1);
for k_cond = 1:num_conditions
    current_x_base = x_positions(k_cond);
    for iS = 1:num_subjects_to_process
        if ~isnan(subject_condition_accuracy_means(iS, k_cond))
            subj_dot_color = [];
            if iS <= num_defined_custom_colors; subj_dot_color = custom_colors_rgb{iS};
            else
                color_idx_cycle = mod(iS - 1, num_defined_custom_colors) + 1;
                subj_dot_color = custom_colors_rgb{color_idx_cycle};
            end
            x_jittered = current_x_base + (rand - 0.5) * jitter_range;
            h_dot = plot(x_jittered, subject_condition_accuracy_means(iS, k_cond), 'o', ...
                 'MarkerFaceColor', subj_dot_color, 'MarkerEdgeColor', subj_dot_color*0.8, 'MarkerSize', 5);
            if ~subject_has_legend_entry(iS)
                subject_legend_handles(iS) = h_dot; subject_has_legend_entry(iS) = true;
            end
        end
    end
end
for k_cond = 1:num_conditions
    current_x_base = x_positions(k_cond);
    if ~isnan(overall_pooled_mean_accuracy(k_cond))
        errorbar(current_x_base, overall_pooled_mean_accuracy(k_cond), overall_pooled_se_accuracy(k_cond), ...
                 'k', 'LineWidth', 1.5, 'CapSize', 6, 'LineStyle','none');
        plot(current_x_base, overall_pooled_mean_accuracy(k_cond), 'o', ...
             'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k', 'MarkerSize', 5);
    end
end
% --- Statistical Tests and Significance Bars ---
stat_pairs_indices = {[2,3], [4,5], [6,7], [8,9]}; 
stat_bar_y_level = 0.91; 
star_text_y_offset = 0.005; 
cap_height_on_bar = 0.003; 
for p_idx = 1:length(stat_pairs_indices)
    pair = stat_pairs_indices{p_idx};
    idx1 = pair(1);
    idx2 = pair(2);
    cond1_subject_means = subject_condition_accuracy_means(:, idx1);
    cond2_subject_means = subject_condition_accuracy_means(:, idx2);
    valid_pairs_mask = ~isnan(cond1_subject_means) & ~isnan(cond2_subject_means);
    data1_paired = cond1_subject_means(valid_pairs_mask);
    data2_paired = cond2_subject_means(valid_pairs_mask);
    if length(data1_paired) >= 2 
        p_value = signrank(data1_paired, data2_paired);
        star_str = 'n.s.'; text_font_weight = 'normal';
        p_val_thresholds = [0.001, 0.01, 0.05]; star_levels = {'***', '**', '*'};
        for s_idx = 1:length(p_val_thresholds)
            if p_value < p_val_thresholds(s_idx)
                star_str = star_levels{s_idx}; text_font_weight = 'bold'; break; 
            end
        end
        x1_pos_stat = x_positions(idx1); 
        x2_pos_stat = x_positions(idx2);
        plot([x1_pos_stat, x2_pos_stat], [stat_bar_y_level, stat_bar_y_level], '-k', 'LineWidth', 1);
        plot([x1_pos_stat, x1_pos_stat], [stat_bar_y_level - cap_height_on_bar, stat_bar_y_level], '-k', 'LineWidth', 1);
        plot([x2_pos_stat, x2_pos_stat], [stat_bar_y_level - cap_height_on_bar, stat_bar_y_level], '-k', 'LineWidth', 1);
        text(mean([x1_pos_stat,x2_pos_stat]), stat_bar_y_level + star_text_y_offset, star_str, ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
             'FontSize', plot_font_size -1, 'FontWeight', text_font_weight);
    end
end
hold off;
% --- Aesthetics ---
ax = gca;
set(ax, 'XTick', x_positions);
set(ax, 'XTickLabel', condition_labels);
xtickangle(ax, 30);
xlim([min(x_positions) - 0.75, max(x_positions) + 0.75]);
xlabel('Condition', 'FontSize', plot_font_size);
ylabel('Average Plateau Accuracy', 'FontSize', plot_font_size);
title('Plateau Accuracy by Condition', 'FontSize', plot_font_size + 1);
ax.FontSize = plot_font_size;
grid off;
final_y_lim_top = stat_bar_y_level + star_text_y_offset + 0.015; 
ylim([0.6 min(1.0, max(0.9, final_y_lim_top))]); 
handles_for_actual_legend = subject_legend_handles(subject_has_legend_entry);
labels_for_actual_legend = subject_legend_labels(subject_has_legend_entry);
if ~isempty(handles_for_actual_legend) && ~isempty(labels_for_actual_legend)
    legend(handles_for_actual_legend, labels_for_actual_legend, ...
           'Location', 'eastoutside', 'FontSize', plot_font_size - 2);
else
    disp('No subject data plotted; legend not created.');
end
% --- Saving Files ---
figure(gcf);
base_filename = 'flexlearning_plateau_accuracy_by_condition_grouped';
date_str = datestr(now, 'yyyymmdd_HHMM');
full_base_filename = [base_filename '_' date_str];
save_folder = 'FL_PlateauAccuracy';
if ~exist(save_folder, 'dir')
   mkdir(save_folder);
end
filepath_base = fullfile(save_folder, full_base_filename);

% --- START: MODIFIED SECTION FOR SAVING CSV DATA ---
try
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %%% --- NEW CSV SAVING LOGIC (DATA) --- %%%
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % 1. Create Data Table
    % Start with condition labels
    T_data = table(condition_labels(:), 'VariableNames', {'Condition'});
    
    % Add Overall Mean and SE
    T_data.Overall_Mean = overall_pooled_mean_accuracy(:);
    T_data.Overall_SE_Pooled = overall_pooled_se_accuracy(:);
    
    % Add Subject-wise Mean data
    for iS = 1:num_subjects_to_process
        % Clean up legend label for use as a variable name
        safe_label = strrep(subject_legend_labels{iS}, ' ', '_');
        var_name = sprintf('%s_Mean', safe_label);
        T_data.(var_name) = subject_condition_accuracy_means(iS, :)';
    end
    
    % Save Data CSV
    data_csv_filename = [filepath_base '_data.csv'];
    writetable(T_data, data_csv_filename);
    fprintf('Data saved as: %s\n', data_csv_filename);
    
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %%% --- NEW CSV SAVING LOGIC (STATS) --- %%%
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % 1. Initialize Stats Cell Array
    stats_data_to_save = {'Comparison', 'P_Value', 'Z_Value', 'N_Pairs'};
    
    % 2. Recalculate p-values and get stats
    for p_idx = 1:length(stat_pairs_indices)
        pair = stat_pairs_indices{p_idx};
        idx1 = pair(1);
        idx2 = pair(2);
        comparison_label = sprintf('%s_vs_%s', condition_labels{idx1}, condition_labels{idx2});
        
        cond1_subject_means = subject_condition_accuracy_means(:, idx1);
        cond2_subject_means = subject_condition_accuracy_means(:, idx2);
        valid_pairs_mask = ~isnan(cond1_subject_means) & ~isnan(cond2_subject_means);
        data1_paired = cond1_subject_means(valid_pairs_mask);
        data2_paired = cond2_subject_means(valid_pairs_mask);
        
        n_pairs = length(data1_paired);
        p_val = NaN;
        z_val = NaN; % z-value from signrank
        
        if n_pairs >= 2 
            % Use [p, h, stats] = signrank(...) to get zval
            [p_val, ~, stats] = signrank(data1_paired, data2_paired);
            
            % Check if 'zval' field exists (it should for signrank)
            if isfield(stats, 'zval')
                z_val = stats.zval;
            elseif isfield(stats, 'signedrank') % Fallback
                z_val = stats.signedrank;
            end
        end
        
        % Add row to cell array
        stats_data_to_save(end+1, :) = {comparison_label, p_val, z_val, n_pairs};
    end
    
    % 3. Convert cell to table and save
    stats_csv_filename = [filepath_base '_stats.csv'];
    T_stats = cell2table(stats_data_to_save(2:end,:), 'VariableNames', stats_data_to_save(1,:));
    writetable(T_stats, stats_csv_filename);
    fprintf('Stats saved as: %s\n', stats_csv_filename);
    
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %%% --- END OF NEW CSV LOGIC --- %%%
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
catch ME
    fprintf('Error saving CSV data: %s\n', ME.message);
end
% --- END: MODIFIED SECTION FOR SAVING CSV DATA ---

% Save Figure (PNG and EPS)
try
    png_filename = [filepath_base '.png'];
    saveas(gcf, png_filename);
    fprintf('Figure saved as: %s\n', png_filename);
    eps_filename = [filepath_base '.eps'];
    print(gcf, eps_filename, '-depsc', '-vector'); 
    fprintf('Figure saved as: %s\n', eps_filename);
catch ME
    fprintf('Error saving figure: %s\n', ME.message);
end
end


function plot_flexlearning_learning_points_by_condition(metrics_mt)
% PLOT_LEARNING_POINTS_BY_CONDITION Generates a plot showing average
% trials to criterion for various experimental conditions, with pooled block-level
% error bars for the overall mean, and pairwise statistical comparisons on subject means.
% X-axis conditions are grouped visually.
%
% MODIFIED:
%   - CSV saving logic has been completely rewritten.
%   - Saves '_data.csv' with subjects and overall data in a structured table.
%   - Saves '_stats.csv' with comparison labels, p-values, z-values, and N.
%
% Args:
%   metrics_mt (cell array): Learning data structure. metrics_mt{iS}
%                            is expected to correspond to the i-th subject
%                            in the hardcoded list.
% --- Configuration ---
plot_font_size = 14;
figure_width = 500; % MODIFIED: Figure width
figure_height = 300;
condition_labels = {'All', 'ID', 'ED', 'SAME', 'NEW', 'L-3', 'L-1', 'G2', 'G4'};
num_conditions = length(condition_labels);
% Define x-positions for grouped layout
x_group_spacing = 1.2; % Space between distinct groups/pairs
x_within_pair_spacing = 0.8;  % Space between items within a statistical pair
x_positions = zeros(1, num_conditions);
current_x = 1;
% Condition 1: 'All'
x_positions(1) = current_x;
current_x = current_x + x_group_spacing;
% Pair 1: ID (2), ED (3)
x_positions(2) = current_x;
x_positions(3) = current_x + x_within_pair_spacing;
current_x = current_x + x_within_pair_spacing + x_group_spacing;
% Pair 2: SAME (4), NEW (5)
x_positions(4) = current_x;
x_positions(5) = current_x + x_within_pair_spacing;
current_x = current_x + x_within_pair_spacing + x_group_spacing;
% Pair 3: L-3 (6), L-1 (7)
x_positions(6) = current_x;
x_positions(7) = current_x + x_within_pair_spacing;
current_x = current_x + x_within_pair_spacing + x_group_spacing;
% Pair 4: G2 (8), G4 (9)
x_positions(8) = current_x;
x_positions(9) = current_x + x_within_pair_spacing;
% End of x_positions definition
subject_id_list_hardcoded = {
    'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan'
};
num_hardcoded_subject_details = length(subject_id_list_hardcoded);
custom_colors_rgb = {
    [245, 124, 110]/255; [242, 181, 110]/255; [251, 231, 158]/255;
    [132, 195, 183]/255; [136, 215, 218]/255; [113, 184, 237]/255;
    [184, 174, 234]/255; [242, 168, 218]/255
};
num_defined_custom_colors = length(custom_colors_rgb);
num_subjects_to_process = length(metrics_mt);
if num_subjects_to_process == 0
    disp('Input metrics_mt is empty. No subjects to plot.');
    return;
end
% --- Data Aggregation (Same as before) ---
subject_condition_ttc_means = nan(num_subjects_to_process, num_conditions);
subject_legend_labels = cell(1, num_subjects_to_process);
pooled_block_ttc_by_condition = cell(1, num_conditions);
for k=1:num_conditions; pooled_block_ttc_by_condition{k} = []; end
for iS = 1:num_subjects_to_process
    if iS <= num_hardcoded_subject_details
        subject_name_from_list = subject_id_list_hardcoded{iS};
        name_parts = strsplit(subject_name_from_list, '_');
        subject_name = name_parts{1};
        if ~isempty(subject_name)
            subject_initial = upper(subject_name(1));
            subject_legend_labels{iS} = sprintf('Subject %s', subject_initial);
        else
            subject_legend_labels{iS} = ['Subject ' num2str(iS)];
        end
    else
        subject_legend_labels{iS} = ['Subject ' num2str(iS)];
    end
    block_ttcs_for_this_subject_conditions = cell(1, num_conditions);
    for k=1:num_conditions; block_ttcs_for_this_subject_conditions{k} = []; end
    if isempty(metrics_mt{iS})
        fprintf('Data for Subject %s (Index %d) is empty, skipping.\n', subject_legend_labels{iS}, iS);
        continue;
    end
    for iD = 1:length(metrics_mt{iS})
        session_data = metrics_mt{iS}(iD);
        if ~isfield(session_data, 'trialsToCriterion_fl') || ...
           ~isfield(session_data, 'blockID_fl') || ...
           ~isfield(session_data, 'tokenCondition_fl')
            fprintf('Warning: Missing required fields for session %d, subject %s. Skipping session.\n', iD, subject_legend_labels{iS});
            continue;
        end
        num_blocks = size(session_data.trialsToCriterion_fl, 1);
        for j = 1:num_blocks
            ttc_value = session_data.trialsToCriterion_fl(j,1);
            if isnan(ttc_value); continue; end
            if j > length(session_data.blockID_fl) || isempty(session_data.blockID_fl{j}) || ...
               j > length(session_data.tokenCondition_fl) || isempty(session_data.tokenCondition_fl{j})
                continue;
            end
            block_id_str = session_data.blockID_fl{j};
            token_cond_str = session_data.tokenCondition_fl{j};
            
            conditions_met_indices = [];
            conditions_met_indices = [conditions_met_indices, 1]; % All
            if strncmp(block_id_str, 'ID', 2); conditions_met_indices = [conditions_met_indices, 2]; end
            if strncmp(block_id_str, 'ED', 2); conditions_met_indices = [conditions_met_indices, 3]; end
            if (strncmp(block_id_str,'IDSame',6) || strncmp(block_id_str,'EDSame',6)); conditions_met_indices = [conditions_met_indices, 4]; end
            if (strncmp(block_id_str,'IDNew',5) || strncmp(block_id_str,'EDNew',5)); conditions_met_indices = [conditions_met_indices, 5]; end
            if (strcmp(token_cond_str,'L-3G2') || strcmp(token_cond_str,'L-3G4')); conditions_met_indices = [conditions_met_indices, 6]; end
            if (strcmp(token_cond_str,'L-1G2') || strcmp(token_cond_str,'L-1G4')); conditions_met_indices = [conditions_met_indices, 7]; end
            if (strcmp(token_cond_str,'L-1G2') || strcmp(token_cond_str,'L-3G2')); conditions_met_indices = [conditions_met_indices, 8]; end
            if (strcmp(token_cond_str,'L-1G4') || strcmp(token_cond_str,'L-3G4')); conditions_met_indices = [conditions_met_indices, 9]; end
            
            for k_idx = unique(conditions_met_indices)
                block_ttcs_for_this_subject_conditions{k_idx} = [block_ttcs_for_this_subject_conditions{k_idx}, ttc_value];
                pooled_block_ttc_by_condition{k_idx} = [pooled_block_ttc_by_condition{k_idx}, ttc_value];
            end
        end
    end
    for k_cond = 1:num_conditions
        if ~isempty(block_ttcs_for_this_subject_conditions{k_cond})
            subject_condition_ttc_means(iS, k_cond) = nanmean(block_ttcs_for_this_subject_conditions{k_cond});
        end
    end
end
overall_pooled_mean_ttc = nan(1, num_conditions);
overall_pooled_se_ttc   = nan(1, num_conditions);
for k_cond = 1:num_conditions
    current_condition_pooled_blocks = pooled_block_ttc_by_condition{k_cond};
    if ~isempty(current_condition_pooled_blocks)
        overall_pooled_mean_ttc(k_cond) = nanmean(current_condition_pooled_blocks);
        if length(current_condition_pooled_blocks) > 1
            overall_pooled_se_ttc(k_cond) = nanstd(current_condition_pooled_blocks) / sqrt(length(current_condition_pooled_blocks));
        else
            overall_pooled_se_ttc(k_cond) = 0; 
        end
    end
end
% --- Plotting ---
screen_size = get(0, 'ScreenSize');
fig_pos_x = (screen_size(3) - figure_width) / 2;
fig_pos_y = (screen_size(4) - figure_height) / 2;
figure('Position', [fig_pos_x, fig_pos_y, figure_width, figure_height]);
hold on;
jitter_range = 0.3 / x_within_pair_spacing; % Jitter relative to spacing within a pair
subject_legend_handles = gobjects(num_subjects_to_process, 1);
subject_has_legend_entry = false(num_subjects_to_process, 1);
% Plot individual subject dots using new x_positions
for k_cond = 1:num_conditions
    current_x_base = x_positions(k_cond);
    for iS = 1:num_subjects_to_process
        if ~isnan(subject_condition_ttc_means(iS, k_cond))
            subj_dot_color = [];
            if iS <= num_defined_custom_colors; subj_dot_color = custom_colors_rgb{iS};
            else
                color_idx_cycle = mod(iS - 1, num_defined_custom_colors) + 1;
                subj_dot_color = custom_colors_rgb{color_idx_cycle};
            end
            x_jittered = current_x_base + (rand - 0.5) * jitter_range;
            h_dot = plot(x_jittered, subject_condition_ttc_means(iS, k_cond), 'o', ...
                 'MarkerFaceColor', subj_dot_color, 'MarkerEdgeColor', subj_dot_color*0.8, 'MarkerSize', 5);
            if ~subject_has_legend_entry(iS)
                subject_legend_handles(iS) = h_dot; subject_has_legend_entry(iS) = true;
            end
        end
    end
end
% Plot overall average dots and error bars using new x_positions
for k_cond = 1:num_conditions
    current_x_base = x_positions(k_cond);
    if ~isnan(overall_pooled_mean_ttc(k_cond))
        errorbar(current_x_base, overall_pooled_mean_ttc(k_cond), overall_pooled_se_ttc(k_cond), ...
                 'k', 'LineWidth', 1.5, 'CapSize', 6, 'LineStyle','none');
        plot(current_x_base, overall_pooled_mean_ttc(k_cond), 'o', ...
             'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k', 'MarkerSize', 5);
    end
end
% --- Determine Y-level for Stats Bars (Dynamically) ---
temp_ax_for_ylim = gca;
current_y_limits_after_data = get(temp_ax_for_ylim, 'YLim');
max_data_y_on_plot = current_y_limits_after_data(2);
all_overall_means_with_se = overall_pooled_mean_ttc + overall_pooled_se_ttc;
max_overall_mean_plus_se = max(all_overall_means_with_se(~isnan(all_overall_means_with_se)));
if ~isempty(max_overall_mean_plus_se)
    max_data_y_on_plot = max(max_data_y_on_plot, max_overall_mean_plus_se);
end
max_subj_dot_y = max(subject_condition_ttc_means(:));
if ~isempty(max_subj_dot_y) && ~isnan(max_subj_dot_y)
    max_data_y_on_plot = max(max_data_y_on_plot, max_subj_dot_y);
end
if isempty(max_data_y_on_plot) || isnan(max_data_y_on_plot) || max_data_y_on_plot == 0 || ~isfinite(max_data_y_on_plot)
    max_data_y_on_plot = 10; % Fallback
end
min_data_y_on_plot = current_y_limits_after_data(1);
if isnan(min_data_y_on_plot) || ~isfinite(min_data_y_on_plot); min_data_y_on_plot = 0; end
y_span_data = max_data_y_on_plot - min_data_y_on_plot;
if y_span_data <=0; y_span_data = max_data_y_on_plot; end 
stat_bar_y_level_start_offset = 0.08 * y_span_data; 
stat_bar_y_level = max_data_y_on_plot + stat_bar_y_level_start_offset;
star_text_y_offset = 0.02 * y_span_data; 
cap_height_on_bar = 0.01 * y_span_data;
min_stat_bar_y_level = max_data_y_on_plot + 0.05 * max(1,y_span_data); % Ensure some minimal space
stat_bar_y_level = max(stat_bar_y_level, min_stat_bar_y_level);
% --- Statistical Tests and Significance Bars ---
stat_pairs_indices = {[2,3], [4,5], [6,7], [8,9]}; 
for p_idx = 1:length(stat_pairs_indices)
    pair = stat_pairs_indices{p_idx};
    idx1 = pair(1);
    idx2 = pair(2);
    cond1_subject_means = subject_condition_ttc_means(:, idx1);
    cond2_subject_means = subject_condition_ttc_means(:, idx2);
    valid_pairs_mask = ~isnan(cond1_subject_means) & ~isnan(cond2_subject_means);
    data1_paired = cond1_subject_means(valid_pairs_mask);
    data2_paired = cond2_subject_means(valid_pairs_mask);
    if length(data1_paired) >= 2 
        p_value = signrank(data1_paired, data2_paired);
        star_str = 'n.s.'; text_font_weight = 'normal';
        p_val_thresholds = [0.001, 0.01, 0.05]; star_levels = {'***', '**', '*'};
        for s_idx = 1:length(p_val_thresholds)
            if p_value < p_val_thresholds(s_idx)
                star_str = star_levels{s_idx}; text_font_weight = 'bold'; break; 
            end
        end
        
        x1_pos = x_positions(idx1); x2_pos = x_positions(idx2);
        plot([x1_pos, x2_pos], [stat_bar_y_level, stat_bar_y_level], '-k', 'LineWidth', 1);
        plot([x1_pos, x1_pos], [stat_bar_y_level - cap_height_on_bar, stat_bar_y_level], '-k', 'LineWidth', 1);
        plot([x2_pos, x2_pos], [stat_bar_y_level - cap_height_on_bar, stat_bar_y_level], '-k', 'LineWidth', 1);
        text(mean([x1_pos,x2_pos]), stat_bar_y_level + star_text_y_offset, star_str, ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
             'FontSize', plot_font_size -1, 'FontWeight', text_font_weight);
    end
end
hold off;
% --- Aesthetics ---
ax = gca;
set(ax, 'XTick', x_positions); % Use new x_positions for ticks
set(ax, 'XTickLabel', condition_labels);
xtickangle(ax, 30);
xlim([min(x_positions) - 0.75, max(x_positions) + 0.75]); % Adjust xlim based on new positions
xlabel('Condition', 'FontSize', plot_font_size);
ylabel('Average Trials to Criterion', 'FontSize', plot_font_size);
title('Learning Points by Condition', 'FontSize', plot_font_size + 1);
ax.FontSize = plot_font_size;
grid off;
final_y_lim_top = stat_bar_y_level + star_text_y_offset + (0.03 * y_span_data);
current_plot_min_y = min_data_y_on_plot; % Use the determined min from actual data
if isnan(current_plot_min_y) || ~isfinite(current_plot_min_y); current_plot_min_y = 0; end
ylim_bottom = floor(max(0, current_plot_min_y - (0 * y_span_data)));
ylim_top = ceil(final_y_lim_top);
if ylim_bottom >= ylim_top; ylim_top = ylim_bottom +1; end % Ensure top > bottom
ylim([ylim_bottom ylim_top]);
handles_for_actual_legend = subject_legend_handles(subject_has_legend_entry);
labels_for_actual_legend = subject_legend_labels(subject_has_legend_entry);
if ~isempty(handles_for_actual_legend) && ~isempty(labels_for_actual_legend)
    legend(handles_for_actual_legend, labels_for_actual_legend, ...
           'Location', 'eastoutside', 'FontSize', plot_font_size - 2);
else
    disp('No subject data plotted; legend not created.');
end
% --- Saving the Figure and Data ---
figure(gcf); % Bring current figure to the front
base_filename = 'flexlearning_learning_points_by_condition_grouped';
date_str = datestr(now, 'yyyymmdd_HHMM');
full_base_filename = [base_filename '_' date_str];
save_folder = 'FL_LearningPointConditions';
if ~exist(save_folder, 'dir')
   mkdir(save_folder);
end
filepath_base = fullfile(save_folder, full_base_filename);

try
    % Save as PNG
    png_filename = [filepath_base '.png'];
    saveas(gcf, png_filename);
    fprintf('Figure saved as: %s\n', png_filename);
    
    % Save as EPS
    eps_filename = [filepath_base '.eps'];
	print(gcf, eps_filename, '-depsc', '-vector');
    fprintf('Figure saved as: %s\n', eps_filename);
    
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %%% --- NEW CSV SAVING LOGIC (DATA) --- %%%
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % 1. Create Data Table
    % Start with condition labels
    T_data = table(condition_labels(:), 'VariableNames', {'Condition'});
    
    % Add Overall Mean and SE
    T_data.Overall_Mean = overall_pooled_mean_ttc(:);
    T_data.Overall_SE_Pooled = overall_pooled_se_ttc(:);
    
    % Add Subject-wise Mean data
    for iS = 1:num_subjects_to_process
        % Clean up legend label for use as a variable name
        safe_label = strrep(subject_legend_labels{iS}, ' ', '_');
        var_name = sprintf('%s_Mean', safe_label);
        T_data.(var_name) = subject_condition_ttc_means(iS, :)';
    end
    
    % Save Data CSV
    data_csv_filename = [filepath_base '_data.csv'];
    writetable(T_data, data_csv_filename);
    fprintf('Data saved as: %s\n', data_csv_filename);
    
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %%% --- NEW CSV SAVING LOGIC (STATS) --- %%%
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % 1. Initialize Stats Cell Array
    stats_data_to_save = {'Comparison', 'P_Value', 'Z_Value', 'N_Pairs'};
    
    % 2. Recalculate p-values and get stats
    for p_idx = 1:length(stat_pairs_indices)
        pair = stat_pairs_indices{p_idx};
        idx1 = pair(1);
        idx2 = pair(2);
        comparison_label = sprintf('%s_vs_%s', condition_labels{idx1}, condition_labels{idx2});
        
        cond1_subject_means = subject_condition_ttc_means(:, idx1);
        cond2_subject_means = subject_condition_ttc_means(:, idx2);
        valid_pairs_mask = ~isnan(cond1_subject_means) & ~isnan(cond2_subject_means);
        data1_paired = cond1_subject_means(valid_pairs_mask);
        data2_paired = cond2_subject_means(valid_pairs_mask);
        
        n_pairs = length(data1_paired);
        p_val = NaN;
        z_val = NaN; % z-value from signrank
        
        if n_pairs >= 2 
            % Use [p, h, stats] = signrank(...) to get zval
            [p_val, ~, stats] = signrank(data1_paired, data2_paired);
            
            % Check if 'zval' field exists (it should for signrank)
            if isfield(stats, 'zval')
                z_val = stats.zval;
            elseif isfield(stats, 'signedrank') % Fallback
                z_val = stats.signedrank;
            end
        end
        
        % Add row to cell array
        stats_data_to_save(end+1, :) = {comparison_label, p_val, z_val, n_pairs};
    end
    
    % 3. Convert cell to table and save
    stats_csv_filename = [filepath_base '_stats.csv'];
    T_stats = cell2table(stats_data_to_save(2:end,:), 'VariableNames', stats_data_to_save(1,:));
    writetable(T_stats, stats_csv_filename);
    fprintf('Stats saved as: %s\n', stats_csv_filename);
    
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %%% --- END OF NEW CSV LOGIC --- %%%
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
catch ME
    fprintf('Error saving figure or CSVs: %s\n', ME.message);
    % Display more detailed error information if needed
    % disp(ME.getReport());
end
end


function plot_flexlearning_learningcurves(metrics_mt)
% PLOT_FLEXLEARNING_CURVES Generates a learning curve plot
% for a predefined list of subjects with specific visual styling including
% disconnected lines at block start, semi-transparent subject lines, and
% a shaded standard error area for the overall average.
%
% Args:
%   metrics_mt (cell array): Learning data structure. metrics_mt{iS}
%                            is expected to correspond to the i-th subject
%                            in the hardcoded list.

% --- Configuration ---
target_x_values = -4:25; % Desired x-axis trial numbers for data processing
plot_font_size = 14;
figure_width = 500;
figure_height = 300;

% Hardcoded subject identifiers (order is important and must match metrics_mt)
subject_id_list_hardcoded = {
    'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan'
};
num_hardcoded_subject_details = length(subject_id_list_hardcoded);

% Hardcoded custom colors (RGB normalized to 0-1)
custom_colors_rgb = {
    [245, 124, 110]/255;  % 1. Bard (#f57c6e)
    [242, 181, 110]/255;  % 2. Frey (#f2b56e)
    [251, 231, 158]/255;  % 3. Igor (#fbe79e)
    [132, 195, 183]/255;  % 4. Reider (#84c3b7)
    [136, 215, 218]/255;  % 5. Sindri (#88d7da)
    [113, 184, 237]/255;  % 6. Wotan (#71b8ed)
    [184, 174, 234]/255;  % 7. Extra color
    [242, 168, 218]/255   % 8. Extra color
};
num_defined_custom_colors = length(custom_colors_rgb);

num_subjects_to_plot = length(metrics_mt);
if num_subjects_to_plot == 0
    disp('Input metrics_mt is empty. No subjects to plot.');
    return;
end

% --- Data Extraction and Processing ---
subject_avg_curves = nan(num_subjects_to_plot, length(target_x_values));
subject_legend_labels = cell(1, num_subjects_to_plot);
all_session_curves = []; %MODIFIED: To store all session curves from all subjects for overall average

for iS = 1:num_subjects_to_plot
    if iS <= num_hardcoded_subject_details
        subject_name_from_list = subject_id_list_hardcoded{iS};
        subject_name_parts = strsplit(subject_name_from_list, '_');
        subject_name = subject_name_parts{1};
        if ~isempty(subject_name)
            subject_initial = upper(subject_name(1));
            subject_legend_labels{iS} = sprintf('Subject %s', subject_initial);
        else
            subject_legend_labels{iS} = ['Subject ' num2str(iS)];
        end
    else
        subject_legend_labels{iS} = ['Subject ' num2str(iS)];
    end

    block_curves_for_this_subject = [];
    if isempty(metrics_mt{iS})
        fprintf('Data for Subject %s (Index %d) is empty, skipping.\n', subject_legend_labels{iS}, iS);
        continue;
    end

    for iD = 1:length(metrics_mt{iS})
        session_data = metrics_mt{iS}(iD);
        if ~isfield(session_data, 'learningCurveRaw_fl') || isempty(session_data.learningCurveRaw_fl)
            continue;
        end

        for j = 1:length(session_data.learningCurveRaw_fl)
            single_block_lc = session_data.learningCurveRaw_fl{j};
            if ~isempty(single_block_lc) && isnumeric(single_block_lc) && ~all(isnan(single_block_lc))
                if iscolumn(single_block_lc)
                    single_block_lc = single_block_lc';
                end
                processed_block_lc = nan(1, length(target_x_values));
                num_points_to_copy = min(length(single_block_lc), length(target_x_values));
                processed_block_lc(1:num_points_to_copy) = single_block_lc(1:num_points_to_copy);
                block_curves_for_this_subject = [block_curves_for_this_subject; processed_block_lc];
            end
        end
    end

    if ~isempty(block_curves_for_this_subject)
        subject_avg_curves(iS, :) = nanmean(block_curves_for_this_subject, 1);
        all_session_curves = [all_session_curves; block_curves_for_this_subject]; %MODIFIED: Aggregate all sessions
    else
        fprintf('Subject %s (Index %d) had no valid curves.\n', subject_legend_labels{iS}, iS);
    end
end

% --- Plotting ---
screen_size = get(0, 'ScreenSize');
fig_pos_x = (screen_size(3) - figure_width) / 2;
fig_pos_y = (screen_size(4) - figure_height) / 2;
figure('Position', [fig_pos_x, fig_pos_y, figure_width, figure_height]);
hold on;

plot_handles = [];
active_legend_labels = {};

% Define indices for splitting the plot
idx_pre_block_end = find(target_x_values == -1, 1);
idx_curr_block_start = find(target_x_values == 1, 1);
if isempty(idx_pre_block_end) || isempty(idx_curr_block_start)
    error('Could not find break points -1 or 1 in target_x_values. Check definition.');
end
x_segment1 = target_x_values(1:idx_pre_block_end);
x_segment2 = target_x_values(idx_curr_block_start:end);

% Plot individual subject average learning curves in two segments
for iS = 1:num_subjects_to_plot
    if ~all(isnan(subject_avg_curves(iS, :)))
        current_curve_data = subject_avg_curves(iS, :);
        y_segment1 = current_curve_data(1:idx_pre_block_end);
        y_segment2 = current_curve_data(idx_curr_block_start:end);
        
        plot_color = custom_colors_rgb{mod(iS - 1, num_defined_custom_colors) + 1};
        if iS > num_defined_custom_colors
             warning('Subject %s (Index %d): Cycling custom colors.', subject_legend_labels{iS}, iS);
        end
        
        % MODIFIED: Plot segments with 0.7 alpha transparency
        plot_color_with_alpha = [plot_color, 1];
        h1 = plot(x_segment1, y_segment1, 'LineWidth', 1.5, 'Color', plot_color_with_alpha);
        plot(x_segment2-1, y_segment2, 'LineWidth', 1.5, 'Color', plot_color_with_alpha);
        
        plot_handles = [plot_handles, h1];
        active_legend_labels{end+1} = subject_legend_labels{iS};
    end
end

% MODIFIED: Calculate overall average and SE based on ALL sessions
if ~isempty(all_session_curves)
    % Calculate mean from all valid sessions
    overall_avg_curve = nanmean(all_session_curves, 1);
    % N = number of valid sessions for each trial point
    num_valid_sessions_per_trial = sum(~isnan(all_session_curves), 1);
    % Calculate Standard Error of the Mean (SEM)
    overall_se_curve = nanstd(all_session_curves, 0, 1) ./ sqrt(num_valid_sessions_per_trial);

    % --- Plot Shaded SE Area and Average Line ---
    % Define upper and lower bounds for the shaded area
    upper_bound = overall_avg_curve + overall_se_curve;
    lower_bound = overall_avg_curve - overall_se_curve;
    
    % Split bounds into segments
    upper_bound_seg1 = upper_bound(1:idx_pre_block_end);
    lower_bound_seg1 = lower_bound(1:idx_pre_block_end);
    upper_bound_seg2 = upper_bound(idx_curr_block_start:end);
    lower_bound_seg2 = lower_bound(idx_curr_block_start:end);
    
    % Plot shaded area for segment 1
    fill_x1 = [x_segment1, fliplr(x_segment1)];
    fill_y1 = [upper_bound_seg1, fliplr(lower_bound_seg1)];
    fill(fill_x1, fill_y1, 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');

    % Plot shaded area for segment 2
    fill_x2 = [x_segment2-1, fliplr(x_segment2-1)];
    fill_y2 = [upper_bound_seg2, fliplr(lower_bound_seg2)];
    fill(fill_x2, fill_y2, 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');

    % Now plot the average line on top
    y_avg_segment1 = overall_avg_curve(1:idx_pre_block_end);
    y_avg_segment2 = overall_avg_curve(idx_curr_block_start:end);
    h_avg1 = plot(x_segment1, y_avg_segment1, '--k', 'LineWidth', 2);
    plot(x_segment2-1, y_avg_segment2, '--k', 'LineWidth', 2);
    
    plot_handles = [plot_handles, h_avg1];
    active_legend_labels{end+1} = 'Overall Average';
end

% Add vertical dashed gray line at x=0
y_limits_for_vline = ylim;
line([0 0], y_limits_for_vline, 'Color', [0.6 0.6 0.6], 'LineStyle', '--', 'LineWidth', 1);
hold off;

% --- Aesthetics ---
xlabel('Trial Number', 'FontSize', plot_font_size);
ylabel('Average Performance', 'FontSize', plot_font_size);
title('Flex Learning: Subject Learning Curves', 'FontSize', plot_font_size + 1);
xlim([min(target_x_values) max(target_x_values)-1]);
ylim([0 1.05]);
ax = gca;
ax.FontSize = plot_font_size;
grid off;

% Customize XTickLabels: label x=0 as '1', x=1 as '2', etc.
original_ticks = get(ax, 'XTick');
new_tick_labels = cell(size(original_ticks));
for k_tick = 1:length(original_ticks)
    tick_val = original_ticks(k_tick);
    if tick_val < 0
        new_tick_labels{k_tick} = num2str(tick_val);
    else
        new_tick_labels{k_tick} = num2str(tick_val + 1);
    end
end
set(ax, 'XTick', original_ticks, 'XTickLabel', new_tick_labels);

if ~isempty(plot_handles) && ~isempty(active_legend_labels)
    legend(plot_handles, active_legend_labels, 'Location', 'eastoutside', 'FontSize', plot_font_size - 2);
end

% Create a new folder for saving files
folder_name = 'FL_LearningCurvesOutput';
if ~exist(folder_name, 'dir')
    mkdir(folder_name);
end

% Save the figure as an EPS file
eps_file_path = fullfile(folder_name, 'FL_learning_curves.eps');
print(gcf, eps_file_path, '-depsc', '-vector');

% Save the average performance data as a CSV file
csv_file_path = fullfile(folder_name, 'FL_average_performance.csv');
writetable(array2table(subject_avg_curves), csv_file_path);

end

function plot_flexlearning_learningcurves_RT(metrics_mt)
% PLOT_FLEXLEARNING_LEARNINGCURVES_RT Generates a reaction time learning curve plot
% for a predefined list of subjects with specific visual styling.
%
% MODIFIED: This function now saves the figure as PNG and EPS files, and the
% underlying plot data as a CSV file, into a dedicated folder.
%
% Args:
%   metrics_mt (cell array): Learning data structure. metrics_mt{iS}
%                            is expected to correspond to the i-th subject
%                            in the hardcoded list.

% --- Configuration ---
target_x_values = -4:25;
plot_font_size = 14;
figure_width = 500;
figure_height = 300;
subject_id_list_hardcoded = {
    'Bard';   'Frey';   'Igor';
    'Reider'; 'Sindri'; 'Wotan'
};
num_hardcoded_subject_details = length(subject_id_list_hardcoded);
custom_colors_rgb = {
    [245, 124, 110]/255;  [242, 181, 110]/255;  [251, 231, 158]/255;
    [132, 195, 183]/255;  [136, 215, 218]/255;  [113, 184, 237]/255;
    [184, 174, 234]/255;  [242, 168, 218]/255
};
num_defined_custom_colors = length(custom_colors_rgb);
num_subjects_to_plot = length(metrics_mt);
if num_subjects_to_plot == 0
    disp('Input metrics_mt is empty. No subjects to plot.');
    return;
end

% --- Data Extraction and Processing ---
subject_avg_curves = nan(num_subjects_to_plot, length(target_x_values));
subject_legend_labels = cell(1, num_subjects_to_plot);
for iS = 1:num_subjects_to_plot
    if iS <= num_hardcoded_subject_details
        subject_name_from_list = subject_id_list_hardcoded{iS};
        subject_name_parts = strsplit(subject_name_from_list, '_');
        subject_name = subject_name_parts{1};
        if ~isempty(subject_name)
            subject_initial = upper(subject_name(1));
            subject_legend_labels{iS} = sprintf('Subject %s', subject_initial);
        else
            subject_legend_labels{iS} = ['Subject ' num2str(iS)];
        end
    else
        subject_legend_labels{iS} = ['Subject ' num2str(iS)];
    end
    
    block_curves_for_this_subject = [];
    if isempty(metrics_mt{iS})
        fprintf('Data for Subject %s (Index %d) is empty, skipping.\n', subject_legend_labels{iS}, iS);
        continue;
    end
    
    for iD = 1:length(metrics_mt{iS})
        session_data = metrics_mt{iS}(iD);
        if ~isfield(session_data, 'learningCurveRTRaw_fl') || isempty(session_data.learningCurveRTRaw_fl)
            continue;
        end
        
        for j = 1:length(session_data.learningCurveRTRaw_fl)
            single_block_lc = session_data.learningCurveRTRaw_fl{j};
            if ~isempty(single_block_lc) && isnumeric(single_block_lc) && ~all(isnan(single_block_lc))
                if iscolumn(single_block_lc)
                    single_block_lc = single_block_lc';
                end
                processed_block_lc = nan(1, length(target_x_values));
                num_points_to_copy = min(length(single_block_lc), length(target_x_values));
                processed_block_lc(1:num_points_to_copy) = single_block_lc(1:num_points_to_copy);
                block_curves_for_this_subject = [block_curves_for_this_subject; processed_block_lc];
            end
        end
    end
    
    if ~isempty(block_curves_for_this_subject)
        subject_avg_curves(iS, :) = nanmean(block_curves_for_this_subject, 1);
    else
        fprintf('Subject %s (Index %d) had no valid curves.\n', subject_legend_labels{iS}, iS);
    end
end

% --- Plotting ---
screen_size = get(0, 'ScreenSize');
fig_pos_x = (screen_size(3) - figure_width) / 2;
fig_pos_y = (screen_size(4) - figure_height) / 2;
figure('Position', [fig_pos_x, fig_pos_y, figure_width, figure_height]);
hold on;
plot_handles = [];
active_legend_labels = {};
idx_pre_block_end = find(target_x_values == -1, 1);
idx_curr_block_start = find(target_x_values == 1, 1);
if isempty(idx_pre_block_end) || isempty(idx_curr_block_start)
    error('Could not find break points -1 or 1 in target_x_values. Check target_x_values definition.');
end
x_segment1 = target_x_values(1:idx_pre_block_end);
x_segment2 = target_x_values(idx_curr_block_start:end);

for iS = 1:num_subjects_to_plot
    if ~all(isnan(subject_avg_curves(iS, :)))
        current_curve_data = subject_avg_curves(iS, :);
        y_segment1 = current_curve_data(1:idx_pre_block_end);
        y_segment2 = current_curve_data(idx_curr_block_start:end);
        
        plot_color = [];
        if iS <= num_defined_custom_colors
            plot_color = custom_colors_rgb{iS};
        else
            color_idx_cycle = mod(iS - 1, num_defined_custom_colors) + 1;
            plot_color = custom_colors_rgb{color_idx_cycle};
        end
        
        h1 = plot(x_segment1, y_segment1, 'LineWidth', 1.5, 'Color', plot_color);
        plot(x_segment2-1, y_segment2, 'LineWidth', 1.5, 'Color', plot_color);
        
        plot_handles = [plot_handles, h1];
        active_legend_labels{end+1} = subject_legend_labels{iS};
    end
end

overall_avg_curve = nanmean(subject_avg_curves, 1);
if ~all(isnan(overall_avg_curve))
    y_avg_segment1 = overall_avg_curve(1:idx_pre_block_end);
    y_avg_segment2 = overall_avg_curve(idx_curr_block_start:end);
    
    h_avg1 = plot(x_segment1, y_avg_segment1, '--k', 'LineWidth', 2);
    plot(x_segment2-1, y_avg_segment2, '--k', 'LineWidth', 2);
    
    plot_handles = [plot_handles, h_avg1];
    active_legend_labels{end+1} = 'Overall Average';
end

y_limits_for_vline = ylim;
line([0 0], y_limits_for_vline, 'Color', [0.6 0.6 0.6], 'LineStyle', '--', 'LineWidth', 1);
hold off;

% --- Aesthetics ---
xlabel('Trial Number', 'FontSize', plot_font_size);
ylabel('Reaction Time (s)', 'FontSize', plot_font_size);
title('Flex Learning: Subject Reaction Time Curves', 'FontSize', plot_font_size + 1);
xlim([min(target_x_values) max(target_x_values)]);
ax = gca;
ax.FontSize = plot_font_size;
grid off;

original_ticks = get(ax, 'XTick');
new_tick_labels = cell(size(original_ticks));
for k_tick = 1:length(original_ticks)
    tick_val = original_ticks(k_tick);
    if tick_val < 0
        new_tick_labels{k_tick} = num2str(tick_val);
    else
        new_tick_labels{k_tick} = num2str(tick_val + 1);
    end
end
set(ax, 'XTick', original_ticks, 'XTickLabel', new_tick_labels);

if ~isempty(plot_handles) && ~isempty(active_legend_labels)
    legend(plot_handles, active_legend_labels, 'Location', 'eastoutside', 'FontSize', plot_font_size - 2);
end

% --- START: MODIFIED SECTION FOR SAVING FILES ---
figure(gcf);
base_filename = 'flexlearning_RT_learning_curve';
date_str = datestr(now, 'yyyymmdd_HHMM');
full_base_filename = [base_filename '_' date_str];

% Define the specific save folder
save_folder = 'FL_RT_Session'; 
if ~exist(save_folder, 'dir')
    mkdir(save_folder);
    fprintf('Created save directory: %s\n', save_folder);
end
filepath_base = fullfile(save_folder, full_base_filename);

% --- Save Plot Data as CSV ---
try
    % Prepare data for the table
    row_labels = [subject_legend_labels, {'Overall Average'}]';
    data_to_save = [subject_avg_curves; overall_avg_curve];
    
    % Create valid column headers from target_x_values (e.g., 'Trial_n4', 'Trial_1')
    col_headers = cell(1, length(target_x_values));
    for i = 1:length(target_x_values)
        col_headers{i} = ['Trial_' strrep(num2str(target_x_values(i)), '-', 'n')];
    end
    
    % Create and save the table
    T_curves = array2table(data_to_save, 'VariableNames', col_headers);
    T_curves = [table(row_labels, 'VariableNames', {'Subject'}), T_curves];
    
    csv_filename = [filepath_base '_avg_RT_curves.csv'];
    writetable(T_curves, csv_filename);
    fprintf('Plot data saved as: %s\n', csv_filename);
    
catch ME_csv
    fprintf('Error saving CSV data: %s\n', ME_csv.message);
end

% --- Save Figure (PNG and EPS) ---
try
    % Save as PNG
    png_filename = [filepath_base '.png'];
    saveas(gcf, png_filename);
    fprintf('Figure saved as: %s\n', png_filename);
    
    % Save as EPS
    eps_filename = [filepath_base '.eps'];
    print(gcf, eps_filename, '-depsc', '-painters');
    fprintf('Figure saved as: %s\n', eps_filename);
    
catch ME_fig
    fprintf('Error saving figure: %s\n', ME_fig.message);
end
% --- END: MODIFIED SECTION FOR SAVING FILES ---

end