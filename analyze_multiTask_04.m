
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



% Example:
% A cell array of required variable names for analysis, matching the
% final output_table from the generate_behavioral_summary_table function.

required_vars = {
    % --- Continuous Recognition (CR) Task ---
	'CR_RT_Slope_Stim2to20',
    'CR_RT_Intercept_Stim2to20',     
    'CR_Acc_Slope_Stim2to20',
    'CR_AvgTrialsToError',
    'CR_EstStimAt75Acc',
	'CR_nBackSlope',
	'CR_nBackSlopeOfSlope',
    

    % --- Flexible Learning (FL) Task ---
    'FL_RT_Overall',
	'FL_Plateau_Accuracy',  
	'FL_Plateau_Accuracy_L3vsL1',
	'FL_Plateau_Accuracy_G4vsG2',
	'FL_Plateau_Accuracy_EDvsID',
	'FL_Plateau_Accuracy_NewvsSame',
	'FL_CEn_Proportion',     
    'FL_CEn_Proportion_L3vsL1',
    'FL_CEn_Proportion_G4vsG2',
    'FL_CEn_Proportion_EDvsID',
    'FL_CEn_Proportion_NewvsSame',
	'FL_Trials_To_Criterion',        
    'FL_Trials_To_Criterion_L3vsL1',
    'FL_Trials_To_Criterion_G4vsG2',
    'FL_Trials_To_Criterion_EDvsID',
    'FL_Trials_To_Criterion_NewvsSame',
              
   
    
	
    % --- Working Memory (WM) Task ---
    'WM_RT_Above05_Overall',         
    'WM_RT_Intercept',     
	'WM_RTDiff_NoPSD_PSDPres',
	'WM_RTDiff_2D3D',
    'WM_RTDiff_LowHighTDS',
    'WM_Acc_SlopeVsDelay',
    'WM_AccDiff_NoPSD_PSDPres',
    'WM_AccDiff_2D3D',
    'WM_AccDiff_LowHighTDS',
    'WM_Acc_Overall',
    
    % --- Anti-Saccade (AS) Task ---
    'AS_Accuracy_Anti',
	'AS_Accuracy_Overall',
    'AS_RTDiff_ProAnti',
    'AS_RTDiff_CongIncong',
    'AS_AccuracyDiff_ProAnti',
    'AS_AccuracyDiff_CongIncong',
    'AS_RT_Overall'
    
};





output_table = generate_behavioral_summary_table(metrics_mt);
modified_output_table = apply_performance_score_transform2(output_table);
renamed_modified_output_table = rename_variables(modified_output_table);




%% overall heatmap + dendrogram
% [R_matrix, P_matrix, variable_labels, fig_handle] = plot_rmc_heatmap_matlab(modified_output_table, required_vars);



% plotSubjectDifferences(output_table);
% [R_matrix_observed, P_matrix_perm, Z_matrix_perm, heatmap_labels, fig_handle_zscore, R_perm_distributions] =  plot_rmc_heatmap_permutation(output_table, required_vars,2000,'PlotExampleDistributionForPair', {'WM_AccDiff_LowHighTDS', 'AS_AccuracyDiff_CongIncong'});
 

% 
% [role_task_variable_names_original_order, role_task_variable_names_sorted] = rename_variables_with_roles(required_vars);
% renamed_table = rename_table_columns(modified_output_table, required_vars, role_task_variable_names_original_order);
% [R_matrix_observed, fig_handle] = plot_significant_rmc_heatmap(modified_output_table, required_vars, 0.05,'task_sorted_r_values.png');
% [fig_dendro, leaf_order] = plot_correlation_dendrogram_from_R(R_matrix_observed, required_vars)
% [R_matrix_observed_role_sorted, fig_handle] = plot_significant_rmc_heatmap(renamed_table, role_task_variable_names_sorted, 0.05, 'role_sorted_r_values.png');
% [fig_dendro, leaf_order] = plot_correlation_dendrogram_from_R(R_matrix_observed_role_sorted, role_task_variable_names_sorted)



%% Individual scatter and heatmap
% plot_behavioral_scatter_grid_matlab(output_table, required_vars)

% plot_subject_correlation_heatmaps(output_table, required_vars)






%% Try to remove Frey's data
% frey_data_removed_modified_output_table = modified_output_table(~strcmp(modified_output_table.SubjectID, 'Frey'), :);
% frey_data_removed_renamed_output_table = renamed_table(~strcmp(renamed_table.SubjectID, 'Frey'), :);
% [R_matrix_observed, fig_handle] = plot_significant_rmc_heatmap(frey_data_removed_modified_output_table, required_vars, 0.05,'task_sorted_r_values_NoFrey.png');
% [fig_dendro, leaf_order] = plot_correlation_dendrogram_from_R(R_matrix_observed, required_vars)
% [R_matrix_observed_role_sorted, fig_handle] = plot_significant_rmc_heatmap(frey_data_removed_renamed_output_table, role_task_variable_names_sorted, 0.05, 'role_sorted_r_values_NoFrey.png');
% [fig_dendro, leaf_order] = plot_correlation_dendrogram_from_R(R_matrix_observed_role_sorted, role_task_variable_names_sorted)






function renamed_table = rename_table_columns(input_table, original_names, new_names)
% RENAME_TABLE_COLUMNS renames specified columns in a table.
%
% This function safely renames columns from an original list to a new list.
% It performs checks to ensure the operation is possible before attempting it.
% If any variable in 'original_names' is not found in the table, the
% function will stop and throw an error.
%
% Args:
%   input_table (table): The MATLAB table to modify.
%   original_names (cell array of strings): The list of current column names
%       that need to be changed.
%   new_names (cell array of strings): The corresponding list of new column
%       names. The order must match 'original_names'.
%
% Returns:
%   renamed_table (table): A new table with the columns renamed.

% --- 1. Input Validation ---
if ~istable(input_table)
    error('rename_table_columns:InvalidInput', 'Input "input_table" must be a MATLAB table.');
end
if ~iscellstr(original_names) && ~isstring(original_names)
    error('rename_table_columns:InvalidInput', 'Input "original_names" must be a cell array of strings.');
end
if ~iscellstr(new_names) && ~isstring(new_names)
    error('rename_table_columns:InvalidInput', 'Input "new_names" must be a cell array of strings.');
end
if numel(original_names) ~= numel(new_names)
    error('rename_table_columns:MismatchSize', 'The number of original names (%d) must match the number of new names (%d).', numel(original_names), numel(new_names));
end
disp('Starting column renaming process...');

% --- 2. Check for Variable Existence ---
% Ensure all original names are present in the table before attempting to rename.
% The ismember function checks for each element of original_names in the table's variable names.
are_vars_present = ismember(original_names, input_table.Properties.VariableNames);

% If not all variables are present, find the first missing one and throw an error.
if ~all(are_vars_present)
    first_missing_idx = find(~are_vars_present, 1, 'first');
    missing_var_name = original_names{first_missing_idx};
    
    % "Error then drop out" as requested.
    error('rename_table_columns:VarNotFound', ...
          'The original variable name ''%s'' was not found in the input table. Aborting operation.', missing_var_name);
end

% --- 3. Perform Renaming ---
% If all checks passed, it's safe to rename the variables.
% renamevars is the recommended and most efficient function for this.
renamed_table = renamevars(input_table, original_names, new_names);

fprintf('Successfully renamed %d columns.\n', numel(original_names));
disp('Column renaming process complete.');

end


function [role_task_variable_names_original_order, role_task_variable_names_sorted] = rename_variables_with_roles(required_vars)
% RENAME_VARIABLES_WITH_ROLES modifies a list of variable names based on predefined rules.
%
% This function performs two main modifications:
%   1. Adds a two-letter cognitive role prefix to each variable.
%   2. Renames the 'WM' task prefix to 'DMTS'.
%
% Args:
%   required_vars (cell array): The original list of variable names.
%
% Returns:
%   role_task_variable_names_original_order (cell array): The new list of variable
%       names, maintaining the original order.
%   role_task_variable_names_sorted (cell array): The new list of variable
%       names, sorted alphabetically.

disp('Mapping variables to cognitive roles and renaming tasks...');

% Define the mapping from variable name to role prefix
% Roles: IC=Inhibitory Control, PS=Processing Speed, UP=Updating,
%        VL=Valence, SS=Set Shifting, WM=Working Memory, XX=Unassigned
keys = { ...
    'CR_RT_Intercept_Stim2to20', 'CR_Acc_Slope_Stim2to20', 'CR_RT_Slope_Stim2to20', ...
    'CR_AvgTrialsToError', 'CR_EstStimAt75Acc', 'CR_nBackSlope' , 'CR_nBackSlopeOfSlope', ...
    'FL_RT_Overall', 'FL_CEn_Proportion', 'FL_CEn_Proportion_L3vsL1', ...
    'FL_CEn_Proportion_G4vsG2', 'FL_CEn_Proportion_EDvsID', 'FL_CEn_Proportion_NewvsSame', ...
    'FL_Trials_To_Criterion', 'FL_Trials_To_Criterion_L3vsL1', 'FL_Trials_To_Criterion_G4vsG2', ...
    'FL_Trials_To_Criterion_EDvsID', 'FL_Trials_To_Criterion_NewvsSame', 'FL_Plateau_Accuracy', ...
    'WM_RT_Above05_Overall', 'WM_RT_Intercept', 'WM_Acc_SlopeVsDelay', ...
    'WM_AccDiff_NoPSD_PSDPres', 'WM_AccDiff_2D3D', 'WM_AccDiff_LowHighTDS', ...
    'WM_Acc_Overall', 'AS_Accuracy_Anti', 'AS_Accuracy_Overall', ...
    'AS_RTDiff_ProAnti', 'AS_RTDiff_CongIncong', 'AS_AccuracyDiff_ProAnti', ...
    'AS_AccuracyDiff_CongIncong', 'AS_RT_Overall',...
    };

values = { ...
    'PS', 'UP', 'UP', 'UP', 'UP', 'IC', 'IC', ... % CR variables
    'PS', 'UP', 'VL', 'VL', 'IC', 'IC', ... % FL CEn variables
    'SS', 'VL', 'VL', 'IC', 'IC', 'IC', ... % FL Trials & Plateau variables
    'WM', 'PS', 'WM', 'IC', 'IC', 'IC', ... % WM variables
    'WM', 'IC', 'WM', 'IC', 'IC', 'IC', ... % AS variables
    'IC', 'PS', ...
    };

% Create a dictionary-like map for efficient lookup
role_map = containers.Map(keys, values);

% Initialize the output cell array
num_vars = numel(required_vars);
role_task_variable_names_original_order = cell(num_vars, 1);

% Process each variable in the original list
for i = 1:num_vars
    original_name = required_vars{i};
    
    % 1. Get the role prefix from the map
    if isKey(role_map, original_name)
        role_prefix = role_map(original_name);
    else
        role_prefix = 'XX'; % Assign 'XX' if variable is not in the map
    end
    
    % 2. Rename task from WM to DMTS
    modified_name = original_name;
    if startsWith(modified_name, 'WM_')
        modified_name = strrep(modified_name, 'WM_', 'DMTS_');
    end
    
    % 3. Construct the new variable name
    new_name = [role_prefix, '_', modified_name];
    role_task_variable_names_original_order{i} = new_name;
    
    fprintf('Transformed: ''%s'' -> ''%s''\n', original_name, new_name);
end

% Create the second output list, sorted alphabetically
role_task_variable_names_sorted = sort(role_task_variable_names_original_order);

disp('Variable renaming complete.');

end







function [R_matrix_observed, P_matrix_parametric, fig_handle] = plot_significant_rmc_heatmap(output_table, required_variables, p_threshold, figure_name)
% PLOT_SIGNIFICANT_RMC_HEATMAP Generates a filtered RMC heatmap showing only
% significant correlations, with task-based grouping borders.
%
% Features:
% - Calculates observed RMC R-values and their parametric p-values.
% - Displays only R-values where the p-value is below a specified threshold.
% - Renders non-significant cells in a light gray.
% - Automatically detects task groups based on variable name prefixes (e.g., 'CR_')
%   and draws black borders around them.
%
% Inputs:
%   output_table (table): MATLAB table with 'SubjectID' and behavioral variables.
%   required_variables (cell array of strings): Numeric variables for RMC.
%   p_threshold (double, optional): Significance level. Default is 0.05.
%
% Outputs:
%   R_matrix_observed (matrix): Matrix of all observed RMC R-values.
%   P_matrix_parametric (matrix): Matrix of corresponding parametric p-values.
%   fig_handle (figure handle): Handle to the generated heatmap figure.

% --- 1. Initialization and Input Validation ---
if nargin < 3
    p_threshold = 0.05;
end

fig_handle = [];
if ~istable(output_table)
    error('Input "output_table" must be a MATLAB table.');
end
% ... (rest of validation from previous function) ...
required_variables = cellstr(required_variables);
var_types = varfun(@class, output_table, 'InputVariables', required_variables, 'OutputFormat','cell');
numeric_mask = cellfun(@(x) ismember(x, {'double', 'single', 'logical'}), var_types);
numeric_vars_for_heatmap = required_variables(numeric_mask);
if numel(numeric_vars_for_heatmap) < 2
    disp('Not enough numeric variables to create a heatmap.');
    R_matrix_observed=[]; P_matrix_parametric=[];
    return;
end
disp(['Using p-value threshold: ', num2str(p_threshold)]);


% --- 2. Data Cleaning ---
% (This section is identical to the previous function)
dataTable_cleaned = output_table;
dataTable_cleaned.Properties.RowNames = {};
dataTable_cleaned = rmmissing(dataTable_cleaned, 'DataVariables', unique([numeric_vars_for_heatmap(:)', {'SubjectID'}]));
if height(dataTable_cleaned) < 3
    disp('Not enough valid data rows after filtering for NaN/missing values.');
    R_matrix_observed=[]; P_matrix_parametric=[];
    return;
end
dataTable_cleaned.SubjectID = categorical(dataTable_cleaned.SubjectID);


% --- 3. Calculate OBSERVED RMC and Parametric P-values ---
num_vars = numel(numeric_vars_for_heatmap);
R_matrix_observed = NaN(num_vars, num_vars);
P_matrix_parametric = NaN(num_vars, num_vars); % Matrix to store p-values

disp('Calculating Observed RMCs and P-values...');
for i = 1:num_vars
    for j = 1:i
        if i == j
            R_matrix_observed(i,j) = 1;
            P_matrix_parametric(i,j) = 0;
            continue;
        end
        var1_name = numeric_vars_for_heatmap{i};
        var2_name = numeric_vars_for_heatmap{j};
        
        % Use the local RMC function that returns p-values
        [r_val, p_val] = local_calculate_rmc_with_p(dataTable_cleaned.(var1_name), ...
            dataTable_cleaned.(var2_name), dataTable_cleaned.SubjectID);
        
        R_matrix_observed(i,j) = r_val;
        P_matrix_parametric(i,j) = p_val;
    end
end
disp('Calculation complete.');


% --- 4. Create the Filtered Heatmap ---
heatmap_labels = strrep(numeric_vars_for_heatmap, '_', ' ');
fig_handle = figure('Name', 'Significant RMC R-values with Task Grouping', 'Color','w', ...
    'Units','normalized','OuterPosition',[0.25 0.1 0.5 0.8]);
ax_handle = axes(fig_handle);

% Prepare data for plotting: only show R-values if p < threshold
R_display_for_plot = NaN(num_vars, num_vars);
for r_idx = 1:num_vars
    for c_idx = 1:r_idx-1 % Strictly lower triangle
        if P_matrix_parametric(r_idx, c_idx) < p_threshold
            R_display_for_plot(r_idx, c_idx) = R_matrix_observed(r_idx, c_idx);
        end
    end
end

% Draw the heatmap image
im_r = imagesc(ax_handle, R_display_for_plot);
axis(ax_handle, 'image');
im_r.AlphaData = ~isnan(R_display_for_plot);
ax_handle.Color = [0.92 0.92 0.92]; % Light gray for non-significant cells

% Apply aesthetics
custom_cmap = diverging_cmap([0 0 1], [1 1 1], [1 0 0]);
colormap(ax_handle, custom_cmap);
clim(ax_handle, [-1 1]);
h_cb = colorbar(ax_handle);
ylabel(h_cb, 'R-value');
title(ax_handle, ['Observed RMC R-values (p < ' num2str(p_threshold) ')']);
xlabel(ax_handle, 'Variables');
ylabel(ax_handle, 'Variables');
xticks(ax_handle, 1:num_vars);
xticklabels(ax_handle, heatmap_labels);
xtickangle(ax_handle, 45);
yticks(ax_handle, 1:num_vars);
yticklabels(ax_handle, heatmap_labels);

% Add text labels ONLY for significant cells
for r_text = 1:num_vars
    for c_text = 1:r_text-1
        r_val = R_display_for_plot(r_text, c_text);
        if ~isnan(r_val)
            text_str = sprintf('%.2f', r_val);
            th = text(ax_handle, c_text, r_text, text_str, ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 8, 'FontWeight', 'bold');
            if r_val > 0.6 || r_val < -0.6; th.Color = 'white'; else; th.Color = 'black'; end
        end
    end
end

% --- 5. Add Task Group Borders ---
hold(ax_handle, 'on');
% Extract task prefixes (e.g., 'CR', 'FL', 'WM', 'AS')
task_prefixes = cellfun(@(x) strtok(x, '_'), numeric_vars_for_heatmap, 'UniformOutput', false);
[unique_tasks, ~, task_indices] = unique(task_prefixes, 'stable');

for k = 1:numel(unique_tasks)
    indices_for_task = find(task_indices == k);
    if isempty(indices_for_task), continue; end
    
    start_idx = min(indices_for_task);
    end_idx = max(indices_for_task);
    block_size = end_idx - start_idx + 1;
    
    if block_size > 0
        rectangle(ax_handle, 'Position', [start_idx-0.5, start_idx-0.5, block_size, block_size], ...
            'EdgeColor', 'k', 'LineWidth', 2, 'LineStyle', '-');
    end
end
hold(ax_handle, 'off');
drawnow;


% --- 6. (Optional) Save the Figure ---
disp('Saving the figure to a high-quality PNG file...');

% Define the output filename and the desired resolution in Dots Per Inch (DPI).
% 300 DPI is standard for most publications. Use 600 for higher quality.
output_filename = figure_name;
resolution_dpi = 300;

% Use the print command to save the figure with the specified properties.
% The '-dpng' flag specifies a PNG file format.
% The '-r' flag sets the resolution.
print(fig_handle, output_filename, '-dpng', ['-r' num2str(resolution_dpi)]);

fprintf('Figure successfully saved to: %s\n', fullfile(pwd, output_filename));

end



function [r, p_val] = local_calculate_rmc_with_p(v1, v2, subjects_cat)
    % Calculates RMC R-value and its parametric p-value.
    v1 = v1(:); v2 = v2(:); subjects_cat = subjects_cat(:);
    r = NaN; p_val = NaN;

    % De-mean the data within each subject
    unique_subj_ids = unique(subjects_cat);
    demeaned_v1 = []; demeaned_v2 = [];
    N_total_obs = 0; num_subjects_contributing = 0;

    for i_s = 1:numel(unique_subj_ids)
        subj_id = unique_subj_ids(i_s);
        idx_s = (subjects_cat == subj_id);
        if sum(idx_s) >= 2
            demeaned_v1 = [demeaned_v1; v1(idx_s) - mean(v1(idx_s))];
            demeaned_v2 = [demeaned_v2; v2(idx_s) - mean(v2(idx_s))];
            N_total_obs = N_total_obs + sum(idx_s);
            num_subjects_contributing = num_subjects_contributing + 1;
        end
    end
    
    if isempty(demeaned_v1) || num_subjects_contributing < 1, return; end
    if std(demeaned_v1, 'omitnan') < eps || std(demeaned_v2, 'omitnan') < eps, return; end

    % Calculate correlation and p-value on de-meaned data
    [corr_matrix, ~] = corrcoef(demeaned_v1, demeaned_v2, 'Rows','complete');
    if numel(corr_matrix) < 4, return; end
    r = corr_matrix(1,2);
    if isnan(r), return; end
    
    df = N_total_obs - num_subjects_contributing - 1;
    if df <= 0, return; end
    
    t_stat = r * sqrt(df / (1 - r^2));
    p_val = 2 * tcdf(-abs(t_stat), df);
end



function [R_matrix_observed, P_matrix_parametric, P_matrix_perm, Z_matrix_perm, heatmap_labels, fig_handle_zscore, fig_handle_r_parametric_p, fig_handle_r_combined_p, R_perm_distributions] = ...
    plot_rmc_heatmap_permutation(output_table, required_variables, num_permutations, varargin)
% PLOT_RMC_HEATMAP_PERMUTATION Generates Z-score and R-score heatmaps (lower triangle) 
% of Repeated Measures Correlations (RMC) with p-values from permutation testing and parametric tests.
%
% Features:
% - Calculates observed RMC R-values and their parametric p-values.
% - Performs permutation testing to generate null distributions and permutation p-values/Z-scores.
% - Figure 1: Z-scores (permutation-based) in lower-left triangle. Boxed if perm. p < 0.05.
% - Figure 2: Observed R-scores in lower-left triangle. Boxed if parametric p < 0.05.
% - Figure 3: Observed R-scores in lower-left triangle. Boxed if BOTH perm. p < 0.05 AND parametric p < 0.05.
% - Uses imagesc for heatmaps and a diverging colormap.
% - Optionally plots the permutation distribution for a specified variable pair.
%
% Inputs:
%   output_table (table): MATLAB table with 'SubjectID' and behavioral variables.
%   required_variables (cell array of strings): Numeric variables for RMC.
%   num_permutations (integer): Number of permutations to perform for each pair.
%   varargin: Optional arguments:
%       'PlotExampleDistributionForPair' (cell array of 2 strings): 
%           e.g., {'VarA', 'VarB'}. Plots histogram of permuted R-values for this pair.
%
% Outputs:
%   R_matrix_observed (matrix): Observed Repeated Measures Correlation R-values.
%   P_matrix_parametric (matrix): Parametric p-values for observed R-values.
%   P_matrix_perm (matrix): Permutation-based p-values for observed R-values.
%   Z_matrix_perm (matrix): Z-scores from permutation distributions.
%   heatmap_labels (cell array of strings): Labels for heatmap axes.
%   fig_handle_zscore (figure handle): Handle to the Z-score heatmap (Fig 1).
%   fig_handle_r_parametric_p (figure handle): Handle to R-score heatmap (parametric p-value boxing, Fig 2).
%   fig_handle_r_combined_p (figure handle): Handle to R-score heatmap (combined p-value boxing, Fig 3).
%   R_perm_distributions (cell matrix): Stores permuted R-value distributions for each pair.

% --- 0. Parse Optional Arguments ---
p = inputParser;
addRequired(p, 'output_table', @istable);
addRequired(p, 'required_variables', @(x) iscellstr(x) || isstring(x));
addRequired(p, 'num_permutations', @(x) isnumeric(x) &&isscalar(x) && x > 0 && floor(x) == x);
addParameter(p, 'PlotExampleDistributionForPair', [], @(x) isempty(x) || (iscell(x) && numel(x)==2 && all(cellfun(@ischar, x) | cellfun(@isstring, x))));
parse(p, output_table, required_variables, num_permutations, varargin{:});

output_table = p.Results.output_table;
required_variables = cellstr(p.Results.required_variables); 
num_permutations = p.Results.num_permutations;
plot_example_pair = cellstr(p.Results.PlotExampleDistributionForPair); 

% Initialize all output figure handles to empty in case of early return
fig_handle_zscore = [];
fig_handle_r_parametric_p = [];
fig_handle_r_combined_p = [];

% --- 1. Input Validation ---
if isempty(output_table)
    disp('Warning: output_table is empty.'); 
    R_matrix_observed=[]; P_matrix_parametric=[]; P_matrix_perm=[]; Z_matrix_perm=[]; heatmap_labels={}; R_perm_distributions={}; 
    return; 
end
if numel(required_variables) < 2
    disp('At least two variables are required for pairing.'); 
    R_matrix_observed=[]; P_matrix_parametric=[]; P_matrix_perm=[]; Z_matrix_perm=[]; heatmap_labels={}; R_perm_distributions={}; 
    return; 
end
if ~ismember('SubjectID', output_table.Properties.VariableNames)
    error("'SubjectID' column not found in output_table."); 
end

var_types = varfun(@class, output_table, 'InputVariables', required_variables, 'OutputFormat','cell');
numeric_mask = cellfun(@(x) strcmp(x, 'double') || strcmp(x, 'single') || strcmp(x,'logical'), var_types);
numeric_vars_for_heatmap = required_variables(numeric_mask);

if numel(numeric_vars_for_heatmap) < 2
    disp('Not enough numeric variables found in required_variables to create a heatmap.');
    R_matrix_observed=[]; P_matrix_parametric=[]; P_matrix_perm=[]; Z_matrix_perm=[]; heatmap_labels={}; R_perm_distributions={}; 
    return;
end
disp(['Using numeric variables for RMC heatmap: ', strjoin(numeric_vars_for_heatmap, ', ')]);

% --- 2. Initial Data Cleaning ---
dataTable = output_table;
if ismember('SessionDateTime', dataTable.Properties.VariableNames)
    sdt = dataTable.SessionDateTime;
    rows_to_remove_sdi = false(height(dataTable), 1);
    if iscellstr(sdt) || isstring(sdt)
        if iscellstr(sdt); sdt = string(sdt); end
        rows_to_remove_sdi = contains(sdt, "SessionIndex", "IgnoreCase", true);
    elseif iscategorical(sdt)
        rows_to_remove_sdi = contains(string(sdt), "SessionIndex", "IgnoreCase", true);
    end
    dataTable = dataTable(~rows_to_remove_sdi, :);
    if isempty(dataTable)
        disp('No data after SessionIndex filter.'); 
        R_matrix_observed=[]; P_matrix_parametric=[]; P_matrix_perm=[]; Z_matrix_perm=[]; heatmap_labels={}; R_perm_distributions={}; 
        return; 
    end
end

vars_to_check_nan = unique([numeric_vars_for_heatmap(:)', {'SubjectID'}]);
missing_any_required = false(height(dataTable), 1);
for i_var_check = 1:numel(vars_to_check_nan)
    current_var_data = dataTable.(vars_to_check_nan{i_var_check});
    if isnumeric(current_var_data) || islogical(current_var_data)
        missing_any_required = missing_any_required | isnan(current_var_data);
    elseif iscategorical(current_var_data) || isdatetime(current_var_data) || isduration(current_var_data) || isstring(current_var_data)
        missing_any_required = missing_any_required | ismissing(current_var_data);
    elseif iscell(current_var_data) 
        missing_any_required = missing_any_required | cellfun(@(x) (ischar(x) && isempty(x)) || (isnumeric(x) && isnan(x)) || (isstring(x) && ismissing(x)) || (isa(x,'missing')), current_var_data);
    end
end
dataTable_cleaned = dataTable(~missing_any_required, :);

if height(dataTable_cleaned) < 2 
    disp('Not enough data after NaN/missing filter for relevant columns.');
    R_matrix_observed=[]; P_matrix_parametric=[]; P_matrix_perm=[]; Z_matrix_perm=[]; heatmap_labels={}; R_perm_distributions={}; 
    return;
end

if isnumeric(dataTable_cleaned.SubjectID) || islogical(dataTable_cleaned.SubjectID)
    dataTable_cleaned.SubjectID = categorical(dataTable_cleaned.SubjectID);
elseif iscellstr(dataTable_cleaned.SubjectID) || isstring(dataTable_cleaned.SubjectID)
    dataTable_cleaned.SubjectID = categorical(string(dataTable_cleaned.SubjectID)); 
elseif ~iscategorical(dataTable_cleaned.SubjectID)
    try
        dataTable_cleaned.SubjectID = categorical(dataTable_cleaned.SubjectID);
    catch
        error('SubjectID must be convertible to categorical.');
    end
end

% --- 3. Calculate OBSERVED RMC and Parametric P-values for each pair ---
num_vars = numel(numeric_vars_for_heatmap);
R_matrix_observed = NaN(num_vars, num_vars);
P_matrix_parametric = NaN(num_vars, num_vars); % MODIFIED: Store parametric p-values

disp('Calculating Observed Repeated Measures Correlations and Parametric P-values...');
for i = 1:num_vars % Row index
    for j = 1:i % Column index (up to i for lower triangle and diagonal)
        var1_name = numeric_vars_for_heatmap{i};
        var2_name = numeric_vars_for_heatmap{j};
        if i == j
            R_matrix_observed(i,j) = 1; 
            P_matrix_parametric(i,j) = 0; % p=0 for self-correlation
            continue;
        end
        
        v1_data = dataTable_cleaned.(var1_name);
        v2_data = dataTable_cleaned.(var2_name);
        subjects_data = dataTable_cleaned.SubjectID;
        
        [r_val, p_param, ~, ~] = local_calculate_rmc(v1_data, v2_data, subjects_data); 
        
        R_matrix_observed(i,j) = r_val;
        P_matrix_parametric(i,j) = p_param; % MODIFIED: Store parametric p-value
        
        if ~isnan(r_val)
            fprintf('Observed RMC for %s vs %s: R=%.3f, Parametric p=%.4f\n', var1_name, var2_name, r_val, p_param);
        else
            fprintf('Observed RMC for %s vs %s: Could not be computed.\n', var1_name, var2_name);
        end
    end
end
disp('Observed RMC and Parametric P-value calculations complete.');

% --- 4. Permutation Test ---
P_matrix_perm = NaN(num_vars, num_vars);
Z_matrix_perm = NaN(num_vars, num_vars);
R_perm_distributions = cell(num_vars, num_vars);

disp('Starting permutation testing for RMC (lower triangle)...');
for i = 1:num_vars      
    for j = 1:i-1   
        var1_name = numeric_vars_for_heatmap{i}; 
        var2_name = numeric_vars_for_heatmap{j}; 

        fprintf('Permutation test for %s (row %d) vs %s (col %d)...\n', var1_name, i, var2_name, j);

        v1_data_original = dataTable_cleaned.(var1_name);
        subjects_data_original = dataTable_cleaned.SubjectID;
        
        r_observed_current = R_matrix_observed(i,j); 
        
        if isnan(r_observed_current)
            P_matrix_perm(i,j) = NaN; 
            Z_matrix_perm(i,j) = NaN; 
            R_perm_distributions{i,j} = NaN(num_permutations, 1);
            fprintf('  Skipping permutation, observed R is NaN.\n');
            continue;
        end

        permuted_r_values_for_pair = NaN(num_permutations, 1);
        
        for k_perm = 1:num_permutations 
            perm_indices = randperm(height(dataTable_cleaned));
            v2_data_shuffled = dataTable_cleaned.(var2_name)(perm_indices); 

            [r_perm, ~, ~, ~] = local_calculate_rmc(v1_data_original, v2_data_shuffled, subjects_data_original);
            permuted_r_values_for_pair(k_perm) = r_perm;
        end
        
        R_perm_distributions{i,j} = permuted_r_values_for_pair;

        valid_perms = permuted_r_values_for_pair(~isnan(permuted_r_values_for_pair));
        num_valid_perms = length(valid_perms);

        if num_valid_perms == 0
            p_val_perm = NaN;
            z_score_perm = NaN;
        else
            p_val_perm = (sum(abs(valid_perms) >= abs(r_observed_current)) + 1) / (num_valid_perms + 1);
            
            mean_perm_r = mean(valid_perms);
            std_perm_r = std(valid_perms);

            if std_perm_r < 1e-9 
                if abs(r_observed_current - mean_perm_r) < 1e-9 
                    z_score_perm = 0;
                else 
                    z_score_perm = sign(r_observed_current - mean_perm_r) * Inf;
                end
            else
                z_score_perm = (r_observed_current - mean_perm_r) / std_perm_r;
            end
        end
        
        P_matrix_perm(i,j) = p_val_perm; 
        Z_matrix_perm(i,j) = z_score_perm; 
        
        fprintf('  Observed R=%.3f, Permutation p=%.4f, Z=%.2f (%d/%d valid perms)\n', ...
            r_observed_current, p_val_perm, z_score_perm, num_valid_perms, num_permutations);
    end
end
for i_diag = 1:num_vars % Fill diagonal for permutation matrices as well
    P_matrix_perm(i_diag, i_diag) = 0; 
    Z_matrix_perm(i_diag, i_diag) = Inf; 
    if isempty(R_perm_distributions{i_diag, i_diag}) % Ensure it's initialized
        R_perm_distributions{i_diag, i_diag} = ones(num_permutations, 1);
    end
end
disp('Permutation testing complete.');


% --- 5. Create Heatmaps ---
heatmap_labels = strrep(numeric_vars_for_heatmap, '_', ' ');

% FIGURE 1: Z-score Heatmap (Permutation p-value boxing)
fig_handle_zscore = figure('Name', 'RMC Permutation Test Z-scores (Lower Triangle)', 'Color','w', ...
    'Units','normalized','OuterPosition',[0.05 0.1 0.3 0.75]); 
ax_zscore = axes(fig_handle_zscore); 
Z_display_for_plot = NaN(num_vars, num_vars); 
cell_text_labels_z = cell(num_vars, num_vars); 

finite_zs_lower_triangle = [];
for r_idx = 1:num_vars
    for c_idx = 1:r_idx-1 
        if ~isinf(Z_matrix_perm(r_idx, c_idx)) && ~isnan(Z_matrix_perm(r_idx, c_idx))
            finite_zs_lower_triangle = [finite_zs_lower_triangle; Z_matrix_perm(r_idx, c_idx)]; %#ok<AGROW>
        end
    end
end
if isempty(finite_zs_lower_triangle)
    max_abs_z_display = 1.96; 
else
    max_abs_z_display = max(abs(finite_zs_lower_triangle));
    if max_abs_z_display == 0; max_abs_z_display = 1.96; end 
end

for r_idx = 1:num_vars
    for c_idx = 1:r_idx-1 
        current_Z = Z_matrix_perm(r_idx, c_idx);
        Z_display_for_plot(r_idx, c_idx) = current_Z; 
        if isinf(current_Z)
            Z_display_for_plot(r_idx, c_idx) = sign(current_Z) * max_abs_z_display * 1.1;
        end
        if ~isnan(current_Z)
            text_str = sprintf('%.2f', current_Z);
            if P_matrix_perm(r_idx, c_idx) < 0.05 
                text_str = [text_str, '*'];
            end
            cell_text_labels_z{r_idx, c_idx} = text_str;
        else
            cell_text_labels_z{r_idx, c_idx} = ''; 
        end
    end
end

im_z = imagesc(ax_zscore, Z_display_for_plot); 
axis(ax_zscore, 'image'); 
im_z.AlphaData = ~isnan(Z_display_for_plot); 
ax_zscore.Color = [0.95 0.95 0.95]; 
custom_cmap = diverging_cmap([0 0 1], [1 1 1], [1 0 0]); % Blue-White-Red
colormap(ax_zscore, custom_cmap);
clim(ax_zscore, [-max_abs_z_display, max_abs_z_display]); 
colorbar(ax_zscore);
title(ax_zscore, 'Permutation Z-scores (Lower Tri, Perm. p < 0.05 boxed*)');
xlabel(ax_zscore, 'Variables'); ylabel(ax_zscore, 'Variables');
xticks(ax_zscore, 1:num_vars); xticklabels(ax_zscore, heatmap_labels); xtickangle(ax_zscore, 45); 
yticks(ax_zscore, 1:num_vars); yticklabels(ax_zscore, heatmap_labels);

for r_text = 1:num_vars
    for c_text = 1:r_text-1 
        if ~isempty(cell_text_labels_z{r_text, c_text})
            val_for_text_color = Z_display_for_plot(r_text, c_text); 
            th = text(ax_zscore, c_text, r_text, cell_text_labels_z{r_text, c_text}, ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 8);
            if contains(cell_text_labels_z{r_text, c_text}, '*'), th.FontWeight = 'bold'; end
            if ~isnan(val_for_text_color) && (val_for_text_color > max_abs_z_display*0.6 || val_for_text_color < -max_abs_z_display*0.6)
                 th.Color = 'white'; else; th.Color = 'black'; end
        end
    end
end
hold(ax_zscore, 'on'); 
for r_box = 1:num_vars
    for c_box = 1:r_box-1 
        if P_matrix_perm(r_box, c_box) < 0.05 && ~isnan(Z_matrix_perm(r_box, c_box))
            rectangle(ax_zscore, 'Position', [c_box-0.5, r_box-0.5, 1, 1], 'EdgeColor', 'k', 'LineWidth', 1.5, 'FaceColor', 'none'); 
        end
    end
end
hold(ax_zscore, 'off'); 
drawnow;


% FIGURE 2: R-score Heatmap (Parametric p-value boxing)
boxing_condition_fig2 = P_matrix_parametric < 0.05;
fig_handle_r_parametric_p = plot_custom_r_heatmap_lower_triangle(R_matrix_observed, boxing_condition_fig2, ...
    'Observed R-scores (Lower Tri, Parametric p < 0.05 boxed*)', ...
    heatmap_labels, num_vars, [0.35 0.1 0.3 0.75]); % New position

% FIGURE 3: R-score Heatmap (Combined Parametric and Permutation p-value boxing)
boxing_condition_fig3 = (P_matrix_parametric < 0.05) & (P_matrix_perm < 0.05);
fig_handle_r_combined_p = plot_custom_r_heatmap_lower_triangle(R_matrix_observed, boxing_condition_fig3, ...
    'Observed R-scores (Lower Tri, Param. & Perm. p < 0.05 boxed*)', ...
    heatmap_labels, num_vars, [0.65 0.1 0.3 0.75]); % New position


% --- 6. (Optional) Plot example permutation distribution ---
if ~isempty(plot_example_pair) && ~isempty(plot_example_pair{1}) 
    var1_name_ex = plot_example_pair{1}; 
    var2_name_ex = plot_example_pair{2}; 

    idx1 = find(strcmp(numeric_vars_for_heatmap, var1_name_ex));
    idx2 = find(strcmp(numeric_vars_for_heatmap, var2_name_ex));

    if ~isempty(idx1) && ~isempty(idx2)
        row_idx_ex = max(idx1, idx2);
        col_idx_ex = min(idx1, idx2);

        if row_idx_ex == col_idx_ex 
             warning('Cannot plot distribution for a variable against itself.');
        else
            r_obs_ex = R_matrix_observed(row_idx_ex, col_idx_ex); 
            p_perm_ex = P_matrix_perm(row_idx_ex, col_idx_ex);   
            dist_ex = R_perm_distributions{row_idx_ex, col_idx_ex};
            
            valid_dist_ex = dist_ex(~isnan(dist_ex));

            if ~isempty(valid_dist_ex)
                fig_dist = figure('Name', ['Permutation Distribution: ' strrep(var1_name_ex,'_',' ') ' vs ' strrep(var2_name_ex,'_',' ')]);
                histogram(valid_dist_ex, min(50, round(length(valid_dist_ex)/10)+1), 'Normalization', 'pdf', 'FaceColor', [0.6 0.6 0.6], 'EdgeColor', [0.3 0.3 0.3]);
                hold on; yl = ylim;
                line([r_obs_ex r_obs_ex], yl, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '--');
                title({sprintf('Permutation Distribution for RMC(%s, %s)', strrep(var1_name_ex,'_',' '), strrep(var2_name_ex,'_',' ')), ...
                       sprintf('Observed R = %.3f, Permutation p = %.4f', r_obs_ex, p_perm_ex)});
                xlabel('Permuted R-values'); ylabel('Density');
                legend('Permuted R Distribution', 'Observed R', 'Location', 'best');
                grid on; box on; hold off;
            else
                warning('Could not plot example distribution for "%s" vs "%s": no valid permuted R values.', var1_name_ex, var2_name_ex);
            end
        end
    else
        warning('Example distribution plot: One or both variables not found: "%s", "%s".', var1_name_ex, var2_name_ex);
    end
end

end % End of main function


% --- Helper function for custom R-value heatmap (lower triangle) ---
function fig_handle = plot_custom_r_heatmap_lower_triangle(R_values, P_boxing_condition, fig_title_str, heatmap_labels, num_vars, fig_position)
    fig_handle = figure('Name', fig_title_str, 'Color','w', ...
        'Units','normalized','OuterPosition', fig_position); 
    ax_handle = axes(fig_handle); 

    R_display_for_plot = NaN(num_vars, num_vars); 
    cell_text_labels_r = cell(num_vars, num_vars); 

    for r_idx = 1:num_vars
        for c_idx = 1:r_idx-1 % Strictly lower triangle
            current_R = R_values(r_idx, c_idx);
            R_display_for_plot(r_idx, c_idx) = current_R; 
            
            if ~isnan(current_R)
                text_str_r = sprintf('%.2f', current_R);
                if P_boxing_condition(r_idx, c_idx) % Check boolean condition directly
                    text_str_r = [text_str_r, '*'];
                end
                cell_text_labels_r{r_idx, c_idx} = text_str_r;
            else
                cell_text_labels_r{r_idx, c_idx} = ''; 
            end
        end
    end

    im_r = imagesc(ax_handle, R_display_for_plot); 
    axis(ax_handle, 'image'); 
    im_r.AlphaData = ~isnan(R_display_for_plot); 
    ax_handle.Color = [0.95 0.95 0.95]; 
    
    custom_cmap = diverging_cmap([0 0 1], [1 1 1], [1 0 0]); % Blue-White-Red
    colormap(ax_handle, custom_cmap);
    clim(ax_handle, [-1 1]); % R-values are between -1 and 1
    
    colorbar(ax_handle);
    title(ax_handle, fig_title_str);
    xlabel(ax_handle, 'Variables'); ylabel(ax_handle, 'Variables');
    xticks(ax_handle, 1:num_vars); xticklabels(ax_handle, heatmap_labels); xtickangle(ax_handle, 45); 
    yticks(ax_handle, 1:num_vars); yticklabels(ax_handle, heatmap_labels);

    for r_text = 1:num_vars
        for c_text = 1:r_text-1 
            if ~isempty(cell_text_labels_r{r_text, c_text})
                val_for_text_color = R_display_for_plot(r_text, c_text); 
                th_r = text(ax_handle, c_text, r_text, cell_text_labels_r{r_text, c_text}, ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 8);
                if contains(cell_text_labels_r{r_text, c_text}, '*'), th_r.FontWeight = 'bold'; end
                % For R-values, clim is [-1, 1]. 0.6 corresponds to 60% of the range.
                if ~isnan(val_for_text_color) && (val_for_text_color > 0.6 || val_for_text_color < -0.6)
                     th_r.Color = 'white'; else; th_r.Color = 'black'; end
            end
        end
    end
    
    hold(ax_handle, 'on'); 
    for r_box = 1:num_vars
        for c_box = 1:r_box-1 
            if P_boxing_condition(r_box, c_box) && ~isnan(R_values(r_box, c_box))
                rectangle(ax_handle, 'Position', [c_box-0.5, r_box-0.5, 1, 1], 'EdgeColor', 'k', 'LineWidth', 1.5, 'FaceColor', 'none'); 
            end
        end
    end
    hold(ax_handle, 'off'); 
    drawnow;
end

% --- Helper function for diverging colormap ---
function cmap = diverging_cmap(neg_color, zero_color, pos_color, num_levels)
    if nargin < 4; num_levels = 100; end
    cmap_neg_half = [linspace(neg_color(1), zero_color(1), num_levels/2)', ...
                     linspace(neg_color(2), zero_color(2), num_levels/2)', ...
                     linspace(neg_color(3), zero_color(3), num_levels/2)'];
    cmap_pos_half = [linspace(zero_color(1), pos_color(1), num_levels/2)', ...
                     linspace(zero_color(2), pos_color(2), num_levels/2)', ...
                     linspace(zero_color(3), pos_color(3), num_levels/2)'];
    cmap = [cmap_neg_half(1:end-1,:); cmap_pos_half]; % Avoid double white
end


% --- Helper function for RMC calculation (nested or separate .m file) ---
function [r, p_parametric, df_rmc, N_total_paired_obs] = local_calculate_rmc(v1, v2, subjects_cat)
    v1 = v1(:); v2 = v2(:); subjects_cat = subjects_cat(:);
    nan_mask_pair = isnan(v1) | isnan(v2) | ismissing(subjects_cat);
    v1_clean_pair = v1(~nan_mask_pair);
    v2_clean_pair = v2(~nan_mask_pair);
    subjects_clean_pair = subjects_cat(~nan_mask_pair);
    
    r = NaN; p_parametric = NaN; df_rmc = NaN; N_total_paired_obs = 0;

    if length(v1_clean_pair) < 3, return; end
    unique_subj_ids_pair = unique(subjects_clean_pair);
    k_subjects_in_pair = numel(unique_subj_ids_pair);
    if k_subjects_in_pair < 1, return; end
    
    demeaned_v1 = []; demeaned_v2 = []; num_subjects_contributing = 0;
    for i_s = 1:k_subjects_in_pair
        subj_id_current = unique_subj_ids_pair(i_s);
        idx_s_current = (subjects_clean_pair == subj_id_current);
        v1_s_current = v1_clean_pair(idx_s_current);
        v2_s_current = v2_clean_pair(idx_s_current);
        if numel(v1_s_current) >= 2 
            demeaned_v1 = [demeaned_v1; v1_s_current - mean(v1_s_current)]; 
            demeaned_v2 = [demeaned_v2; v2_s_current - mean(v2_s_current)]; 
            N_total_paired_obs = N_total_paired_obs + numel(v1_s_current);
            num_subjects_contributing = num_subjects_contributing + 1;
        end
    end
    
    if isempty(demeaned_v1) || num_subjects_contributing < 1, return; end
    if std(demeaned_v1, 'omitnan') < eps || std(demeaned_v2, 'omitnan') < eps
        r = NaN; p_parametric = NaN; 
        df_rmc = N_total_paired_obs - num_subjects_contributing -1; 
        if df_rmc <=0; df_rmc = NaN; end
        return;
    end
    if length(demeaned_v1) < 2, return; end

    [corr_matrix_demeaned, ~] = corrcoef(demeaned_v1, demeaned_v2, 'Rows','complete'); 
    if numel(corr_matrix_demeaned) < 4 || ~ismatrix(corr_matrix_demeaned) || size(corr_matrix_demeaned,1)<2 || size(corr_matrix_demeaned,2)<2
        return;
    end
    r = corr_matrix_demeaned(1,2);
    if isnan(r), return; end
    
    df_rmc = N_total_paired_obs - num_subjects_contributing - 1; 
    if df_rmc <= 0, p_parametric = NaN; return; end
    
    if abs(r) >= 1.0 - 10*eps 
        t_stat = sign(r) * Inf;
    else
        t_stat = r * sqrt(df_rmc / (1 - r^2));
    end
    p_parametric = 2 * tcdf(-abs(t_stat), df_rmc); 
end






function plotSubjectDifferences(output_table)
% plotSubjectDifferences Creates a plot showing subject differences from the mean.
%
%   plotSubjectDifferences(output_table)
%   Takes a MATLAB table 'output_table' as input. The table must contain
%   a 'SubjectID' column and columns for each variable specified in
%   'required_vars'. This version averages multiple entries (sessions) for
%   the same subject, ignoring NaNs. Error bars represent subject-specific
%   SE across sessions. Dots for the same subject are connected by lines.
%
%   The function generates a plot where:
%   - Each row represents a variable.
%   - Points show individual subject deviations from the overall mean of
%     normalized data for that variable.
%   - Horizontal error bars show the subject-specific Standard Error (SE)
%     of their session data for that variable, scaled to the normalized plot.
%   - Lines connect the data points for each subject across variables.
%   - Styling (colors, fonts, layout) is applied.

    % --- Configuration ---
    subject_id_list_hardcoded = {
        'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan'
    }; 
    custom_colors_rgb = { 
        [245, 124, 110]/255; [242, 181, 110]/255; [251, 231, 158]/255;
        [132, 195, 183]/255; [136, 215, 218]/255; [113, 184, 237]/255;
        [184, 174, 234]/255; [242, 168, 218]/255
    };
    required_vars = {
        'FL_CEn_Proportion_SessionMean', 'FL_Plateau_Accuracy_SessionMean', ...
        'FL_Trials_To_Criterion_SessionMean', 'AS_Accuracy_Overall', ...
        'AS_RTDiff_ProAnti', 'AS_AccuracyDiff_ProAnti', ...
        'AS_AccuracyDiff_CongIncong', 'AS_RTDiff_CongIncong', ...
        'CR_EstStimAt75Acc', 'CR_AvgTrialsToError', ...
        'CR_RT_Slope_Stim2to20', 'CR_Acc_Slope_Stim2to20', ...
        'WM_Acc_Overall', 'WM_AccDiff_LowHighTDS', ...
        'WM_AccDiff_2D3D', 'WM_AccDiff_NoPSD_PSDPres', ...
        'WM_Acc_SlopeVsDelay'
    };

    num_subjects_hardcoded = length(subject_id_list_hardcoded);
    if num_subjects_hardcoded > length(custom_colors_rgb)
        warning('Not enough custom colors for all subjects. Adding random colors.');
        for i = (length(custom_colors_rgb)+1):num_subjects_hardcoded
            custom_colors_rgb{i} = rand(1,3);
        end
    end

    % --- Data Preparation and Aggregation ---
    if ~istable(output_table)
        error('Input must be a MATLAB table.');
    end
    if ~ismember('SubjectID', output_table.Properties.VariableNames)
        error('output_table must contain a "SubjectID" column.');
    end

    output_table_subject_ids_raw = output_table.SubjectID;
    if iscategorical(output_table_subject_ids_raw)
        output_table_subject_ids_cell = cellstr(output_table_subject_ids_raw);
    elseif isnumeric(output_table_subject_ids_raw)
        output_table_subject_ids_cell = arrayfun(@num2str, output_table_subject_ids_raw, 'UniformOutput', false);
    elseif iscellstr(output_table_subject_ids_raw)
        output_table_subject_ids_cell = output_table_subject_ids_raw;
    elseif iscell(output_table_subject_ids_raw) && all(cellfun(@(x) ischar(x) || isstring(x), output_table_subject_ids_raw))
        output_table_subject_ids_cell = cellstr(output_table_subject_ids_raw);
    elseif isstring(output_table_subject_ids_raw)
        output_table_subject_ids_cell = cellstr(output_table_subject_ids_raw);
    else
        error('SubjectID column has an unsupported data type.');
    end

    num_req_vars = length(required_vars);
    num_hardcoded_subj = length(subject_id_list_hardcoded);

    aggregated_data_for_hardcoded_subjects = NaN(num_hardcoded_subj, num_req_vars);
    raw_subject_SEs = NaN(num_hardcoded_subj, num_req_vars); % For subject-specific SEs
    is_subject_present_in_table = false(1, num_hardcoded_subj);

    for i = 1:num_hardcoded_subj
        current_subj_id = subject_id_list_hardcoded{i};
        subject_rows_logical_idx = strcmp(output_table_subject_ids_cell, current_subj_id);
        
        if ~any(subject_rows_logical_idx)
            continue; 
        end
        is_subject_present_in_table(i) = true;

        for k = 1:num_req_vars
            var_name = required_vars{k};
            if ~ismember(var_name, output_table.Properties.VariableNames)
                if i == find(is_subject_present_in_table,1,'first') 
                    warning('Variable "%s" not found in output_table. Skipping this variable.', var_name);
                end
                continue;
            end
            
            data_for_current_subject_var_sessions = output_table.(var_name)(subject_rows_logical_idx);
            
            processed_data_for_stats = NaN; % Default for non-numeric
            if isnumeric(data_for_current_subject_var_sessions) || islogical(data_for_current_subject_var_sessions)
                if islogical(data_for_current_subject_var_sessions)
                    processed_data_for_stats = double(data_for_current_subject_var_sessions);
                else
                    processed_data_for_stats = data_for_current_subject_var_sessions;
                end
                
                aggregated_data_for_hardcoded_subjects(i, k) = mean(processed_data_for_stats, 'omitnan');
                
                non_nan_data = processed_data_for_stats(~isnan(processed_data_for_stats));
                n_sessions = length(non_nan_data);
                if n_sessions > 1
                    raw_subject_SEs(i, k) = std(non_nan_data, 0) / sqrt(n_sessions); % std(X,0) is default
                elseif n_sessions == 1
                    raw_subject_SEs(i, k) = 0; % SE of 1 data point is 0
                else 
                    raw_subject_SEs(i, k) = NaN; % No data points
                end
            else
                 if i == find(is_subject_present_in_table,1,'first')
                     warning('Data for variable "%s" for subject "%s" is not numeric/logical. Mean/SE will be NaN.', var_name, current_subj_id);
                 end
                aggregated_data_for_hardcoded_subjects(i, k) = NaN;
                raw_subject_SEs(i, k) = NaN;
            end
        end
    end

    present_subject_ids_in_order = subject_id_list_hardcoded(is_subject_present_in_table);
    data_for_plotting_matrix = aggregated_data_for_hardcoded_subjects(is_subject_present_in_table, :);
    subject_specific_raw_SEs_matrix = raw_subject_SEs(is_subject_present_in_table, :);
    subject_plot_colors = custom_colors_rgb(is_subject_present_in_table);

    if isempty(present_subject_ids_in_order)
        error('No subjects from subject_id_list_hardcoded found in output_table.SubjectID.');
    end
    
    num_present_subjects = length(present_subject_ids_in_order);

    % --- Normalization, Difference Calculation, and SE Scaling ---
    all_subject_differences = NaN(num_present_subjects, num_req_vars);
    subject_specific_normalized_SEs = NaN(num_present_subjects, num_req_vars);

    for k_var_idx = 1:num_req_vars % Iterate through variables
        current_var_subject_means = data_for_plotting_matrix(:, k_var_idx); 
        
        if all(isnan(current_var_subject_means))
            continue; 
        end

        min_val = min(current_var_subject_means); 
        max_val = max(current_var_subject_means);
        range_val = max_val - min_val;
        
        normalized_means_col = NaN(num_present_subjects, 1); 
        if range_val == 0 
            if ~isnan(min_val) 
                 non_nan_indices = ~isnan(current_var_subject_means);
                 normalized_means_col(non_nan_indices) = 0.5;
            end
        else
            normalized_means_col = (current_var_subject_means - min_val) / range_val;
        end
        
        overall_mean_normalized = mean(normalized_means_col, 'omitnan');
        
        if isnan(overall_mean_normalized) 
            all_subject_differences(:, k_var_idx) = NaN;
        else
            all_subject_differences(:, k_var_idx) = normalized_means_col - overall_mean_normalized;
        end
        
        % Scale subject-specific SEs for this variable
        current_raw_SEs_for_var = subject_specific_raw_SEs_matrix(:, k_var_idx);
        temp_norm_SEs_for_var = NaN(num_present_subjects, 1);
        if range_val > 0
            temp_norm_SEs_for_var = current_raw_SEs_for_var / range_val;
        else % range_val is 0 (and current_var_subject_means not all NaN)
            if ~isnan(min_val) 
                for subj_idx = 1:num_present_subjects
                    raw_se_val = current_raw_SEs_for_var(subj_idx);
                    if raw_se_val == 0 
                        temp_norm_SEs_for_var(subj_idx) = 0;
                    elseif ~isnan(raw_se_val) % raw_se_val > 0
                        temp_norm_SEs_for_var(subj_idx) = NaN; % Cannot scale non-zero SE if range is 0
                    else % raw_se_val is NaN
                        temp_norm_SEs_for_var(subj_idx) = NaN;
                    end
                end
            end
        end
        subject_specific_normalized_SEs(:, k_var_idx) = temp_norm_SEs_for_var;
    end
    
    % --- Plotting ---
    fig = figure('Position', [100, 100, 700, 700]); 
    ax = axes('Parent', fig);
    hold(ax, 'on');
    max_jitter = 0.20; 
    plot_handles_for_legend = gobjects(1, num_present_subjects);
    legend_names = cell(1, num_present_subjects);

    for s = 1:num_present_subjects 
        subject_color = subject_plot_colors{s};
        subject_specific_jitters = (rand(1, num_req_vars) - 0.5) * max_jitter;
        
        subject_x_coords_for_line = NaN(1, num_req_vars);
        subject_y_coords_for_line = NaN(1, num_req_vars);

        for k = 1:num_req_vars 
            diff_value = all_subject_differences(s, k);
            if isnan(diff_value)
                continue; 
            end
            
            y_position = k + subject_specific_jitters(k);
            se_value_for_plot = subject_specific_normalized_SEs(s, k);

            h_eb = errorbar(ax, diff_value, y_position, se_value_for_plot, 'horizontal');
            set(h_eb, ...
                'Color', subject_color, ...
                'Marker', 'o', ...
                'MarkerFaceColor', subject_color, ...
                'MarkerSize', 7, ... % Slightly larger marker
                'LineWidth', 1.5, ...
                'CapSize', 5, ...    % Slightly larger cap
                'LineStyle', 'none'); 
            if ~isgraphics(plot_handles_for_legend(s)) 
                plot_handles_for_legend(s) = h_eb;
            end

            subject_x_coords_for_line(k) = diff_value;
            subject_y_coords_for_line(k) = y_position;
        end
        
        % Plot connecting line for the current subject
        valid_line_points = ~isnan(subject_x_coords_for_line);
        if sum(valid_line_points) > 1 % Only plot line if more than one point
            plot(ax, subject_x_coords_for_line(valid_line_points), ...
                 subject_y_coords_for_line(valid_line_points), ...
                 '-', 'Color', subject_color, 'LineWidth', 1.0);
        end

        subj_id_for_legend = present_subject_ids_in_order{s};
        initial = subj_id_for_legend(1); 
        legend_names{s} = sprintf('Subject %s', initial); % Shortened "Subject"
    end

    plot(ax, [0 0], [0.5 num_req_vars + 0.5], 'Color', [0.7 0.7 0.7], 'LineWidth', 1);

    % --- Axes and Figure Configuration ---
    clean_var_names = strrep(required_vars, '_', ' ');
    set(ax, 'YTick', 1:num_req_vars);
    set(ax, 'YTickLabel', clean_var_names);
    set(ax, 'YDir', 'reverse'); 
    ylim(ax, [0.5, num_req_vars + 0.5]);
    set(ax, 'XLim', [-1, 1]);
    set(ax, 'XTick', [-1, -0.5, 0, 0.5, 1]);
    ax.XAxisLocation = 'top'; 
    title('Subject Deviations from Overall Mean (Normalized Data)', 'FontSize', 14); % Added title
    xlabel(ax, 'Normalized Value - Overall Mean', 'FontSize', 14, 'FontWeight', 'bold'); % Added X label


    valid_handles_idx = isgraphics(plot_handles_for_legend);
    if any(valid_handles_idx)
        lgd = legend(ax, plot_handles_for_legend(valid_handles_idx), legend_names(valid_handles_idx), ...
                     'Location', 'eastoutside', 'FontSize', 12); % Adjusted font size
        lgd.Box = 'off'; 
    end
    
    set(ax, 'FontSize', 12); % Adjusted axis font size
    set(fig, 'Color', 'w'); 
    box(ax, 'off');
    grid(ax, 'off');
    
    hold(ax, 'off');
    disp('Plot generated with subject-specific SE and connecting lines.');
end



function [fig_dendro, leaf_order] = plot_correlation_dendrogram_from_R(R_matrix, variable_labels, linkage_method, distance_metric_type)
% PLOT_CORRELATION_DENDROGRAM_FROM_R_MODIFIED Generates a customized dendrogram from a correlation matrix.
%
% This function performs hierarchical clustering, plots the resulting dendrogram,
% and applies specific color, label, and layout formatting.
%
% Inputs:
%   R_matrix (square matrix): The correlation matrix.
%   variable_labels (cell array of strings): Labels for the variables.
%   linkage_method (string, optional): Linkage method. Default: 'average'.
%   distance_metric_type (string, optional): Method to convert correlation to distance. Default: '1-R'.
%
% Outputs:
%   fig_dendro (figure handle): Handle to the generated dendrogram figure.
%   leaf_order (vector): Order of the leaves in the dendrogram.

% --- 1. Input Validation and Defaults ---
if nargin < 2, error('Requires at least R_matrix and variable_labels.'); end
if ~ismatrix(R_matrix) || size(R_matrix,1) ~= size(R_matrix,2), error('R_matrix must be a square matrix.'); end
if ~iscellstr(variable_labels) && ~isstring(variable_labels), error('variable_labels must be a cell array of strings or a string array.'); end
if size(R_matrix,1) ~= numel(variable_labels), error('Number of variable_labels must match the dimensions of R_matrix.'); end
n_vars = size(R_matrix,1);
if n_vars < 2, warning('Cannot create a meaningful dendrogram with less than 2 variables.'); fig_dendro = []; leaf_order = []; return; end
if nargin < 3 || isempty(linkage_method), linkage_method = 'average'; end
if nargin < 4 || isempty(distance_metric_type), distance_metric_type = '1-R'; end

% --- 2. Pre-process Labels ---
variable_labels = strrep(variable_labels, '_', ' ');

% --- 3. Convert Correlation to Distance ---
switch lower(distance_metric_type)
    case '1-r'
        dist_matrix = 1 - R_matrix;
        dist_axis_label = 'Distance (1 - R)';
    case '1-|r|'
        dist_matrix = 1 - abs(R_matrix);
        dist_axis_label = 'Distance (1 - |R|)';
    otherwise
        error('Invalid distance_metric_type. Choose "1-R" or "1-|R|".');
end
dist_matrix(logical(eye(n_vars))) = 0;
dist_vector = squareform(dist_matrix, 'tovector');

% --- 4. Perform Hierarchical Clustering ---
Z = linkage(dist_vector, linkage_method);

% --- 5. Optimize Leaf Order ---
if n_vars > 2
    optimal_order_indices = optimalleaforder(Z, dist_vector, 'Criteria', 'group');
else
    optimal_order_indices = 1:n_vars;
end

% --- 6. Plot Dendrogram and Customize Appearance ---
% NEW: Create figure with a specific size [width, height] in pixels
fig_dendro = figure('Name', ['Correlation Dendrogram | Linkage: ' linkage_method], ...
                  'Color', 'w', ...
                  'Position', [100, 100, 520, 600]); % [left, bottom, width, height]

if n_vars > 1
    % Plot the dendrogram WITHOUT labels to start
    [H, ~, leaf_order] = dendrogram(Z, n_vars, ...
                                'Orientation', 'right', ...
                                'Reorder', optimal_order_indices, ...
                                'Labels', {}); 

    title({['Hierarchical Clustering of Variables']; ...
           ['Linkage: ' linkage_method ', Distance: ' distance_metric_type]});
    xlabel(dist_axis_label);
    % NEW: 'ylabel' has been removed as requested.
    
    % --- CUSTOMIZATION ---

    % 1. Make all dendrogram lines black
    set(H, 'Color', 'k');
    
    ax = gca; % Get current axes handle
    
    % Adjust axes position to create more space for labels on the left.
    pos = get(ax, 'Position'); 
    % NEW: margin_increase is now 0.2
    margin_increase = 0.2; 
    pos(1) = pos(1) + margin_increase;
    pos(3) = pos(3) - margin_increase;
    set(ax, 'Position', pos);

    % 2. Create and place colored labels manually
    yticks = get(ax, 'YTick');
    ordered_labels = variable_labels(leaf_order);

    roles = cellfun(@(s) s(1:2), ordered_labels, 'UniformOutput', false);
    unique_roles = unique(roles, 'stable');
    
    colors = lines(numel(unique_roles));
    role_color_map = containers.Map(unique_roles, num2cell(colors, 2));

    % Clear the original (now invisible) y-ticks
    set(ax, 'YTickLabel', [], 'YTick', []);

    % Place each label manually using colored text objects
    for i = 1:numel(ordered_labels)
        current_label = ordered_labels{i};
        role = current_label(1:2);
        color = role_color_map(role);
        
        % NEW: Place text at x = -0.05
        text(0.05, yticks(i), current_label, ...
            'HorizontalAlignment', 'right', ...
            'VerticalAlignment', 'middle', ...
            'Color', color, ...
            'FontSize', 10, ...
            'Clipping', 'off'); % Allow text to be drawn outside plot box
    end
    
else
    clf;
    text(0.5, 0.5, 'Not enough variables for a dendrogram.', 'HorizontalAlignment', 'center');
    leaf_order = optimal_order_indices;
end

disp('Dendrogram generated and customized.');

end






function fig_handles = plot_subject_correlation_heatmaps(output_table, required_variables)
% PLOT_SUBJECT_CORRELATION_HEATMAPS Generates heatmaps of correlations for each subject using imagesc.
%
% Features:
% - For each subject, calculates Pearson correlations (R-values and p-values) 
%   for pairs of specified variables.
% - Displays R-values in a heatmap (using imagesc) for each subject.
% - Arranges subject heatmaps in 3x2 subplots on A4-sized figures.
% - Only shows color for cells where p-value < 0.05. Other cells are white.
% - Displays R-values in cells. Bolds R-values and adds '*' where p-value < 0.05 (non-diagonal).
%   Diagonal R-values (1.00) are also bolded.
% - Uses a custom 100-level interpolated colormap.
% - Replaces underscores with spaces in axis tick labels.
%
% Inputs:
%   output_table (table): MATLAB table with 'SubjectID' and behavioral variables.
%   required_variables (cell array of strings): Numeric variables for correlation.
%
% Outputs:
%   fig_handles (array): Array of figure handles created.

    % --- 1. Input Validation ---
    if ~istable(output_table)
        error('Input `output_table` must be a MATLAB table.');
    end
    if isempty(output_table)
        disp('Warning: output_table is empty.');
        fig_handles = gobjects(0); % Return empty graphics object array
        return;
    end
    if ~iscellstr(required_variables) && ~isstring(required_variables) % Allow string array too
        error('Input `required_variables` must be a cell array of strings or a string array.');
    end
    if numel(required_variables) < 2
        disp('At least two variables are required for pairing to create a heatmap.');
        fig_handles = gobjects(0);
        return;
    end
    if ~ismember('SubjectID', output_table.Properties.VariableNames)
        error("'SubjectID' column not found in output_table.");
    end
    
    % Convert string array to cell array of strings if necessary for consistency
    if isstring(required_variables)
        required_variables = cellstr(required_variables);
    end

    var_types = varfun(@class, output_table, 'InputVariables', required_variables, 'OutputFormat','cell');
    numeric_mask = cellfun(@(x) strcmp(x, 'double') || strcmp(x, 'single') || strcmp(x,'logical'), var_types);
    numeric_vars_for_heatmap = required_variables(numeric_mask);

    if numel(numeric_vars_for_heatmap) < 2
        disp('Not enough numeric variables found in required_variables to create a heatmap of pairs.'); 
        fig_handles = gobjects(0);
        return;
    end
    disp(['Using numeric variables for correlation heatmaps: ', strjoin(numeric_vars_for_heatmap, ', ')]);

    % --- 2. Initial Data Cleaning (SessionDateTime "SessionIndex" removal) ---
    dataTable = output_table;
    if ismember('SessionDateTime', dataTable.Properties.VariableNames)
        sdt = dataTable.SessionDateTime; 
        rows_to_remove_sdi = false(height(dataTable), 1);
        if iscellstr(sdt) || isstring(sdt) 
            if iscellstr(sdt); sdt = string(sdt); end % Convert cellstr to string array
            rows_to_remove_sdi = contains(sdt, "SessionIndex", "IgnoreCase", true); 
        elseif iscategorical(sdt)
            rows_to_remove_sdi = contains(string(sdt), "SessionIndex", "IgnoreCase", true); 
        end
        dataTable = dataTable(~rows_to_remove_sdi, :);
        if isempty(dataTable)
            disp('No data after SessionIndex filter.');
            fig_handles = gobjects(0);
            return;
        end
    end
    
    % Ensure SubjectID is categorical for consistent processing
    if isnumeric(dataTable.SubjectID) || islogical(dataTable.SubjectID)
        dataTable.SubjectID = categorical(dataTable.SubjectID);
    elseif iscellstr(dataTable.SubjectID) || isstring(dataTable.SubjectID) % Handle if SubjectID is cell or string
        dataTable.SubjectID = categorical(cellstr(dataTable.SubjectID)); % Convert to cellstr then categorical
    elseif ~iscategorical(dataTable.SubjectID)
        error('SubjectID must be convertible to categorical for grouping.'); 
    end

    unique_subjects = unique(dataTable.SubjectID);
    if isempty(unique_subjects)
        disp('No unique subjects found in the table.');
        fig_handles = gobjects(0);
        return;
    end

    num_total_subjects = numel(unique_subjects);
    disp(['Found ', num2str(num_total_subjects), ' unique subjects.']);

    % --- 3. Prepare Colormap (same as original) ---
    anchor_colors_rgb = [
        97/255, 170/255, 207/255;  % C1: Blue (negative extreme)
        152/255, 202/255, 221/255; % C2: Light Blue
        234/255, 239/255, 246/255; % C3: Very Light Blue/Gray (near zero, negative side)
        249/255, 239/255, 239/255; % C4: Very Light Red/Gray (near zero, positive side)
        233/255, 198/255, 198/255; % C5: Light Red
        218/255, 149/255, 153/255   % C6: Red (positive extreme)
    ];
    map_indices = [0, 0.25, 0.499, 0.501, 0.75, 1]; % Positions for C1 to C6
    num_cmap_levels = 100;
    query_indices = linspace(0, 1, num_cmap_levels);
    interp_cmap_r = interp1(map_indices, anchor_colors_rgb(:,1), query_indices, 'linear');
    interp_cmap_g = interp1(map_indices, anchor_colors_rgb(:,2), query_indices, 'linear');
    interp_cmap_b = interp1(map_indices, anchor_colors_rgb(:,3), query_indices, 'linear');
    final_cmap_100 = [interp_cmap_r', interp_cmap_g', interp_cmap_b'];
    final_cmap_100(final_cmap_100 < 0) = 0; % Clip any potential small negative due to interpolation
    final_cmap_100(final_cmap_100 > 1) = 1; % Clip any potential small values > 1

    processed_tick_labels = strrep(numeric_vars_for_heatmap, '_', ' ');
    num_vars = numel(numeric_vars_for_heatmap);

    % --- 4. Loop Through Subjects and Plot ---
    subplots_per_fig = 6; % 3 rows, 2 columns
    current_fig_handle = gobjects(0); % Initialize with empty graphics object
    fig_handles = gobjects(0); % Store all figure handles
    plot_count_on_current_fig = 0;
    subjects_plotted_count = 0;
    min_obs_for_corr_pair = 3; % Min non-NaN pairs for corrcoef to run robustly for a pair

    for i_subj = 1:num_total_subjects
        current_subject_id = unique_subjects(i_subj);
        subject_data_table = dataTable(dataTable.SubjectID == current_subject_id, :);
        
        fprintf('\nProcessing Subject: %s\n', char(current_subject_id));

        if height(subject_data_table) < min_obs_for_corr_pair 
            fprintf('Skipping Subject %s: Insufficient observations overall (has %d, needs at least %d for any pair).\n', ...
                char(current_subject_id), height(subject_data_table), min_obs_for_corr_pair);
            continue;
        end
        
        R_matrix_subject = NaN(num_vars, num_vars);
        P_matrix_subject = NaN(num_vars, num_vars);
        valid_pairs_for_subject = 0;

        for i = 1:num_vars
            for j = i:num_vars 
                var1_name = numeric_vars_for_heatmap{i};
                var2_name = numeric_vars_for_heatmap{j};
                
                v1_data_subj = subject_data_table.(var1_name);
                v2_data_subj = subject_data_table.(var2_name);

                if i == j
                    R_matrix_subject(i,j) = 1;
                    P_matrix_subject(i,j) = 0; 
                    valid_pairs_for_subject = valid_pairs_for_subject + 1;
                    continue;
                end
                
                common_non_nan = ~isnan(v1_data_subj) & ~isnan(v2_data_subj);
                v1_pair_clean = v1_data_subj(common_non_nan);
                v2_pair_clean = v2_data_subj(common_non_nan);

                if numel(v1_pair_clean) < min_obs_for_corr_pair || ...
                   std(v1_pair_clean, 'omitnan') == 0 || std(v2_pair_clean, 'omitnan') == 0 % Added 'omitnan' for std
                    R_matrix_subject(i,j) = NaN; P_matrix_subject(i,j) = NaN;
                    R_matrix_subject(j,i) = NaN; P_matrix_subject(j,i) = NaN;
                    continue; 
                end
                
                try
                    [R_pair, P_pair_values] = corrcoef(v1_data_subj, v2_data_subj, 'Rows', 'pairwise');
                    
                    if numel(R_pair) == 4 
                        r_val = R_pair(1,2);
                        p_val = P_pair_values(1,2);

                        if ~isnan(r_val)
                            R_matrix_subject(i,j) = r_val;
                            R_matrix_subject(j,i) = r_val; 
                            P_matrix_subject(i,j) = p_val;
                            P_matrix_subject(j,i) = p_val; 
                            valid_pairs_for_subject = valid_pairs_for_subject + 1;
                        end
                    end
                catch ME
                    fprintf('  Subject %s: Error calculating corr for %s vs %s: %s\n', char(current_subject_id), var1_name, var2_name, ME.message);
                    R_matrix_subject(i,j) = NaN; P_matrix_subject(i,j) = NaN; % Ensure NaN on error
                    R_matrix_subject(j,i) = NaN; P_matrix_subject(j,i) = NaN;
                end
            end
        end

        if sum(~isnan(R_matrix_subject(:))) == 0 || valid_pairs_for_subject < num_vars 
             fprintf('Skipping Subject %s: No valid correlation pairs could be computed.\n', char(current_subject_id));
             continue;
        end
        
        R_display_matrix_subject = R_matrix_subject;
        is_significant_and_not_diagonal = (P_matrix_subject < 0.05 & repmat((1:num_vars)',1,num_vars) ~= repmat(1:num_vars,num_vars,1));
        is_diagonal = (eye(num_vars) == 1);
        R_display_matrix_subject(~(is_significant_and_not_diagonal | is_diagonal)) = NaN;

        % --- Plotting ---
        if mod(plot_count_on_current_fig, subplots_per_fig) == 0
            fig_num = floor(subjects_plotted_count/subplots_per_fig) + 1;
            current_fig_handle = figure('Name', ['Subject Correlations Page ', num2str(fig_num)], ...
                                     'Color','w', ...
                                     'Units','centimeters', ...
                                     'Visible', 'on'); 
            set(current_fig_handle, 'PaperType', 'A4', 'PaperOrientation', 'portrait', 'Units','normalized', 'OuterPosition',[0.03 0.03 0.94 0.94]);
            % set(current_fig_handle, 'PaperUnits', 'normalized'); 
            % set(current_fig_handle, 'OuterPosition', [0 0 1 1]); 
            % set(current_fig_handle, 'PaperUnits', 'centimeters'); 
            % set(current_fig_handle, 'PaperPositionMode', 'manual'); 
            % set(current_fig_handle, 'PaperSize', [21 29.7]); 
            % set(current_fig_handle, 'PaperPosition', [1.5 1.5 18 26.7]); % Adjusted margins for better layout [left bottom width height]
            
            if isempty(fig_handles) || ~any(fig_handles == current_fig_handle)
                 fig_handles(end+1) = current_fig_handle;
            end
            plot_count_on_current_fig = 0; 
        end
        
        plot_count_on_current_fig = plot_count_on_current_fig + 1;
        subjects_plotted_count = subjects_plotted_count + 1;
        
        subplot_idx = plot_count_on_current_fig;
        figure(current_fig_handle); 
        ax = subplot(2, 3, subplot_idx); 
        
        % Use imagesc
        img_h = imagesc(ax, R_display_matrix_subject);
        
        % Set background color of axes to white (will show through for transparent NaN cells)
        set(ax, 'Color', [1 1 1]);
        
        % Create AlphaData to make NaN cells transparent
        alpha_data = ones(size(R_display_matrix_subject));
        alpha_data(isnan(R_display_matrix_subject)) = 0; % Transparent for NaN
        set(img_h, 'AlphaData', alpha_data);
        
        colormap(ax, final_cmap_100);
        clim(ax, [-1 1]);
        
        % Set ticks and labels
        set(ax, 'XTick', 1:num_vars, ...
                'XTickLabel', processed_tick_labels, ...
                'XTickLabelRotation', 45, ... % Rotate for readability
                'YTick', 1:num_vars, ...
                'YTickLabel', processed_tick_labels, ...
                'FontSize', 10, ...
                'TickDir', 'out');
        
        axis(ax, 'image'); % Make cells square
        box(ax, 'on'); % Add a box around the heatmap
        
        title(ax, sprintf('Subject: %s', char(current_subject_id)), 'FontSize', 14);
        
        % Add a colorbar
        cb = colorbar(ax);
        ylabel(cb, 'R-value', 'FontSize', 12);
        
        % --- Manually Add Cell Labels (R-value and significance star) ---
        for r_idx = 1:num_vars
            for c_idx = 1:num_vars
                r_val_cell = R_matrix_subject(r_idx, c_idx); 
                p_val_cell = P_matrix_subject(r_idx, c_idx);
                
                if ~isnan(r_val_cell)
                    label_str = sprintf('%.2f', r_val_cell);
                    font_weight_val = 'normal';
                    
                    if r_idx == c_idx 
                        font_weight_val = 'bold';
                    elseif p_val_cell < 0.05 
                        label_str = [label_str, '*'];
                        font_weight_val = 'bold';
                    end
                    
                    % x_pos and y_pos for text are c_idx and r_idx for imagesc
                    text(ax, c_idx, r_idx, label_str, ...
                        'HorizontalAlignment', 'center', ...
                        'VerticalAlignment', 'middle', ...
                        'FontSize', 8, ... 
                        'FontWeight', font_weight_val, ...
                        'Color', 'k'); 
                end
            end
        end
        
        fprintf('Plotted heatmap for Subject %s using imagesc.\n', char(current_subject_id));
    end % End of subject loop

    if subjects_plotted_count == 0
        disp('No subjects had sufficient data to plot heatmaps.');
        for fh_idx = 1:numel(fig_handles)
            if isgraphics(fig_handles(fh_idx)) && isempty(get(fig_handles(fh_idx),'Children'))
                close(fig_handles(fh_idx));
            end
        end
        fig_handles = gobjects(0); 
    else
        disp(['Successfully generated heatmaps for ', num2str(subjects_plotted_count), ' subjects.']);
    end
    
    for fh_idx = 1:numel(fig_handles)
        if isgraphics(fig_handles(fh_idx))
            figure(fig_handles(fh_idx)); 
            try
                if numel(findobj(fig_handles(fh_idx), 'type', 'axes')) > 0 % Check for axes, not just children
                     page_num_sg = find(fig_handles == fig_handles(fh_idx),1,'first'); % Simpler page number
                     sgtitle(fig_handles(fh_idx), ['Subject Correlation Heatmaps - Page ', num2str(page_num_sg)], 'FontWeight', 'bold', 'FontSize',16);
                end
            catch ME_sgtitle
                disp(['Note: Could not set super title for figure page. ', ME_sgtitle.message]);
            end
        end
    end

end % End of main function



function [R_matrix_out, P_matrix_out, heatmap_labels, fig_handle] = plot_rmc_heatmap_matlab(output_table, required_variables)
% PLOT_RMC_HEATMAP_MATLAB_MODIFIED Generates a heatmap of Repeated Measures Correlations
% and outputs the R-matrix, P-matrix, and labels.
%
% Features:
% - Calculates RMC R-values and p-values for pairs of specified variables.
% - SubjectID is used as the grouping factor for RMC.
% - Displays R-values in a heatmap.
% - Only shows color for cells where p-value < 0.05. Other cells are not colored.
% - Bolds R-values and adds a '*' to cell labels where p-value < 0.05 (manual text objects - NB: actual text object creation not in this snippet).
% - Uses a custom 100-level interpolated colormap.
% - Makes the heatmap figure square.
% - Replaces underscores with spaces in axis tick labels.
%
% Inputs:
%   output_table (table): MATLAB table with 'SubjectID' and behavioral variables.
%   required_variables (cell array of strings): Numeric variables for RMC.
%
% Outputs:
%   R_matrix_out (matrix): Matrix of Repeated Measures Correlation R-values.
%   P_matrix_out (matrix): Matrix of p-values corresponding to the R-values.
%   heatmap_labels (cell array of strings): Labels used for the heatmap axes.
%   fig_handle (figure handle): Handle to the generated heatmap figure.

% --- 1. Input Validation ---
if ~istable(output_table); error('Input `output_table` must be a MATLAB table.'); end
if isempty(output_table); disp('Warning: output_table is empty.'); R_matrix_out=[]; P_matrix_out=[]; heatmap_labels={}; fig_handle=[]; return; end
if ~iscellstr(required_variables) && ~isstring(required_variables); error('Input `required_variables` must be a cell array of strings or a string array.'); end %#ok<ISCLSTR>
if numel(required_variables) < 2; disp('At least two variables are required for pairing to create a heatmap.'); R_matrix_out=[]; P_matrix_out=[]; heatmap_labels={}; fig_handle=[]; return; end
if ~ismember('SubjectID', output_table.Properties.VariableNames); error("'SubjectID' column not found in output_table."); end

% Identify numeric variables from required_variables
var_types = varfun(@class, output_table, 'InputVariables', required_variables, 'output','cell');
numeric_mask = cellfun(@(x) strcmp(x, 'double') || strcmp(x, 'single') || strcmp(x,'logical'), var_types); % Add other numeric types if needed
numeric_vars_for_heatmap = required_variables(numeric_mask);

if numel(numeric_vars_for_heatmap) < 2
    disp('Not enough numeric variables found in required_variables to create a heatmap of pairs.');
    R_matrix_out=[]; P_matrix_out=[]; heatmap_labels={}; fig_handle=[]; return;
end
disp(['Using numeric variables for RMC heatmap: ', strjoin(numeric_vars_for_heatmap, ', ')]);

% --- 2. Initial Data Cleaning (SessionDateTime "SessionIndex" removal & NaN Filtering) ---
dataTable = output_table;
if ismember('SessionDateTime', dataTable.Properties.VariableNames)
    sdt = dataTable.SessionDateTime;
    rows_to_remove_sdi = false(height(dataTable), 1);
    if iscellstr(sdt) || isstring(sdt)
        if iscellstr(sdt); sdt = string(sdt); end
        rows_to_remove_sdi = contains(sdt, "SessionIndex", "IgnoreCase", true);
    elseif iscategorical(sdt)
        rows_to_remove_sdi = contains(string(sdt), "SessionIndex", "IgnoreCase", true);
    end %#ok<ISCLSTR>
    dataTable = dataTable(~rows_to_remove_sdi, :);
    if isempty(dataTable); disp('No data after SessionIndex filter.'); R_matrix_out=[]; P_matrix_out=[]; heatmap_labels={}; fig_handle=[]; return; end
end

vars_to_check_nan = unique([numeric_vars_for_heatmap(:)', {'SubjectID'}]);
missing_any_required = false(height(dataTable), 1);
for i_var_check = 1:numel(vars_to_check_nan)
    current_var_data = dataTable.(vars_to_check_nan{i_var_check});
    if isnumeric(current_var_data) || islogical(current_var_data)
        missing_any_required = missing_any_required | isnan(current_var_data);
    elseif iscategorical(current_var_data) || isdatetime(current_var_data) || isduration(current_var_data) || isstring(current_var_data)
        missing_any_required = missing_any_required | ismissing(current_var_data);
    elseif iscell(current_var_data)
        missing_any_required = missing_any_required | cellfun(@(x) (ischar(x) && isempty(x)) || (isnumeric(x) && isnan(x)) || (isstring(x) && ismissing(x)), current_var_data);
    end
end
dataTable_cleaned = dataTable(~missing_any_required, :);

if height(dataTable_cleaned) == 0
    disp('No data after NaN/missing filter for relevant columns.');
    R_matrix_out=[]; P_matrix_out=[]; heatmap_labels={}; fig_handle=[]; return;
end

% Ensure SubjectID is categorical
if isnumeric(dataTable_cleaned.SubjectID) || islogical(dataTable_cleaned.SubjectID)
    dataTable_cleaned.SubjectID = categorical(dataTable_cleaned.SubjectID);
elseif iscellstr(dataTable_cleaned.SubjectID) || isstring(dataTable_cleaned.SubjectID)
    dataTable_cleaned.SubjectID = categorical(dataTable_cleaned.SubjectID);
elseif ~iscategorical(dataTable_cleaned.SubjectID)
    error('SubjectID must be convertible to categorical.');
end %#ok<ISCLSTR>

% --- 3. Calculate RMC for each pair ---
num_vars = numel(numeric_vars_for_heatmap);
R_matrix = NaN(num_vars, num_vars);
P_matrix = NaN(num_vars, num_vars);
disp('Calculating Repeated Measures Correlations...');
for i = 1:num_vars
    for j = i:num_vars % Calculate for diagonal and upper triangle
        var1_name = numeric_vars_for_heatmap{i};
        var2_name = numeric_vars_for_heatmap{j};
        if i == j
            R_matrix(i,j) = 1;
            P_matrix(i,j) = 0; % p-value for correlation with self is 0
            continue;
        end
        
        v1_data = dataTable_cleaned.(var1_name);
        v2_data = dataTable_cleaned.(var2_name);
        subjects_data = dataTable_cleaned.SubjectID;
        
        % Call local RMC calculation function
        [r_val, p_val, df, N_eff] = local_calculate_rmc(v1_data, v2_data, subjects_data);
        
        R_matrix(i,j) = r_val;
        R_matrix(j,i) = r_val; % Symmetric
        P_matrix(i,j) = p_val;
        P_matrix(j,i) = p_val; % Symmetric
        
        if ~isnan(r_val)
            fprintf('RMC for %s vs %s: R=%.3f, p=%.4f (N_eff=%d, df=%d)\n', var1_name, var2_name, r_val, p_val, N_eff, df);
        else
            fprintf('RMC for %s vs %s: Could not be computed (insufficient data after filtering for pair).\n', var1_name, var2_name);
        end
    end
end
disp('RMC calculations complete.');

% --- Prepare R_matrix for display based on p-value threshold ---
R_display_matrix = R_matrix; % Initialize with all R values
for r_idx = 1:num_vars
    for c_idx = 1:num_vars
        if ~isnan(R_matrix(r_idx, c_idx)) % Only process non-NaN R-values
            % Using P_matrix directly for the p-value check
            if P_matrix(r_idx, c_idx) >= 0.05
                R_display_matrix(r_idx, c_idx) = NaN; % Set to NaN to not show color if p >= 0.05
            end
        else
            R_display_matrix(r_idx, c_idx) = NaN; % Ensure already NaN values remain NaN for display
        end
    end
end
% Diagonal cells (R=1, p=0) will remain as R=1 and thus be colored.

% --- 4. Create Heatmap ---
% Prepare axis labels (replace underscores with spaces)
processed_tick_labels = strrep(numeric_vars_for_heatmap, '_', ' ');

% Create the figure, adjust OuterPosition for a more square window
fig_heatmap = figure('Name', 'Repeated Measures Correlation Heatmap', 'Color','w', ...
    'Units','normalized','OuterPosition',[0.1 0.1 0.7 0.75]);
% Use the R_display_matrix for the heatmap data
h = heatmap(fig_heatmap, processed_tick_labels, processed_tick_labels, R_display_matrix);
h.Title = 'Repeated Measures Correlation (R-values, colored if p < 0.05)';
h.XLabel = 'Variables';
h.YLabel = 'Variables';
h.ColorLimits = [-1 1]; % Keep original limits for consistent coloring of significant values

% Define the 6 anchor colors (RGB [0-1])
anchor_colors_rgb = [
    97/255, 170/255, 207/255;  % C1: 61AACF (for -1)
    152/255, 202/255, 221/255; % C2: 98CADD (for -0.5)
    234/255, 239/255, 246/255; % C3: EAEFF6 (for near 0, negative side)
    249/255, 239/255, 239/255; % C4: F9EFEF (for near 0, positive side)
    233/255, 198/255, 198/255; % C5: E9C6C6 (for +0.5)
    218/255, 149/255, 153/255   % C6: DA9599 (for +1)
];
% Define normalized positions for these anchor colors (0 to 1, mapping to -1 to 1 data)
map_indices = [0, 0.25, 0.499, 0.501, 0.75, 1]; % Positions for C1 to C6
% Create 100 query points for the new colormap
num_cmap_levels = 100;
query_indices = linspace(0, 1, num_cmap_levels);
% Interpolate each RGB channel
interp_cmap_r = interp1(map_indices, anchor_colors_rgb(:,1), query_indices, 'linear');
interp_cmap_g = interp1(map_indices, anchor_colors_rgb(:,2), query_indices, 'linear');
interp_cmap_b = interp1(map_indices, anchor_colors_rgb(:,3), query_indices, 'linear');
% Combine into the final colormap
final_cmap_100 = [interp_cmap_r', interp_cmap_g', interp_cmap_b'];
final_cmap_100(final_cmap_100 < 0) = 0; % Clip any potential small negative due to interpolation
final_cmap_100(final_cmap_100 > 1) = 1; % Clip any potential small values > 1
h.Colormap = final_cmap_100;
h.MissingDataColor = [1 1 1]; % White

disp('Heatmap generated. Cells with p >= 0.05 are not colored.');
disp('Manual label customization (if any) should still apply.');

% --- Assign outputs ---
R_matrix_out = R_matrix;         % The raw R-values from RMC
P_matrix_out = P_matrix;         % The corresponding p-values
heatmap_labels = processed_tick_labels; % Labels used for heatmap axes
fig_handle = fig_heatmap;        % Handle to the generated heatmap figure




% --- Helper function for RMC calculation (nested) ---
function [r, p, df_rmc, N_total_paired_obs] = local_calculate_rmc(v1, v2, subjects_cat)
    % Remove rows with NaNs in v1 or v2 for this specific pair
    nan_mask_pair = isnan(v1) | isnan(v2);
    v1_clean_pair = v1(~nan_mask_pair);
    v2_clean_pair = v2(~nan_mask_pair);
    subjects_clean_pair = subjects_cat(~nan_mask_pair);
    
    unique_subj_ids_pair = unique(subjects_clean_pair);
    k_subjects_in_pair = numel(unique_subj_ids_pair);
    
    r = NaN; p = NaN; df_rmc = NaN; N_total_paired_obs = 0;
    if k_subjects_in_pair < 2 % Need at least 2 subjects for RMC logic
        return;
    end
    
    demeaned_v1 = [];
    demeaned_v2 = [];
    num_subjects_contributing = 0;
    
    for i_s = 1:k_subjects_in_pair
        subj_id_current = unique_subj_ids_pair(i_s);
        idx_s_current = (subjects_clean_pair == subj_id_current);
        
        v1_s_current = v1_clean_pair(idx_s_current);
        v2_s_current = v2_clean_pair(idx_s_current);
        
        if numel(v1_s_current) >= 2 % Each subject needs at least 2 data points for this pair
            demeaned_v1 = [demeaned_v1; v1_s_current - mean(v1_s_current)]; %#ok<AGROW>
            demeaned_v2 = [demeaned_v2; v2_s_current - mean(v2_s_current)]; %#ok<AGROW>
            N_total_paired_obs = N_total_paired_obs + numel(v1_s_current);
            num_subjects_contributing = num_subjects_contributing + 1;
        end
    end
    
    if isempty(demeaned_v1) || num_subjects_contributing < 1 || N_total_paired_obs < 3
        return;
    end
    if std(demeaned_v1, 'omitnan') < eps || std(demeaned_v2, 'omitnan') < eps
        return;
    end
    
    [corr_matrix_demeaned, ~] = corrcoef(demeaned_v1, demeaned_v2, 'Rows','complete');
    
    if numel(corr_matrix_demeaned) < 4 || ~ismatrix(corr_matrix_demeaned) || size(corr_matrix_demeaned,1)<2 || size(corr_matrix_demeaned,2)<2
        return;
    end
    
    r = corr_matrix_demeaned(1,2);
    
    if isnan(r)
        return;
    end
    
    df_rmc = N_total_paired_obs - num_subjects_contributing - 1;
    
    if df_rmc <= 0
        p = NaN;
        return;
    end
    
    if abs(r) >= 1.0 - 10*eps
        t_stat = sign(r) * Inf;
    else
        t_stat = r * sqrt(df_rmc / (1 - r^2));
    end
    
    p = 2 * tcdf(-abs(t_stat), df_rmc); % Two-tailed p-value
end

end % End of main function plot_rmc_heatmap_matlab_modified


function plot_behavioral_scatter_grid_matlab(output_table, required_variables)
% PLOT_BEHAVIORAL_SCATTER_GRID_MATLAB Generates enhanced scatter plots.
%
% Features:
% - Normalizes data in required_variables to 0-1 range before plotting/fitting.
% - Subplot titles are made BOLD RED if LME R^2 > 0.5 AND p-value < 0.05.
% - Handles multiple figures if plots exceed 3x6 (18) subplots per figure.
% - Each figure is saved as PNG and closed.
% - Outlier removal (per subject, per X-Y pair, 3 STD from subject's mean for X or Y).
% - Square subplots.
% - Per-subject linear regression lines.
% - Overall linear mixed model (LMM) fit (Y ~ X + (1|SubjectID)) with a thicker gray line.
% - Annotation of LMM fixed effect slope, p-value, and Marginal R-squared.
% - Requires Statistics and Machine Learning Toolbox for LMM.
%
% Inputs:
% output_table (table): MATLAB table with 'SubjectID' and behavioral variables.
% required_variables (cell array of strings): Variables to plot and normalize.

% --- 1. Input Validation ---
if ~istable(output_table); error('Input `output_table` must be a MATLAB table.'); end
if isempty(output_table); disp('Warning: output_table is empty.'); return; end
if ~iscellstr(required_variables) && ~isstring(required_variables); error('Input `required_variables` must be a cell array of strings or a string array.'); end %#ok<ISCLSTR>
if numel(required_variables) < 2; disp('At least two variables are required for pairing.'); return; end
if ~ismember('SubjectID', output_table.Properties.VariableNames); error("'SubjectID' column not found in output_table."); end
missing_vars = required_variables(~ismember(required_variables, output_table.Properties.VariableNames));
if ~isempty(missing_vars); error('Variables not found: %s', strjoin(missing_vars, ', ')); end
dataTable = output_table;

% --- 2. Initial Data Cleaning (SessionDateTime "SessionIndex" removal) ---
if ismember('SessionDateTime', dataTable.Properties.VariableNames)
    sdt = dataTable.SessionDateTime; rows_to_remove_sdi = false(height(dataTable), 1);
    if iscellstr(sdt) || isstring(sdt); if iscellstr(sdt); sdt = string(sdt); end; rows_to_remove_sdi = contains(sdt, "SessionIndex", "IgnoreCase", true); elseif iscategorical(sdt); rows_to_remove_sdi = contains(string(sdt), "SessionIndex", "IgnoreCase", true); end %#ok<ISCLSTR>
    dataTable = dataTable(~rows_to_remove_sdi, :);
    if isempty(dataTable); disp('No data after SessionIndex filter.'); return; end
end

% --- 3. NaN Filtering for Required Variables & SubjectID ---
vars_to_check_nan = unique([required_variables(:)', {'SubjectID'}]);
missing_any_required = false(height(dataTable), 1);
for i = 1:numel(vars_to_check_nan)
    current_var_data = dataTable.(vars_to_check_nan{i});
    if isnumeric(current_var_data) || islogical(current_var_data); missing_any_required = missing_any_required | isnan(current_var_data);
    elseif iscategorical(current_var_data) || isdatetime(current_var_data) || isduration(current_var_data) || isstring(current_var_data); missing_any_required = missing_any_required | ismissing(current_var_data);
    elseif iscell(current_var_data); missing_any_required = missing_any_required | cellfun(@(x) (ischar(x) && isempty(x)) || (isnumeric(x) && isnan(x)) || (isstring(x) && ismissing(x)), current_var_data);end
end
dataTable_filtered = dataTable(~missing_any_required, :);
if height(dataTable_filtered) == 0; disp('No data after NaN/missing filter.'); return; end

% --- 3.5. Normalize Numeric Data in required_variables to 0-1 Range ---
temp_dataTable_for_norm = dataTable_filtered; 
for i_norm = 1:numel(required_variables)
    var_name_norm = required_variables{i_norm};
    if ismember(var_name_norm, temp_dataTable_for_norm.Properties.VariableNames) && isnumeric(temp_dataTable_for_norm.(var_name_norm))
        col_data = temp_dataTable_for_norm.(var_name_norm);
        min_val = min(col_data, [], 'omitnan');
        max_val = max(col_data, [], 'omitnan');
        if isnan(min_val) || isnan(max_val) 
            normalized_col = col_data; 
        elseif min_val == max_val 
            normalized_col = zeros(size(col_data), 'like', col_data) + 0.5;
            normalized_col(isnan(col_data)) = NaN; 
        else
            normalized_col = (col_data - min_val) / (max_val - min_val);
        end
        temp_dataTable_for_norm.(var_name_norm) = normalized_col;
    end
end
dataTable_filtered = temp_dataTable_for_norm; 
clear temp_dataTable_for_norm;
disp('Numeric data in required_variables normalized to 0-1 range.');

% --- 4. Assign Colors to Subjects using Hardcoded Lists ---
subject_id_list_hardcoded = {'Bard'; 'Frey'; 'Igor'; 'Reider'; 'Sindri'; 'Wotan'};
custom_colors_rgb = {[245,124,110]/255; [242,181,110]/255; [251,231,158]/255; [132,195,183]/255; [136,215,218]/255; [113,184,237]/255; [184,174,234]/255; [242,168,218]/255};
if isnumeric(dataTable_filtered.SubjectID) || islogical(dataTable_filtered.SubjectID); dataTable_filtered.SubjectID = categorical(dataTable_filtered.SubjectID);
elseif iscellstr(dataTable_filtered.SubjectID) || isstring(dataTable_filtered.SubjectID); dataTable_filtered.SubjectID = categorical(dataTable_filtered.SubjectID); elseif ~iscategorical(dataTable_filtered.SubjectID); error('SubjectID must be convertible to categorical.'); end %#ok<ISCLSTR>
data_subject_categories = categories(dataTable_filtered.SubjectID);
if isempty(data_subject_categories); disp('No unique subjects in data.'); return; end
num_data_subjects = numel(data_subject_categories);
subject_color_map = containers.Map('KeyType','char','ValueType','any');
min_fallback_palette_size = max(num_data_subjects,7); fallback_colors_palette = lines(min_fallback_palette_size);
if num_data_subjects > size(fallback_colors_palette,1); fallback_colors_palette = lines(num_data_subjects); end; fallback_color_idx = 1;
for i=1:num_data_subjects; current_subject_name=char(data_subject_categories{i}); [is_hardcoded,loc_in_hardcoded_list]=ismember(current_subject_name,subject_id_list_hardcoded);
    if is_hardcoded && loc_in_hardcoded_list <= numel(custom_colors_rgb); subject_color_map(current_subject_name) = custom_colors_rgb{loc_in_hardcoded_list};
    else; if is_hardcoded; warning('Subject "%s" hardcoded but custom colors short. Using fallback.',current_subject_name); else; warning('Subject "%s" not hardcoded. Using fallback.',current_subject_name); end; subject_color_map(current_subject_name)=fallback_colors_palette(fallback_color_idx,:); fallback_color_idx=mod(fallback_color_idx,size(fallback_colors_palette,1))+1; end; end

% --- 5. Categorize Variables & Generate Pairs ---
task_variable_map=containers.Map('KeyType','char','ValueType','any');
for i=1:numel(required_variables); var_name=char(required_variables{i}); task=get_task_from_variable_prefix(var_name); if ~strcmp(task,'Unknown'); if isKey(task_variable_map,task); task_variable_map(task)=[task_variable_map(task),{var_name}]; else; task_variable_map(task)={var_name}; end; else; warning('Var "%s" (from required_variables) resulted in an Unknown task and was ignored.',var_name); end; end
active_tasks_keys=keys(task_variable_map); if numel(active_tasks_keys)<2; disp('Not enough distinct tasks to form pairs from prefixes.'); return; end
plot_pairs={}; if numel(active_tasks_keys)>=2; task_name_combinations=nchoosek(active_tasks_keys,2); for i=1:size(task_name_combinations,1); vars1_list=task_variable_map(task_name_combinations{i,1}); vars2_list=task_variable_map(task_name_combinations{i,2}); for v1=vars1_list; for v2=vars2_list; plot_pairs{end+1}={v1{1},v2{1}}; end; end; end; end %#ok<AGROW>
if isempty(plot_pairs); disp('No plot pairs generated based on task prefixes.'); return; end

% --- 6. Plotting Setup and Figure Generation Loop ---
num_plots = numel(plot_pairs);
if num_plots == 0; disp('No plots to generate.'); return; end

subplots_per_figure_rows = 3; 
subplots_per_figure_cols = 6; 
subplots_per_figure = subplots_per_figure_rows * subplots_per_figure_cols;
num_figures_total = ceil(num_plots / subplots_per_figure);
use_subplot_legacy = verLessThan('matlab','9.7'); 

for fig_idx = 1:num_figures_total
    fig = figure('Name', sprintf('Enhanced Cross-Task Scatter Plots (Normalized) - Page %d of %d', fig_idx, num_figures_total), ...
                 'Color','w', 'PaperType','A4', 'PaperOrientation','portrait', ...
                 'Units','normalized', 'OuterPosition',[0.03 0.03 0.94 0.94]);
    plots_on_this_page_start_idx = (fig_idx - 1) * subplots_per_figure + 1;
    plots_on_this_page_end_idx = min(fig_idx * subplots_per_figure, num_plots);
    num_subplots_this_figure = plots_on_this_page_end_idx - plots_on_this_page_start_idx + 1;
    
    axes_handles_on_page = gobjects(num_subplots_this_figure, 1);
    first_ax_handle_this_page = []; 
    
	figure_title_str = sprintf('Cross-Task Analysis (Normalized Data) Page %d', fig_idx); 
    if ~use_subplot_legacy 
        plot_layout = tiledlayout(fig, subplots_per_figure_rows, subplots_per_figure_cols, 'TileSpacing','compact', 'Padding','compact');
        title(plot_layout, figure_title_str, 'FontSize',12,'FontWeight','bold', 'Interpreter', 'latex');
    end

    for subplot_k_on_page = 1:num_subplots_this_figure
        k = plots_on_this_page_start_idx + subplot_k_on_page - 1; 
        var_x_name = plot_pairs{k}{1};
        var_y_name = plot_pairs{k}{2};

        highlight_subplot_flag = false; % Initialize for current subplot highlighting
        r_squared_for_highlight = NaN;
        p_value_for_highlight = NaN;
        model_R2_text_for_annotation = 'R^2=N/A'; % For text annotation
        lme_annotation_text = 'LME: N/A';      % For text annotation

        % --- 7a. Outlier Removal for current X-Y pair (operates on normalized data) ---
        is_outlier_current_pair = false(height(dataTable_filtered), 1);
        temp_subject_cats_for_outlier = categories(dataTable_filtered.SubjectID); 
        num_temp_subj_for_outlier = numel(temp_subject_cats_for_outlier);
        for s_idx_outlier = 1:num_temp_subj_for_outlier
            subj_char_outlier = char(temp_subject_cats_for_outlier{s_idx_outlier});
            subj_val_comp_outlier = categorical({subj_char_outlier}); 
            subj_mask_orig = (dataTable_filtered.SubjectID == subj_val_comp_outlier);
            
            if ~any(subj_mask_orig); continue; end 
            
            x_subj_vals = dataTable_filtered.(var_x_name)(subj_mask_orig); 
            y_subj_vals = dataTable_filtered.(var_y_name)(subj_mask_orig); 
            
            x_subj_clean_stats = x_subj_vals(~isnan(x_subj_vals));
            outliers_x_for_subj = false(size(x_subj_vals));
            if numel(x_subj_clean_stats) > 1
                mean_x = mean(x_subj_clean_stats); std_x = std(x_subj_clean_stats);
                if std_x > 1e-9 
                    outliers_x_for_subj = abs(x_subj_vals - mean_x) > 3 * std_x;
                end
            end
            
            y_subj_clean_stats = y_subj_vals(~isnan(y_subj_vals));
            outliers_y_for_subj = false(size(y_subj_vals));
            if numel(y_subj_clean_stats) > 1
                mean_y = mean(y_subj_clean_stats); std_y = std(y_subj_clean_stats);
                if std_y > 1e-9
                    outliers_y_for_subj = abs(y_subj_vals - mean_y) > 3 * std_y;
                end
            end
            
            indices_this_subj_in_dt_filtered = find(subj_mask_orig);
            is_outlier_current_pair(indices_this_subj_in_dt_filtered(outliers_x_for_subj | outliers_y_for_subj)) = true;
        end
        dataTable_cleaned_for_plot = dataTable_filtered(~is_outlier_current_pair, :);
        
        current_subplot_page_idx = subplot_k_on_page; 
        if use_subplot_legacy
            axes_handles_on_page(current_subplot_page_idx) = subplot(subplots_per_figure_rows, subplots_per_figure_cols, current_subplot_page_idx);
        else
            axes_handles_on_page(current_subplot_page_idx) = nexttile(plot_layout);
        end
        ax = axes_handles_on_page(current_subplot_page_idx);
        if isempty(first_ax_handle_this_page) && isgraphics(ax) 
            first_ax_handle_this_page = ax;
        end
        
        if height(dataTable_cleaned_for_plot) < 3 
            title(ax, sprintf('%s (Norm.) vs %s (Norm.)\n(Insufficient data after outlier removal)', strrep(var_x_name,'_',' '), strrep(var_y_name,'_',' ')), 'FontSize',12);
            set(ax, 'XTick', [], 'YTick', []); pbaspect(ax, [1 1 1]);
            % For insufficient data plots, we still add the default annotation text
            full_annotation_text_insufficient = [lme_annotation_text, '\newline', model_R2_text_for_annotation];
            current_xlims_insufficient = xlim(ax); current_ylims_insufficient = ylim(ax); % Get current, even if blank
             % Try to place annotation; might be [0 1] if axes blank
            text_x_pos_insufficient = current_xlims_insufficient(1) + 0.5 * range(current_xlims_insufficient); 
            text_y_pos_insufficient = current_ylims_insufficient(1) + 0.5 * range(current_ylims_insufficient);
            if range(current_xlims_insufficient) == 0; text_x_pos_insufficient = 0.5; end % Default if no range
            if range(current_ylims_insufficient) == 0; text_y_pos_insufficient = 0.5; end
            text(ax, text_x_pos_insufficient, text_y_pos_insufficient, full_annotation_text_insufficient, 'HorizontalAlignment','center', ...
             'VerticalAlignment','middle', 'FontSize',8, 'BackgroundColor',[1 1 0.9], 'EdgeColor','k','Margin',1, 'Interpreter', 'tex');
            continue; 
        end
        
        hold(ax, 'on');
        pbaspect(ax, [1 1 1]);

        % --- Overall LMM Fit (on normalized data) ---
        if license('test', 'Statistics_Toolbox') && exist('fitlme','file')
            try
                all_x = dataTable_cleaned_for_plot.(var_x_name); 
                all_y = dataTable_cleaned_for_plot.(var_y_name); 
                all_subj_id = dataTable_cleaned_for_plot.SubjectID;
                
                nan_mask_lme = isnan(all_x) | isnan(all_y) | ismissing(all_subj_id);
                clean_x = all_x(~nan_mask_lme); clean_y = all_y(~nan_mask_lme); clean_subj = all_subj_id(~nan_mask_lme);
                if numel(clean_x) > (numel(unique(clean_subj)) + 2)
                    lme_tbl = table(clean_x, clean_y, clean_subj, 'VariableNames',{'X','Y','Subject'});
                    lme = fitlme(lme_tbl, 'Y ~ X + (1|Subject)');
                    x_coeff_info = lme.Coefficients(strcmp(lme.CoefficientNames,'X'), :);
                    
                    if ~isempty(x_coeff_info) && size(fixedEffects(lme),1) >=2
                        fe = fixedEffects(lme);
                        lme_annotation_text = sprintf('LME Slope=%.2f, p=%.3f', x_coeff_info.Estimate, x_coeff_info.pValue);
                        p_value_for_highlight = x_coeff_info.pValue; % CAPTURE P-VALUE for highlighting

                        x_overall_fit = linspace(min(clean_x), max(clean_x), 100)';
                        y_overall_fit = fe(1) + fe(2) * x_overall_fit;
                        plot(ax, x_overall_fit, y_overall_fit, 'Color', [0.4 0.4 0.4], 'LineWidth', 3, 'HandleVisibility','off');
                        
                        if isprop(lme, 'Rsquared') && isstruct(lme.Rsquared) && isfield(lme.Rsquared, 'Ordinary')
                            r_squared_for_highlight = lme.Rsquared.Ordinary; % CAPTURE R-SQUARED for highlighting
                            model_R2_text_for_annotation = sprintf('R^2=%.2f', r_squared_for_highlight);
                        else
                            model_R2_text_for_annotation = 'R^2=N/A'; % Ensure reset if not found
                            warning('lme.Rsquared.Ordinary not found for %s vs %s. Model R-squared will be N/A.',var_x_name, var_y_name);
                        end
                    else
                        lme_annotation_text = 'LME Slope=N/A (coeff issue)';
                    end
                else
                    lme_annotation_text = 'LME: Insufficient data';
                end
            catch ME_lme
                warning('LME failed for %s vs %s (post-outlier removal, normalized data): %s', var_x_name, var_y_name, ME_lme.message);
                lme_annotation_text = 'LME: Error';
            end
        else
            lme_annotation_text = 'LME: Toolbox missing';
            if k==1; warning('Statistics & Machine Learning Toolbox needed for LME. This warning appears once.'); end
        end

        % Check highlighting condition
        if ~isnan(r_squared_for_highlight) && ~isnan(p_value_for_highlight)
            % if r_squared_for_highlight > 0.2 && p_value_for_highlight < 0.05
			if p_value_for_highlight < 0.05
                highlight_subplot_flag = true;
            end
        end

        % --- Per-Subject Scatter and Linear Fit (on normalized data) ---
        temp_data_subject_categories = categories(dataTable_cleaned_for_plot.SubjectID);
        num_temp_data_subjects = numel(temp_data_subject_categories);
        for s_idx = 1:num_temp_data_subjects
            subj_char = char(temp_data_subject_categories{s_idx});
            subj_val_comp = categorical({subj_char});
            subj_mask = (dataTable_cleaned_for_plot.SubjectID == subj_val_comp);
            subj_df = dataTable_cleaned_for_plot(subj_mask, :);
            if ~isempty(subj_df)
                color_val = subject_color_map(subj_char);
                x_s = subj_df.(var_x_name); y_s = subj_df.(var_y_name); 
                
                if ismember(subj_char, data_subject_categories) 
                    scatter(ax,x_s,y_s,30,color_val,'filled','MarkerFaceAlpha',0.7,'DisplayName',subj_char);
                else
                    scatter(ax,x_s,y_s,30,color_val,'filled','MarkerFaceAlpha',0.7,'HandleVisibility','off'); 
                end
                if numel(x_s) >= 2 && numel(unique(x_s)) >= 2
                    try
                        p_s = polyfit(x_s, y_s, 1);
                        x_fit_s_range = [min(x_s); max(x_s)];
                        if x_fit_s_range(1) == x_fit_s_range(2); x_fit_s_range(2) = x_fit_s_range(1)+eps*(abs(x_fit_s_range(1))+1); end
                        y_fit_s = polyval(p_s, x_fit_s_range);
                        plot(ax, x_fit_s_range, y_fit_s, 'Color', color_val, 'LineWidth', 1.5, 'HandleVisibility','off');
                    catch ME_polyfit
                        warning('Polyfit failed for subject %s in %s vs %s (normalized data): %s', subj_char, var_x_name, var_y_name, ME_polyfit.message);
                    end
                end
            end
        end
        hold(ax, 'off');
        
        xlabel(ax, [strrep(var_x_name,'_',' ') ' (Norm.)'], 'FontSize',11); 
        ylabel(ax, [strrep(var_y_name,'_',' ') ' (Norm.)'], 'FontSize',11);
        
        % Apply title with potential highlighting
        plot_title_string = sprintf('%s (Norm.)\nvs\n%s (Norm.)', strrep(var_x_name,'_',' '), strrep(var_y_name,'_',' '));
        if highlight_subplot_flag
            title(ax, plot_title_string, 'FontSize',10,'FontWeight','bold', 'Color','red');
        else
            title(ax, plot_title_string, 'FontSize',10,'FontWeight','normal', 'Color', 'black');
        end
        
        grid(ax, 'on'); ax.FontSize = 9; axis(ax, 'tight');
        
        current_xlims = xlim(ax); current_ylims = ylim(ax);
        text_x_pos = current_xlims(2) - 0.03 * range(current_xlims);
        text_y_pos = current_ylims(1) + 0.03 * range(current_ylims);
        full_annotation_text = [lme_annotation_text, '\newline', model_R2_text_for_annotation];
        text(ax, text_x_pos, text_y_pos, full_annotation_text, 'HorizontalAlignment','right', ...
             'VerticalAlignment','bottom', 'FontSize',8, 'BackgroundColor',[1 1 0.9], 'EdgeColor','k','Margin',1, 'Interpreter', 'tex');
    end 

    % --- Add Legend to the current figure ---
    if ~isempty(first_ax_handle_this_page) && isvalid(first_ax_handle_this_page) && num_data_subjects > 0
        lgd = legend(first_ax_handle_this_page); 
        if ~isempty(lgd)
            title(lgd, 'SubjectID', 'FontSize',10);
            lgd.FontSize = 9;
            if use_subplot_legacy 
                try
                    drawnow; 
                    lgd.Units = 'normalized';
                    new_pos_x = 0.85; 
                    lgd_width = lgd.Position(3) * 0.8; 
                    lgd_height = lgd.Position(4);
                    if new_pos_x + lgd_width > 0.98; new_pos_x = 0.98 - lgd_width; end
                    new_pos_x = max(0.01, new_pos_x); 
                    new_pos_y = 0.5 - lgd_height / 2; 
                    new_pos_y = max(0.01, min(new_pos_y, 1 - lgd_height - 0.01));                     
                    lgd.Position = [new_pos_x, new_pos_y, lgd_width, lgd_height];
                catch ME_lgd
                    disp(['Warning: Could not reposition legend on figure ' num2str(fig_idx) '. ' ME_lgd.message]);
                end
            else 
                lgd.Layout.Tile = 'East'; 
            end
        end
    end

    if use_subplot_legacy && exist('sgtitle','file') 
        sgtitle(fig, figure_title_str,'FontSize',14,'FontWeight','bold', 'Interpreter', 'latex'); 
    end

    % --- Save and Close Figure ---
    filename_png = sprintf('Behavioral_Scatter_Grid_Normalized_Page_%d.png', fig_idx); 
    try
        saveas(fig, [pwd filesep filename_png]); 
        fprintf('Figure saved as %s\n', [pwd filesep filename_png]);
    catch ME_save
        warning('Could not save figure %s: %s', filename_png, ME_save.message);
    end
    close(fig); 
end 
disp('All figures generated and saved.');
end 

function task = get_task_from_variable_prefix(variable_name)
    if startsWith(variable_name, 'FL_', 'IgnoreCase', true); task = 'FL';
    elseif startsWith(variable_name, 'AS_', 'IgnoreCase', true); task = 'AS';
    elseif startsWith(variable_name, 'CR_', 'IgnoreCase', true); task = 'CR';
	elseif startsWith(variable_name, 'WM_', 'IgnoreCase', true); task = 'WM';
    else; task = 'Unknown'; end
end

