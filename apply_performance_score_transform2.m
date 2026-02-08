function modified_output_table = apply_performance_score_transform2(output_table)
% APPLY_PERFORMANCE_SCORE_TRANSFORM applies specific transformations to variables
% to align them as performance scores (where higher is better).
%
% This function performs two types of transformations:
% 1. Simple Sign Reversal (score = value * -1): For variables where a lower
%    original value indicates better performance (e.g., reaction time).
% 2. Absolute Value & Reversal (score = abs(value) * -1): For variables
%    where a value closer to zero indicates better performance (e.g., accuracy
%    differences). This is applied to variables identified by specific prefixes.
%
% The function identifies variables by exact name or by prefix.
%
% Inputs:
%   output_table (table): The input MATLAB table from the summary generator.
%
% Outputs:
%   modified_output_table (table): A new table with the specified numeric
%                                  variables transformed.

% --- Input Validation ---
if ~istable(output_table)
    error('Input "output_table" must be a MATLAB table.');
end
modified_output_table = output_table;
all_vars_in_table = modified_output_table.Properties.VariableNames;

% --- Configuration ---
% For these, a lower original number is better, so we flip the sign.
simple_reverse_list = {
    'FL_RT_Overall', ...
    'AS_RT_Overall', 'CR_RT_Intercept_Stim2to20', ...
    'WM_RT_Above05_Overall', 'WM_RT_Slope', 'WM_RT_Intercept', ...
	'FL_CEn_Proportion','FL_Trials_To_Criterion','CR_nBackSlope',
    };

% For variables with these prefixes, a value closer to zero is better.
% We take the absolute value first, then flip the sign.
simple_reverse_prefixes = {
    'WM_AccDiff_','WM_RTDiff_','FL_CEn_Proportion_', 'FL_Trials_To_Criterion_','FL_Plateau_Accuracy_', 'AS_AccuracyDiff_ProAnti', 'AS_AccuracyDiff_CongIncong','AS_RTDiff_ProAnti', 'AS_RTDiff_CongIncong',
    };

% --- Transformation Block 1: Simple Reversal (value * -1) ---
disp('--- Applying simple reversal (score = value * -1) ---');
for i = 1:numel(simple_reverse_list)
    var_name = simple_reverse_list{i};
    if ismember(var_name, all_vars_in_table)
        if isnumeric(modified_output_table.(var_name))
            modified_output_table.(var_name) = modified_output_table.(var_name) * -1;
            fprintf('Reversed: %s\n', var_name);
        else
            warning('Variable "%s" is not numeric and was not transformed.', var_name);
        end
    else
        warning('Variable "%s" from simple_reverse_list not found in the table.', var_name);
    end
end

% --- Transformation Block 2: Absolute Value and Reversal (abs(value) * -1) ---
disp('--- Applying absolute value reversal (score = abs(value) * -1) for prefixes ---');
% Find all variables that match the prefixes
prefix_matched_vars = {};
for p = 1:numel(simple_reverse_prefixes)
    prefix = simple_reverse_prefixes{p};
    matches = all_vars_in_table(startsWith(all_vars_in_table, prefix));
    prefix_matched_vars = [prefix_matched_vars, matches];
end
prefix_matched_vars = unique(prefix_matched_vars, 'stable');

% Apply the transformation to the matched variables
for i = 1:numel(prefix_matched_vars)
    var_name = prefix_matched_vars{i};
    % The variable is guaranteed to be in the table because we just found it.
    if isnumeric(modified_output_table.(var_name))
        modified_output_table.(var_name) = abs(modified_output_table.(var_name)) * -1;
        fprintf('Abs-Reversed: %s\n', var_name);
    else
        warning('Variable "%s" (from prefix match) is not numeric and was not transformed.', var_name);
    end
end

disp('--- Variable transformation process complete. ---');
end
