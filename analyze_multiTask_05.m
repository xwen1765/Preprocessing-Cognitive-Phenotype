
% metadata_table = readtable("multitask_variables.csv", 'VariableNamingRule','preserve');
% output_table = generate_behavioral_summary_table(metrics_mt);
% renamed_output_table = rename_variables(output_table);
% modified_output_table = apply_performance_score_transform(output_table);
% renamed_modified_output_table = rename_variables(modified_output_table);





my_filters = struct('CognitiveControlFactor', "IC");
exclude_list = {};
% exclude_list = {'FL_LearningSpeed_Dimension_Index', 'FL_LearningSpeed_Novalty_Index', ...
% 	'FL_CEn_Perservation_Dimension_Index','FL_CEn_Perservation_Novalty_Index'};
% exclude_list = {'FL_LearningSpeed', 'CR_Accuracy_TrialAt75Acc', 'CR_Accuracy_AvgTrialToError'}

% plotPCALoadings(renamed_output_table, metadata_table, ...
%                 'Filters', my_filters, ...
%                 'Exclude', exclude_list);



% my_filters = struct('CognitiveControlFactor', "IC");
% exclude_list = {'FL_LearningSpeed', 'CR_Accuracy_TrialAt75Acc', 'CR_Accuracy_AvgTrialToError', ...
% 	'FL_LearningSpeed_Dimension_Index', 'FL_LearningSpeed_Novalty_Index', ...
% 	'FL_CEn_Perservation_Dimension_Index','FL_CEn_Perservation_Novalty_Index' ,...
% 	'CR_nBackSlope', 'CR_nBackSlopeOfSlope'};

my_filters = struct('CognitiveControlFactor', "IC", 'MetricesCategory', "Accuracy");
exclude_list = {};


% exclude_list = {'FL_LearningSpeed', 'CR_Accuracy_TrialAt75Acc', 'CR_Accuracy_AvgTrialToError', ...
% 	'FL_LearningSpeed_Dimension_Index', 'FL_LearningSpeed_Novalty_Index', ...
% 	'FL_CEn_Perservation_Dimension_Index','FL_CEn_Perservation_Novalty_Index' ,...
% 	'CR_nBackSlope', 'CR_nBackSlopeOfSlope'};



my_filters = struct('MetricesCategory', "Accuracy");
exclude_list = {'CR_Accuracy_TrialAt75Acc', 'CR_Accuracy_AvgTrialToError'};
% plotPCALoadings(renamed_output_table, metadata_table, ...
%                 'Filters', my_filters, ...
%                 'Exclude', exclude_list);



my_filters = struct('MetricesCategory', "ProcessingSpeed");
exclude_list = {'CR_ProcessingSpeed_Intercept'};
% plotPCALoadings(renamed_output_table, metadata_table, ...
%                 'Filters', my_filters, ...
%                 'Exclude', exclude_list);

% plotPCALoadings(renamed_modified_output_table, metadata_table, ...
%                 'Filters', my_filters, ...
%                 'Exclude', exclude_list);


my_filters = struct();
exclude_list = {};

% [R_matrix_observed, P_matrix_parametric, fig_handle] = plot_RMC_heatmap(renamed_modified_output_table, metadataTable, ...
% 	'Filters', my_filters, ...
%     'Exclude', exclude_list);



my_filters = struct();
exclude_list = {};
% renamed_output_table = plotSubjectPCA(renamed_output_table, metadataTable, 'Filters', my_filters, ...
%     'Exclude', exclude_list);
% renamed_output_table = plotSubjectPCA_ICA(renamed_output_table, metadataTable, 'Filters', my_filters, ...
%     'Exclude', exclude_list);

% plotClusterDifferences(renamed_output_table, metadataTable);


% plotSubjectTSNE(renamed_output_table, metadata_table);
% plotTSNE_DBSCAN(renamed_output_table, metadata_table);
% plotRawDataDBSCAN(renamed_output_table, metadata_table);

% plotUMAP_DBSCAN(renamed_output_table, metadata_table);
 % plotUMAP_KMeans(renamed_output_table, metadata_table);
renamed_modified_output_table_withcluster = plotUMAP_KMeans_Silhouette(renamed_modified_output_table, metadata_table);


% my_filters = struct();
% analyzeMetricClusters_UMAP(renamed_modified_output_table, metadata_table, 'Filters', my_filters);

% [modularityScore, communityAssignments] = performModularityAnalysis(renamed_output_table, metadata_table);



% my_filters = struct('MetricesCategory', "Accuracy");
% analyzeMetricClusters_PCA(renamed_output_table, metadata_table, 'Filters', my_filters);


function analyzeMetricClusters_PCA(dataTable, metadataTable, varargin)
% ANALYZEMETRICCLUSTERS_PCA Performs PCA on metrics and visualizes by factor and cluster.
%
% This function takes a data table of performance metrics and a metadata table,
% performs Principal Component Analysis (PCA) on the metrics, and then applies
% k-means clustering to the resulting principal components.
%
% The primary output is a scatter plot where each point represents a performance
% metric. The position of each point is determined by its score on the first two
% principal components. The color of each point is determined by its 'CognitiveControlFactor'
% from the metadata, and its shape is determined by its assigned k-means cluster.
%
% This allows for the visual exploration of how performance metrics group together
% in a reduced-dimensional space and how these groupings relate to predefined
% cognitive factors.

    % --- 1. Argument Parsing ---
    p = inputParser;
    addRequired(p, 'dataTable', @istable);
    addRequired(p, 'metadataTable', @istable);
    addParameter(p, 'NumClusters', [], @(x) isnumeric(x) && isscalar(x) || isempty(x));
    addParameter(p, 'MaxClusters', 4, @isnumeric);
    addParameter(p, 'Normalize', true, @islogical);
    addParameter(p, 'Filters', struct(), @isstruct);
    addParameter(p, 'Exclude', {}, @iscell);
    addParameter(p, 'output_folder', 'PCA_Metric_Clusters_Results', @ischar);
    parse(p, dataTable, metadataTable, varargin{:});

    k = p.Results.NumClusters;
    max_k = p.Results.MaxClusters;
    normalizeData = p.Results.Normalize;
    filters = p.Results.Filters;
    variablesToExclude = p.Results.Exclude;
    output_folder = p.Results.output_folder;

    % --- 2. Prerequisite Checks & Data Preparation ---
    if ~license('test', 'Statistics_Toolbox'), error('This function requires the Statistics and Machine Learning Toolbox for pca() and kmeans().'); end
    if ~ismember('CognitiveControlFactor', metadataTable.Properties.VariableNames)
        error('metadataTable must contain a "CognitiveControlFactor" column.');
    end

    % Filter metrics based on metadata
    filtered_metadata = metadataTable;
    filterFields = fieldnames(filters);
    if ~isempty(filterFields)
        for i = 1:length(filterFields)
            fieldName = filterFields{i};
            filterValue = filters.(fieldName);
            filtered_metadata = filtered_metadata(strcmp(filtered_metadata.(fieldName), filterValue), :);
        end
    end

    if ~isempty(variablesToExclude)
        is_on_exclude_list = ismember(filtered_metadata.ModifiedName, variablesToExclude);
        filtered_metadata = filtered_metadata(~is_on_exclude_list, :);
    end

    if isempty(filtered_metadata), disp('No variables remained after filtering.'); return; end
    variablesForAnalysis = filtered_metadata.ModifiedName;
    fprintf('Initial analysis on %d performance metrics.\n', numel(variablesForAnalysis));

    dataForAnalysis_table = dataTable(:, variablesForAnalysis);
    
    % --- Remove Correlated Variables and Resynchronize ---
	correlatedGroups = findHighlyCorrelatedGroups(dataForAnalysis_table, filtered_metadata, 0.8);
	[dataForAnalysis_table, removedVars] = removeCorrelatedVariables(dataForAnalysis_table, correlatedGroups);
    if ~isempty(removedVars)
        fprintf('Removed %d correlated variables.\n', numel(removedVars));
        disp(removedVars);
    end
    
    % Use the modified table's columns as the single source of truth
    % to guarantee alignment of data and metadata.
    variablesForAnalysis = dataForAnalysis_table.Properties.VariableNames;
    fprintf('Proceeding with %d variables after correlation removal.\n', numel(variablesForAnalysis));
    filtered_metadata = filtered_metadata(ismember(filtered_metadata.ModifiedName, variablesForAnalysis), :);

    % --- Continue with data cleaning ---
    rowsWithNaN = any(ismissing(dataForAnalysis_table), 2);
    cleanData_table = dataForAnalysis_table(~rowsWithNaN, :);

    if any(rowsWithNaN), fprintf('Removed %d subjects/sessions with incomplete data.\n', sum(rowsWithNaN)); end
    
    % --- 3. TRANSPOSE DATA & PCA Step ---
    cleanData_matrix = cleanData_table.Variables;
    dataForMetricAnalysis = cleanData_matrix'; % Transpose so metrics are rows, subjects are columns
    
    if normalizeData
        disp('Applying Z-score normalization to each metric across subjects...');
        dataForMetricAnalysis = zscore(dataForMetricAnalysis, 0, 2);
    end

    disp('Performing PCA on performance metrics...');
    if size(dataForMetricAnalysis, 1) <= max_k
        warning('Number of metrics (%d) is less than or equal to MaxClusters (%d). Adjusting MaxClusters.', size(dataForMetricAnalysis,1), max_k);
        max_k = size(dataForMetricAnalysis, 1) - 1;
    end
    
    [~, pca_scores, ~, ~, explained] = pca(dataForMetricAnalysis);

    % --- 4. K-Means Clustering ---
    % Cluster on the first two principal components for consistency with the plot
    data_for_clustering = pca_scores(:, 1:2);

    if isempty(k)
        disp(['Finding optimal number of metric clusters (up to k=' num2str(max_k) ') using Silhouette Method...']);
        k_range = 2:max_k;
        avg_silhouette_vals = zeros(length(k_range), 1);
        parfor i = 1:length(k_range)
            current_k = k_range(i);
            try
                cluster_indices_eval = kmeans(data_for_clustering, current_k, 'Replicates', 5);
                silhouette_vals = silhouette(data_for_clustering, cluster_indices_eval);
                avg_silhouette_vals(i) = mean(silhouette_vals);
            catch ME, warning('Could not compute k=%d. Error: %s', current_k, ME.message); avg_silhouette_vals(i) = -inf; end
        end
        [~, max_idx] = max(avg_silhouette_vals);
        optimal_k = k_range(max_idx);
        k = optimal_k;
        fprintf('Automatically determined optimal number of clusters: k = %d\n', k);
    else
        fprintf('Using user-specified number of clusters: k = %d\n', k);
    end
    
    disp(['Running K-Means with k = ' num2str(k) '...']);
    cluster_indices = kmeans(data_for_clustering, k, 'Replicates', 10, 'Display', 'off');

    % --- 5. Create Plot with Custom Visualization ---
    main_fig = figure('Name', 'PCA of Metrics (Color: Factor, Shape: Cluster)', 'Position', [100 100 1200 850]);
    ax = axes(main_fig);
    hold(ax, 'on');

    factors = filtered_metadata.CognitiveControlFactor;
    unique_factors = unique(factors);
    factor_colors = lines(length(unique_factors));
    
    cluster_markers = {'o', 's', 'd', '^', 'v', 'p', 'h', '*'};
    if k > length(cluster_markers)
        warning('Number of clusters exceeds defined marker shapes. Shapes will be reused.');
    end

    for i = 1:length(variablesForAnalysis)
        factor_idx = find(strcmp(unique_factors, factors{i}));
        current_color = factor_colors(factor_idx, :);
        
        cluster_id = cluster_indices(i);
        marker_idx = mod(cluster_id - 1, length(cluster_markers)) + 1;
        current_marker = cluster_markers{marker_idx};
        
        scatter(ax, pca_scores(i, 1), pca_scores(i, 2), 80, ...
                'MarkerFaceColor', current_color, ...
                'MarkerEdgeColor', 'k', ...
                'Marker', current_marker, ...
                'MarkerFaceAlpha', 0.8, 'LineWidth', 1);
    end

    % Add text labels for each metric with dynamic offset
    dx = (max(pca_scores(:,1)) - min(pca_scores(:,1))) * 0.01;
    dy = (max(pca_scores(:,2)) - min(pca_scores(:,2))) * 0.01;
    for i = 1:length(variablesForAnalysis)
        label_name = strrep(variablesForAnalysis{i}, '_', ' ');
        text(ax, pca_scores(i,1)+dx, pca_scores(i,2)+dy, label_name, 'FontSize', 8);
    end
    
    % --- Create Manual Legends ---
    factor_handles = gobjects(length(unique_factors), 1);
    for i = 1:length(unique_factors)
        factor_handles(i) = plot(ax, NaN, NaN, 's', 'MarkerFaceColor', factor_colors(i,:), ...
                                  'MarkerSize', 10, 'LineStyle', 'none', 'DisplayName', strrep(unique_factors{i}, '_', ' '));
    end
    lgd1 = legend(ax, factor_handles, 'Location', 'eastoutside');
    title(lgd1, 'Cognitive Control Factor');
    
    ax_pos = get(ax, 'Position');
    lgd_ax = axes('Position', ax_pos, 'Visible', 'off');
    hold(lgd_ax, 'on');
    
    cluster_handles = gobjects(k, 1);
    for i = 1:k
        marker_idx = mod(i - 1, length(cluster_markers)) + 1;
        cluster_handles(i) = plot(lgd_ax, NaN, NaN, cluster_markers{marker_idx}, 'MarkerSize', 8, ...
                                    'MarkerEdgeColor', 'k', 'MarkerFaceColor', [.7 .7 .7], 'LineStyle', 'none', ...
                                    'DisplayName', ['Cluster ' num2str(i)]);
    end
    lgd2 = legend(lgd_ax, cluster_handles, 'Location', 'eastoutside');
    title(lgd2, 'K-Means Cluster');
    
    drawnow;
    set(lgd1, 'Location', 'northeast');
    pos1 = get(lgd1, 'Position');
    pos2 = get(lgd2, 'Position');
    pos2(1) = pos1(1);
    pos2(2) = pos1(2) - pos2(4) - 0.02;
    set(lgd2, 'Position', pos2);
    
    main_title = ['PCA of Performance Metrics (k=' num2str(k) ')'];
    if normalizeData, main_title = [main_title ' (Z-Scored)']; end
    title(ax, main_title, 'FontSize', 12);
    xlabel(ax, sprintf('Principal Component 1 (%.1f%% Variance)', explained(1)));
    ylabel(ax, sprintf('Principal Component 2 (%.1f%% Variance)', explained(2)));
    grid(ax, 'on'); box(ax, 'on'); axis(ax, 'square');
    hold(ax, 'off');

    % --- 6. Save Outputs ---
    if ~exist(output_folder, 'dir'), mkdir(output_folder); end
    timestamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
    main_filename = fullfile(output_folder, ['PCA_Metric_FactorColor_ClusterShape_' timestamp '.png']);
    disp(['Saving PCA plot to: ' main_filename]);
    print(main_fig, main_filename, '-dpng', '-r300');
end



function [modularityScore, communityAssignments, cleanDataTable, finalMetadata] = performModularityAnalysis(dataTable, metadataTable, varargin)
% PERFORMMODULARITYANALYSIS Cleans data, performs modularity analysis, and
% visualizes the network with nodes colored by a metadata factor.
%
% This version colors nodes based on the 'CognitiveControlFactor' column
% in the metadata table to allow for visual comparison between theoretical
% groupings and algorithmically detected communities.

    % --- Argument Parsing ---
    p = inputParser;
    addRequired(p, 'dataTable', @istable);
    addRequired(p, 'metadataTable', @istable);
    % Cleaning parameters
    addParameter(p, 'Filters', struct(), @isstruct);
    addParameter(p, 'Exclude', {}, @iscell);
    addParameter(p, 'CorrThreshold', 0.9, @(x) isnumeric(x) && x>0 && x<1);
    % Analysis & Plotting parameters
    addParameter(p, 'PlotTitle', 'Modularity Analysis', @ischar);
    addParameter(p, 'MinCorrelation', 0.3, @(x) isnumeric(x) && x>=0 && x<1);
    parse(p, dataTable, metadataTable, varargin{:});
    
    filters = p.Results.Filters;
    variablesToExclude = p.Results.Exclude;
    corrThreshold = p.Results.CorrThreshold;
    plotTitle = p.Results.PlotTitle;
    minCorrViz = p.Results.MinCorrelation;

    % --- STAGE 1: DATA CLEANING (Unchanged) ---
    disp('--- STAGE 1: DATA CLEANING ---');
    filtered_metadata = metadataTable;
    filterFields = fieldnames(filters);
    if ~isempty(filterFields)
        for i = 1:length(filterFields)
            fieldName = filterFields{i};
            filterValue = filters.(fieldName);
            filtered_metadata = filtered_metadata(strcmp(filtered_metadata.(fieldName), filterValue), :);
        end
    end
    if ~isempty(variablesToExclude), filtered_metadata = filtered_metadata(~ismember(filtered_metadata.ModifiedName, variablesToExclude), :); end
    if isempty(filtered_metadata), error('No variables remained after initial filtering.'); end
    initialVars = filtered_metadata.ModifiedName;
    dataForAnalysis = dataTable(:, initialVars);
	[dataAfterCorrRemoval, ~] = removeCorrelatedVars(dataForAnalysis, findCorrelatedGroups(dataForAnalysis, corrThreshold));
    finalVariableNames = dataAfterCorrRemoval.Properties.VariableNames;
    finalMetadata = filtered_metadata(ismember(filtered_metadata.ModifiedName, finalVariableNames), :);
    cleanDataTable = dataAfterCorrRemoval(~any(ismissing(dataAfterCorrRemoval), 2), :);
    fprintf('--- Cleaning Complete. Final data has %d rows and %d variables. ---\n\n', size(cleanDataTable, 1), size(cleanDataTable, 2));

    % --- STAGE 2: MODULARITY ANALYSIS & VISUALIZATION ---
    disp('--- STAGE 2: MODULARITY ANALYSIS & VISUALIZATION ---');
    if size(cleanDataTable, 2) < 2
        error('Cannot perform modularity analysis with fewer than 2 variables remaining.');
    end
    
    dataMatrix = cleanDataTable{:,:};
    adjacencyMatrix = abs(corr(dataMatrix));
    
    % --- Perform Community Detection (runs in background) ---
    try
        [communities, modularityScore] = community_louvain(adjacencyMatrix);
        fprintf('Louvain algorithm found %d communities with a modularity score (Q) of %.4f.\n', max(communities), modularityScore);
    catch
        communities = ones(size(finalVariableNames, 1), 1);
        modularityScore = NaN;
        warning('Could not run community detection. Modularity analysis skipped.');
    end
    communityAssignments = table(finalVariableNames', communities, 'VariableNames', {'Variable', 'CommunityID'});

    % --- VISUALIZATION: Color nodes by CognitiveControlFactor ---
    if ~ismember('CognitiveControlFactor', finalMetadata.Properties.VariableNames)
        error('The metadata table must contain a column named "CognitiveControlFactor" for coloring.');
    end
    
    % Convert CCF text labels to numeric IDs for plotting
    [ccf_ids, ccf_labels] = findgroups(finalMetadata.CognitiveControlFactor);

    fprintf('Coloring %d nodes based on %d unique Cognitive Control Factors.\n', numel(finalVariableNames), numel(ccf_labels));

    % Create graph object for plotting
    G = graph(adjacencyMatrix, finalVariableNames, 'upper');
    G_plot = rmedge(G, find(G.Edges.Weight < minCorrViz));
    
    figure('Name', plotTitle, 'NumberTitle', 'off', 'Color', 'w');
    h = plot(G_plot, 'Layout', 'force', 'NodeLabel', {});
    
    % *** KEY CHANGE: Color nodes using CCF IDs instead of community IDs ***
    h.NodeCData = ccf_ids;
    
    h.LineWidth = 5 * G_plot.Edges.Weight;
    h.MarkerSize = 10;
    text(h.XData, h.YData, h.NodeLabel, 'FontSize', 9, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    title(plotTitle, 'FontSize', 14);
    
    % *** KEY CHANGE: Update colormap and colorbar for CCF labels ***
    colormap(lines(numel(ccf_labels))); % Use a distinct color for each factor
    cb = colorbar;
    cb.Label.String = 'Cognitive Control Factor';
    % Set the ticks and labels to show the actual factor names
    if numel(ccf_labels) > 1
        cb.Ticks = linspace(min(ccf_ids), max(ccf_ids), numel(ccf_labels));
    else
        cb.Ticks = 1;
    end
    cb.TickLabels = ccf_labels;
    
    disp('Community assignments (from algorithm):');
    disp(sortrows(communityAssignments, 'CommunityID'));

end

% --- HELPER FUNCTIONS (Place at the end) ---
function groups = findCorrelatedGroups(dataTable, threshold)
    dataMatrix = dataTable{:,:};
    variableNames = dataTable.Properties.VariableNames;
    corrMatrix = abs(corr(dataMatrix, 'rows', 'pairwise'));
    isHighlyCorrelated = corrMatrix > threshold;
    isHighlyCorrelated(logical(eye(size(corrMatrix)))) = 0;
    G = graph(isHighlyCorrelated, variableNames, 'omitselfloops');
    bins = conncomp(G);
    groups = {};
    for i = 1:max(bins)
        groupMembers = variableNames(bins == i);
        if numel(groupMembers) > 1, groups{end+1} = groupMembers; end
    end
end

function [reducedTable, removedVars] = removeCorrelatedVars(dataTable, correlatedGroups)
    varsToRemove = {};
    for i = 1:length(correlatedGroups)
        varsToRemove = [varsToRemove, correlatedGroups{i}(2:end)];
    end
    removedVars = unique(varsToRemove);
    reducedTable = dataTable(:, ~ismember(dataTable.Properties.VariableNames, removedVars));
end




function analyzeMetricClusters_UMAP(dataTable, metadataTable, varargin)
% ANALYZEMETRICCLUSTERS_UMAP Performs UMAP on metrics and visualizes by factor and cluster.
% ... (function help text remains the same) ...

    % --- 1. Argument Parsing ---
    p = inputParser;
    addRequired(p, 'dataTable', @istable);
    addRequired(p, 'metadataTable', @istable);
    addParameter(p, 'NumClusters', [], @(x) isnumeric(x) && isscalar(x) || isempty(x));
    addParameter(p, 'MaxClusters', 10, @isnumeric);
    addParameter(p, 'Normalize', true, @islogical);
    addParameter(p, 'Filters', struct(), @isstruct);
    addParameter(p, 'Exclude', {}, @iscell);
    addParameter(p, 'output_folder', 'UMAP_Metric_Clusters_Results', @ischar);
    addParameter(p, 'n_neighbors', 3, @isnumeric);
    addParameter(p, 'min_dist', 0.2, @isnumeric);

    parse(p, dataTable, metadataTable, varargin{:});
    k = p.Results.NumClusters;
    max_k = p.Results.MaxClusters;
    normalizeData = p.Results.Normalize;
    filters = p.Results.Filters;
    variablesToExclude = p.Results.Exclude;
    output_folder = p.Results.output_folder;
    n_neighbors = p.Results.n_neighbors;
    min_dist = p.Results.min_dist;

    % --- 2. Prerequisite Checks & Data Preparation ---
    if ~exist('run_umap.m', 'file'), error('UMAP function "run_umap.m" not found.'); end
    if ~license('test', 'Statistics_Toolbox'), error('This function requires the Statistics and Machine Learning Toolbox.'); end
    if ~ismember('CognitiveControlFactor', metadataTable.Properties.VariableNames)
        error('metadataTable must contain a "CognitiveControlFactor" column.');
    end

    % Filter metrics based on metadata
    filtered_metadata = metadataTable;
    filterFields = fieldnames(filters);
    if ~isempty(filterFields)
        for i = 1:length(filterFields)
            fieldName = filterFields{i};
            filterValue = filters.(fieldName);
            filtered_metadata = filtered_metadata(strcmp(filtered_metadata.(fieldName), filterValue), :);
        end
    end
    if ~isempty(variablesToExclude)
        is_on_exclude_list = ismember(filtered_metadata.ModifiedName, variablesToExclude);
        filtered_metadata = filtered_metadata(~is_on_exclude_list, :);
    end
    if isempty(filtered_metadata), disp('No variables remained after filtering.'); return; end

    variablesForAnalysis = filtered_metadata.ModifiedName;
    fprintf('Initial analysis on %d performance metrics.\n', numel(variablesForAnalysis));
    
    dataForAnalysis_table = dataTable(:, variablesForAnalysis);

    % --- Remove Correlated Variables and Resynchronize ---
	correlatedGroups = findHighlyCorrelatedGroups(dataForAnalysis_table, filtered_metadata, 0.9);
	[dataForAnalysis_table, removedVars] = removeCorrelatedVariables(dataForAnalysis_table, correlatedGroups);
    if ~isempty(removedVars)
        fprintf('Removed %d correlated variables.\n', numel(removedVars));
        disp(removedVars);
    end
    
    % =====================================================================
    % **FIX**: Use the modified table's columns as the single source of truth
    % to guarantee alignment of data and metadata.
    % =====================================================================
    % 1. Get the definitive list of variables from the *actual* columns remaining.
    variablesForAnalysis = dataForAnalysis_table.Properties.VariableNames;
    fprintf('Proceeding with %d variables after correlation removal.\n', numel(variablesForAnalysis));

    % 2. Update the metadata table to perfectly match this definitive list.
    filtered_metadata = filtered_metadata(ismember(filtered_metadata.ModifiedName, variablesForAnalysis), :);

    % --- Continue with data cleaning ---
    rowsWithNaN = any(ismissing(dataForAnalysis_table), 2);
    cleanData_table = dataForAnalysis_table(~rowsWithNaN, :);
	
    if any(rowsWithNaN), fprintf('Removed %d subjects/sessions with incomplete data.\n', sum(rowsWithNaN)); end
    if height(cleanData_table) < n_neighbors, error('Fewer subjects (%d) than n_neighbors (%d). Reduce n_neighbors.', height(cleanData_table), n_neighbors); end
    
    % --- 3. TRANSPOSE DATA & UMAP Step ---
    cleanData_matrix = cleanData_table.Variables;
    dataForMetricAnalysis = cleanData_matrix';

    if normalizeData
        disp('Applying Z-score normalization to each metric across subjects...');
        dataForMetricAnalysis = zscore(dataForMetricAnalysis, 0, 2);
    end

    disp('Performing UMAP on performance metrics...');
    if size(dataForMetricAnalysis, 1) <= max_k
        warning('Number of metrics (%d) is less than or equal to MaxClusters (%d). Adjusting MaxClusters.', size(dataForMetricAnalysis,1), max_k);
        max_k = size(dataForMetricAnalysis, 1) - 1;
    end
    
    umap_scores = run_umap(dataForMetricAnalysis, 'n_neighbors', n_neighbors, 'min_dist', min_dist, 'verbose', 'none');

    % --- 4. K-Means Clustering ---
    if isempty(k)
        disp(['Finding optimal number of metric clusters (up to k=' num2str(max_k) ') using Silhouette Method...']);
        k_range = 2:max_k;
        avg_silhouette_vals = zeros(length(k_range), 1);
        parfor i = 1:length(k_range)
            current_k = k_range(i);
            try
                cluster_indices_eval = kmeans(umap_scores, current_k, 'Replicates', 5);
                silhouette_vals = silhouette(umap_scores, cluster_indices_eval);
                avg_silhouette_vals(i) = mean(silhouette_vals);
            catch ME, warning('Could not compute k=%d. Error: %s', current_k, ME.message); avg_silhouette_vals(i) = -inf; end
        end
        [~, max_idx] = max(avg_silhouette_vals);
        optimal_k = k_range(max_idx);
        k = optimal_k;
        fprintf('Automatically determined optimal number of clusters: k = %d\n', k);
    else
        fprintf('Using user-specified number of clusters: k = %d\n', k);
    end
    
    disp(['Running K-Means with k = ' num2str(k) '...']);
    cluster_indices = kmeans(umap_scores, k, 'Replicates', 10, 'Display', 'off');

    % --- 5. Create Plot with Custom Visualization ---
    main_fig = figure('Name', 'UMAP of Metrics (Color: Factor, Shape: Cluster)', 'Position', [100 100 1200 850]);
    ax = axes(main_fig);
    hold(ax, 'on');

    % Get factor info from the *synchronized* metadata for coloring
    factors = filtered_metadata.CognitiveControlFactor;
    unique_factors = unique(factors);
    factor_colors = lines(length(unique_factors));
    
    % Define marker shapes for clusters
    cluster_markers = {'o', 's', 'd', '^', 'v', 'p', 'h', '*'};
    if k > length(cluster_markers)
        warning('Number of clusters exceeds defined marker shapes. Shapes will be reused.');
    end

    % Plot each point individually to control color and shape
    for i = 1:length(variablesForAnalysis)
        % COLOR determined by CognitiveControlFactor
        factor_idx = find(strcmp(unique_factors, factors{i}));
        current_color = factor_colors(factor_idx, :);
        
        % SHAPE determined by K-Means cluster_indices
        cluster_id = cluster_indices(i);
        marker_idx = mod(cluster_id - 1, length(cluster_markers)) + 1;
        current_marker = cluster_markers{marker_idx};
        
        scatter(ax, umap_scores(i, 1), umap_scores(i, 2), 80, ...
                'MarkerFaceColor', current_color, ...
                'MarkerEdgeColor', 'k', ...
                'Marker', current_marker, ...
                'MarkerFaceAlpha', 0.8, 'LineWidth', 1);
    end

    % % Add text labels for each metric
    % dx = 0.02; dy = 0.02; % text offset
    % for i = 1:length(variablesForAnalysis)
    %     label_name = strrep(variablesForAnalysis{i}, '_', ' ');
    %     text(ax, umap_scores(i,1)+dx, umap_scores(i,2)+dy, label_name, 'FontSize', 8);
    % end
    
    % --- Create Manual Legends ---
    % Legend for Cognitive Factors (Color)
    factor_handles = gobjects(length(unique_factors), 1);
    for i = 1:length(unique_factors)
        factor_handles(i) = plot(ax, NaN, NaN, 's', 'MarkerFaceColor', factor_colors(i,:), ...
                                  'MarkerSize', 10, 'LineStyle', 'none', 'DisplayName', strrep(unique_factors{i}, '_', ' '));
    end
    lgd1 = legend(ax, factor_handles, 'Location', 'northeast');
    title(lgd1, 'Cognitive Control Factor');
    
    % Create an invisible axes for the second legend (Clusters)
    ax_pos = get(ax, 'Position');
    lgd_ax = axes('Position', ax_pos, 'Visible', 'off');
    hold(lgd_ax, 'on');
    
    cluster_handles = gobjects(k, 1);
    for i = 1:k
        marker_idx = mod(i - 1, length(cluster_markers)) + 1;
        cluster_handles(i) = plot(lgd_ax, NaN, NaN, cluster_markers{marker_idx}, 'MarkerSize', 8, ...
                                    'MarkerEdgeColor', 'k', 'MarkerFaceColor', [.7 .7 .7], 'LineStyle', 'none', ...
                                    'DisplayName', ['Cluster ' num2str(i)]);
    end
    lgd2 = legend(lgd_ax, cluster_handles, 'Location', 'east');
    title(lgd2, 'K-Means Cluster');
    
    % Adjust legend positions to not overlap
    drawnow;
    set(lgd1, 'Location', 'northeast');
    pos1 = get(lgd1, 'Position');
    pos2 = get(lgd2, 'Position');
    pos2(1) = pos1(1);
    pos2(2) = pos1(2) - pos2(4) - 0.02; % Position second legend below the first
    set(lgd2, 'Position', pos2);
    
    main_title = ['UMAP of Performance Metrics (k=' num2str(k) ')'];
    if normalizeData, main_title = [main_title ' (Z-Scored)']; end
    title(ax, main_title, 'FontSize', 12);
    xlabel(ax, 'UMAP Dimension 1');
    ylabel(ax, 'UMAP Dimension 2');
    grid(ax, 'off'); box(ax, 'off'); axis(ax, 'square');
    hold(ax, 'off');

    % Save the main figure
    if ~exist(output_folder, 'dir'), mkdir(output_folder); end
    timestamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
    main_filename = fullfile(output_folder, ['UMAP_Metric_FactorColor_ClusterShape_' timestamp '.png']);
    disp(['Saving UMAP plot to: ' main_filename]);
    print(main_fig, main_filename, '-dpng', '-r300');
end




function modified_dataTable = plotUMAP_KMeans_Silhouette(dataTable, metadataTable, varargin)
% PLOTUMAP_KMEANS_SILHOUETTE Performs UMAP and then K-Means clustering.
%
% This function reduces data dimensionality using UMAP and then applies K-Means
% to partition the data into a specified number of clusters. It can
% automatically determine the optimal number of clusters ('k') using the
% Silhouette Method.
%
% PREREQUISITE: Requires the UMAP package by Stephen Meehan (File Exchange)
%               and the Statistics and Machine Learning Toolbox (for kmeans, silhouette).
%
% Args:
%       dataTable (table): Main data table with 'SubjectID'.
%       metadataTable (table): Metadata for variable filtering.
% Optional Name-Value Args:
%       'NumClusters' (numeric): The number of clusters 'k' for K-Means.
%                                If empty, 'k' is found automatically. Default is [].
%       'MaxClusters' (numeric): The maximum 'k' to test for the Silhouette Method. Default is 15.
%       'Normalize' (logical): If true, applies Z-score normalization before UMAP. Default is true.
%       (Includes other parameters like 'Filters', 'Exclude', 'output_folder', 'n_neighbors', 'min_dist')
%
% Returns:
%       modified_dataTable (table): Copy of dataTable with UMAP and KMeans_ClusterID columns.

    % --- 1. Argument Parsing ---
    p = inputParser;
    addRequired(p, 'dataTable', @istable);
    addRequired(p, 'metadataTable', @istable);
    
    % K-Means specific parameters
    addParameter(p, 'NumClusters', [], @(x) isnumeric(x) && isscalar(x) || isempty(x));
    addParameter(p, 'MaxClusters', 15, @isnumeric);
    
    % Shared parameters
    addParameter(p, 'Normalize', true, @islogical);
    addParameter(p, 'Filters', struct(), @isstruct);
    addParameter(p, 'Exclude', {}, @iscell);
    addParameter(p, 'output_folder', 'UMAP_KMeans_Results', @ischar);
    addParameter(p, 'n_neighbors', 15, @isnumeric);
    addParameter(p, 'min_dist', 0.1, @isnumeric);
    
    parse(p, dataTable, metadataTable, varargin{:});
    k = p.Results.NumClusters;
    max_k = p.Results.MaxClusters;
    normalizeData = p.Results.Normalize;
    filters = p.Results.Filters;
    variablesToExclude = p.Results.Exclude;
    output_folder = p.Results.output_folder;
    n_neighbors = p.Results.n_neighbors;
    min_dist = p.Results.min_dist;
    
    % --- 2. Prerequisite Checks & Data Prep ---
    if ~exist('run_umap.m', 'file'), error('UMAP function "run_umap.m" not found.'); end
    if ~license('test', 'Statistics_Toolbox'), error('This function requires the Statistics and Machine Learning Toolbox.'); end
    
    modified_dataTable = dataTable;
    if ismember('UMAP1', modified_dataTable.Properties.VariableNames), modified_dataTable.UMAP1 = []; end
    if ismember('UMAP2', modified_dataTable.Properties.VariableNames), modified_dataTable.UMAP2 = []; end
    if ismember('KMeans_ClusterID', modified_dataTable.Properties.VariableNames), modified_dataTable.KMeans_ClusterID = []; end
    
    filtered_metadata = metadataTable;
    filterFields = fieldnames(filters);
    if ~isempty(filterFields)
        for i = 1:length(filterFields), fieldName = filterFields{i}; filterValue = filters.(fieldName); filtered_metadata = filtered_metadata(strcmp(filtered_metadata.(fieldName), filterValue), :); end
    end
    if ~isempty(variablesToExclude), is_on_exclude_list = ismember(filtered_metadata.ModifiedName, variablesToExclude); filtered_metadata = filtered_metadata(~is_on_exclude_list, :); end
    if isempty(filtered_metadata), disp('No variables remained after filtering.'); return; end
    
    variablesForAnalysis = filtered_metadata.ModifiedName;
    fprintf('Performing analysis based on %d variables.\n', numel(variablesForAnalysis));
    
    dataForAnalysis_table = dataTable(:, variablesForAnalysis);
    subjectIDs = dataTable.SubjectID;
    
    rowsWithNaN = any(ismissing(dataForAnalysis_table), 2);
    cleanData_table = dataForAnalysis_table(~rowsWithNaN, :);
    dataForAnalysis_matrix = cleanData_table.Variables;
    subjectIDs_cleaned = subjectIDs(~rowsWithNaN);
    
    if any(rowsWithNaN), fprintf('Removed %d sessions with incomplete data.\n', sum(rowsWithNaN)); end
    if size(dataForAnalysis_matrix, 1) < 2, error('Not enough data to perform analysis after cleaning.'); end
    if size(dataForAnalysis_matrix, 1) <= max_k, warning('Number of data points (%d) is less than or equal to MaxClusters (%d). Adjust MaxClusters.', size(dataForAnalysis_matrix,1), max_k); max_k = size(dataForAnalysis_matrix, 1) - 1; end

    % --- 3. UMAP Step ---
    dataForUmap = dataForAnalysis_matrix;
    if normalizeData, disp('Applying Z-score normalization to data before UMAP...'); dataForUmap = zscore(dataForAnalysis_matrix); end
    
    disp('Performing UMAP...');
    umap_scores = run_umap(dataForUmap, 'n_neighbors', n_neighbors, 'min_dist', min_dist, 'verbose', 'none');
    
    % --- 4. K-Means Clustering ---
    % Create output directory and define a unique timestamp for filenames
    if ~exist(output_folder, 'dir'), mkdir(output_folder); end
    timestamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
    
    if isempty(k)
        % --- Automatic 'k' detection using the Silhouette Method ---
        disp(['Finding optimal number of clusters (up to k=' num2str(max_k) ') using Silhouette Method...']);
        
        k_range = 2:max_k; % Silhouette is not defined for k=1
        avg_silhouette_vals = zeros(length(k_range), 1);
        
        % Use a parallel loop for efficiency if Parallel Computing Toolbox is available
        parfor i = 1:length(k_range)
            current_k = k_range(i);
            try
                cluster_indices_eval = kmeans(umap_scores, current_k, 'Replicates', 5, 'MaxIter', 500);
                silhouette_vals = silhouette(umap_scores, cluster_indices_eval);
                avg_silhouette_vals(i) = mean(silhouette_vals);
            catch ME
                warning('Could not compute k=%d. Error: %s', current_k, ME.message);
                avg_silhouette_vals(i) = -inf; % Assign a very low value on error
            end
        end
        
        % Find the k with the maximum average silhouette value
        [~, max_idx] = max(avg_silhouette_vals);
        optimal_k = k_range(max_idx);
        
        k = optimal_k;
        fprintf('Automatically determined optimal number of clusters: k = %d\n', k);
        
        % --- Create and save the Silhouette Plot for diagnostics ---
        silhouette_fig = figure('Name', 'K-Means Silhouette Method', 'Position', [200 200 700 500], 'Visible', 'off');
        plot(k_range, avg_silhouette_vals, 'b-o', 'LineWidth', 1.5);
        hold on;
        plot(optimal_k, avg_silhouette_vals(max_idx), 'r*', 'MarkerSize', 15, 'LineWidth', 2);
        title('Silhouette Method for Optimal k');
        xlabel('Number of Clusters (k)');
        ylabel('Average Silhouette Value');
        legend('Avg. Silhouette', ['Optimal k = ' num2str(optimal_k)]);
        grid on;
        
        % Adjust Y-axis limit for better visualization
        current_ylim = ylim;
        ylim([current_ylim(1), max(avg_silhouette_vals) + 0.2]);

        silhouette_base_filename = ['Silhouette_Plot_' timestamp];
        silhouette_filename_png = fullfile(output_folder, [silhouette_base_filename '.png']);
        silhouette_filename_eps = fullfile(output_folder, [silhouette_base_filename '.eps']);
        
        disp(['Saving Silhouette Plot to PNG: ' silhouette_filename_png]);
        print(silhouette_fig, silhouette_filename_png, '-dpng', '-r300');
        
        disp(['Saving Silhouette Plot to EPS: ' silhouette_filename_eps]);
        print(silhouette_fig, silhouette_filename_eps, '-depsc', '-painters');
        close(silhouette_fig);
    else
        fprintf('Using user-specified number of clusters: k = %d\n', k);
    end
    
    % --- Run K-Means with the chosen 'k' ---
    disp(['Running K-Means with k = ' num2str(k) '...']);
    cluster_indices = kmeans(umap_scores, k, 'Replicates', 10, 'MaxIter', 1000, 'Display', 'off');
    
    % --- 5. Add Data to Output Table & Plot Results ---
    modified_dataTable.UMAP1 = nan(height(modified_dataTable), 1);
    modified_dataTable.UMAP2 = nan(height(modified_dataTable), 1);
    modified_dataTable.KMeans_ClusterID = nan(height(modified_dataTable), 1);
    
    modified_dataTable.UMAP1(~rowsWithNaN) = umap_scores(:,1);
    modified_dataTable.UMAP2(~rowsWithNaN) = umap_scores(:,2);
    modified_dataTable.KMeans_ClusterID(~rowsWithNaN) = cluster_indices;
    
    main_fig = figure('Name', 'UMAP with K-Means Clusters', 'Position', [100 100 1100 800], 'Visible', 'off');
    ax = axes(main_fig);
    hold(ax, 'on');
    
    unique_subjects = unique(subjectIDs_cleaned);
    
    % MODIFICATION: Use custom colors for subjects
    custom_colors_rgb = [
        245/255, 124/255, 110/255; 242/255, 181/255, 110/255; 251/255, 231/255, 158/255;
        132/255, 195/255, 183/255; 136/255, 215/255, 218/255; 113/255, 184/255, 237/255;
        184/255, 174/255, 234/255; 242/255, 168/255, 218/255
    ];
    subject_colors = custom_colors_rgb;
    if length(unique_subjects) > size(subject_colors, 1)
        warning('More subjects than custom colors provided. Colors will be recycled.');
    end
    
    cluster_markers = {'o', 's', 'd', '^', 'v', 'p', 'h', '*'}; % Define enough markers
    if k > length(cluster_markers)
        warning('Number of clusters exceeds the number of defined marker shapes. Some clusters will have repeated shapes.');
    end
    
    for i = 1:length(umap_scores)
        subject_idx = find(strcmp(unique_subjects, subjectIDs_cleaned(i)));
        % Use modulo to cycle through colors if subjects > custom colors
        color_index = mod(subject_idx - 1, size(subject_colors, 1)) + 1;
        
        cluster_id = cluster_indices(i);
        marker_index = mod(cluster_id - 1, length(cluster_markers)) + 1; % Cycle through markers
        scatter(ax, umap_scores(i, 1), umap_scores(i, 2), 100, subject_colors(color_index,:), ...
                cluster_markers{marker_index}, 'filled');
    end
    
    % --- Create Manual Legends for Subjects (Color) and Clusters (Shape) ---
    % Legend for Subjects
    subject_handles = gobjects(length(unique_subjects), 1);
    for i = 1:length(unique_subjects)
        % MODIFICATION: Rename legend label to 'Subject X'
        s_label = ['Subject ' unique_subjects{i}(1)];
        
        % Use modulo to cycle through colors for the legend to match the plot
        color_index = mod(i - 1, size(subject_colors, 1)) + 1;
        
        subject_handles(i) = plot(ax, NaN, NaN, 's', 'MarkerFaceColor', subject_colors(color_index,:), ...
                                  'MarkerSize', 10, 'LineStyle', 'none', 'DisplayName', s_label);
    end
    lgd1 = legend(ax, subject_handles, 'Location', 'northeast');
    % MODIFICATION: Rename legend title
    title(lgd1, 'Subjects');
    
    % Legend for Clusters (Shape)
    ax_pos = get(ax, 'Position');
    lgd_ax = axes('Position', ax_pos, 'Visible', 'off'); % Invisible axes for second legend
    hold(lgd_ax, 'on');
    
    cluster_handles = gobjects(k, 1);
    cluster_labels = cell(k, 1);
    for i = 1:k
        marker_index = mod(i - 1, length(cluster_markers)) + 1;
        cluster_handles(i) = plot(lgd_ax, NaN, NaN, cluster_markers{marker_index}, 'MarkerSize', 8, ...
                                    'MarkerEdgeColor', 'k', 'LineStyle', 'none');
        cluster_labels{i} = ['Cluster ' num2str(i)];
    end
    
    lgd2 = legend(lgd_ax, cluster_handles, cluster_labels, 'Location', 'east');
    title(lgd2, 'K-Means Clusters');
    
    % Adjust legend positions to not overlap
    drawnow;
    set(lgd1, 'Location', 'northeast');
    pos1 = get(lgd1, 'Position');
    pos2 = get(lgd2, 'Position');
    pos2(1) = pos1(1);
    pos2(2) = pos1(2) - pos2(4) - 0.01;
    set(lgd2, 'Position', pos2);
    main_title = ['UMAP Embedding with K-Means Clustering (k=' num2str(k) ')'];
    if normalizeData, main_title = [main_title ' (Z-Scored)']; end
    title(ax, main_title, 'FontSize', 12);
    xlabel(ax, 'UMAP Dimension 1');
    ylabel(ax, 'UMAP Dimension 2');
    grid(ax, 'off'); box(ax, 'off'); axis(ax, 'square');
    
    % --- 6. Save Output Files ---
    % Save main cluster plot
    main_base_filename = ['UMAP_KMeans_Plot_' timestamp];
    main_filename_png = fullfile(output_folder, [main_base_filename '.png']);
    main_filename_eps = fullfile(output_folder, [main_base_filename '.eps']);
    
    disp(['Saving K-Means Cluster Plot to PNG: ' main_filename_png]);
    print(main_fig, main_filename_png, '-dpng', '-r300');
    
    disp(['Saving K-Means Cluster Plot to EPS: ' main_filename_eps]);
    print(main_fig, main_filename_eps, '-depsc', '-vector');
    close(main_fig);

    % Save plot data to CSV
    disp('Saving plot data to CSV file...');
    plottingData = table(subjectIDs_cleaned, umap_scores(:,1), umap_scores(:,2), cluster_indices, ...
        'VariableNames', {'SubjectID', 'UMAP1', 'UMAP2', 'ClusterID'});
    csv_filename = fullfile(output_folder, ['UMAP_KMeans_PlotData_' timestamp '.csv']);
    writetable(plottingData, csv_filename);
    disp(['Plotting data saved to: ' csv_filename]);
    
end

function modified_dataTable = plotUMAP_KMeans(dataTable, metadataTable, varargin)
% PLOTUMAP_KMEANS Performs UMAP and then K-Means clustering.
%
% This function reduces data dimensionality using UMAP and then applies K-Means
% to partition the data into a specified number of clusters. It can
% automatically determine the optimal number of clusters ('k') using the
% Elbow Method.
%
% PREREQUISITE: Requires the UMAP package by Stephen Meehan (File Exchange)
%               and the Statistics and Machine Learning Toolbox (for kmeans).
%
% Args:
%       dataTable (table): Main data table with 'SubjectID'.
%       metadataTable (table): Metadata for variable filtering.
% Optional Name-Value Args:
%       'NumClusters' (numeric): The number of clusters 'k' for K-Means.
%                                If empty, 'k' is found automatically. Default is [].
%       'MaxClusters' (numeric): The maximum 'k' to test for the Elbow Method. Default is 15.
%       'Normalize' (logical): If true, applies Z-score normalization before UMAP. Default is true.
%       (Includes other parameters like 'Filters', 'Exclude', 'output_folder', 'n_neighbors', 'min_dist')
%
% Returns:
%       modified_dataTable (table): Copy of dataTable with UMAP and KMeans_ClusterID columns.

    % --- 1. Argument Parsing ---
    p = inputParser;
    addRequired(p, 'dataTable', @istable);
    addRequired(p, 'metadataTable', @istable);
    
    % K-Means specific parameters
    addParameter(p, 'NumClusters', [], @(x) isnumeric(x) && isscalar(x) || isempty(x));
    addParameter(p, 'MaxClusters', 15, @isnumeric);
    
    % Shared parameters
    addParameter(p, 'Normalize', true, @islogical);
    addParameter(p, 'Filters', struct(), @isstruct);
    addParameter(p, 'Exclude', {}, @iscell);
    addParameter(p, 'output_folder', 'UMAP_KMeans_Results', @ischar);
    addParameter(p, 'n_neighbors', 15, @isnumeric);
    addParameter(p, 'min_dist', 0.1, @isnumeric);
    
    parse(p, dataTable, metadataTable, varargin{:});

    k = p.Results.NumClusters;
    max_k = p.Results.MaxClusters;
    normalizeData = p.Results.Normalize;
    filters = p.Results.Filters;
    variablesToExclude = p.Results.Exclude;
    output_folder = p.Results.output_folder;
    n_neighbors = p.Results.n_neighbors;
    min_dist = p.Results.min_dist;
    
    % --- Prerequisite Checks & Data Prep (Identical to previous function) ---
    % ... (Full code for checks, filtering, NaN removal) ...
    if ~exist('run_umap.m', 'file'), error('UMAP function "run_umap.m" not found.'); end
    if ~license('test', 'Statistics_Toolbox'), error('This function requires the Statistics and Machine Learning Toolbox.'); end
    modified_dataTable = dataTable;
    if ismember('UMAP1', modified_dataTable.Properties.VariableNames), modified_dataTable.UMAP1 = []; end
    if ismember('UMAP2', modified_dataTable.Properties.VariableNames), modified_dataTable.UMAP2 = []; end
    if ismember('KMeans_ClusterID', modified_dataTable.Properties.VariableNames), modified_dataTable.KMeans_ClusterID = []; end
    filtered_metadata = metadataTable;
    filterFields = fieldnames(filters);
    if ~isempty(filterFields)
        for i = 1:length(filterFields), fieldName = filterFields{i}; filterValue = filters.(fieldName); filtered_metadata = filtered_metadata(strcmp(filtered_metadata.(fieldName), filterValue), :); end
    end
    if ~isempty(variablesToExclude), is_on_exclude_list = ismember(filtered_metadata.ModifiedName, variablesToExclude); filtered_metadata = filtered_metadata(~is_on_exclude_list, :); end
    if isempty(filtered_metadata), disp('No variables remained after filtering.'); return; end
    variablesForAnalysis = filtered_metadata.ModifiedName;
    fprintf('Performing analysis based on %d variables.\n', numel(variablesForAnalysis));
    dataForAnalysis_table = dataTable(:, variablesForAnalysis);
    subjectIDs = dataTable.SubjectID;
    rowsWithNaN = any(ismissing(dataForAnalysis_table), 2);
    cleanData_table = dataForAnalysis_table(~rowsWithNaN, :);
    dataForAnalysis_matrix = cleanData_table.Variables;
    subjectIDs_cleaned = subjectIDs(~rowsWithNaN);
    if any(rowsWithNaN), fprintf('Removed %d sessions with incomplete data.\n', sum(rowsWithNaN)); end
    if size(dataForAnalysis_matrix, 1) < 2, error('Not enough data to perform analysis after cleaning.'); end

    % --- UMAP Step (Identical to previous function) ---
    dataForUmap = dataForAnalysis_matrix;
    if normalizeData, disp('Applying Z-score normalization to data before UMAP...'); dataForUmap = zscore(dataForAnalysis_matrix); end
    disp('Performing UMAP...');
    umap_scores = run_umap(dataForUmap, 'n_neighbors', n_neighbors, 'min_dist', min_dist, 'verbose', 'none');

    % --- 4. K-Means Clustering ---
    if isempty(k)
        % --- Automatic 'k' detection using the Elbow Method ---
        disp(['Finding optimal number of clusters (up to k=' num2str(max_k) ')...']);
        
        sumd_vals = zeros(max_k, 1);
        k_range = 1:max_k;
        
        for i = k_range
            [~, ~, sumd] = kmeans(umap_scores, i, 'Replicates', 5, 'MaxIter', 500);
            sumd_vals(i) = sum(sumd);
        end
        
        % Find the "elbow" point algorithmically
        points = [k_range', sumd_vals];
        lineVec = points(end,:) - points(1,:);
        lineVecN = lineVec / norm(lineVec);
        vecFromFirst = bsxfun(@minus, points, points(1,:));
        scalarProduct = vecFromFirst * lineVecN';
        vecFromFirstParallel = scalarProduct * lineVecN;
        vecToLine = vecFromFirst - vecFromFirstParallel;
        distToLine = sqrt(sum(vecToLine.^2, 2));
        [~, optimal_k] = max(distToLine);
        
        k = optimal_k;
        fprintf('Automatically determined optimal number of clusters: k = %d\n', k);
        
        % --- Create and save the Elbow Plot for diagnostics ---
        elbow_fig = figure('Name', 'K-Means Elbow Method', 'Position', [200 200 700 500]);
        plot(k_range, sumd_vals, 'b-o', 'LineWidth', 1.5);
        hold on;
        plot(optimal_k, sumd_vals(optimal_k), 'r*', 'MarkerSize', 15, 'LineWidth', 2);
        title('Elbow Method for Optimal k');
        xlabel('Number of Clusters (k)');
        ylabel('Total Within-Cluster Sum of Squares');
        legend('Sum of Squares', ['Optimal k = ' num2str(optimal_k)]);
        grid on;
        
        if ~exist(output_folder, 'dir'), mkdir(output_folder); end
        timestamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
        elbow_filename = fullfile(output_folder, ['Elbow_Plot_' timestamp '.png']);
        disp(['Saving Elbow Plot to: ' elbow_filename]);
        print(elbow_fig, elbow_filename, '-dpng', '-r300');
    else
        fprintf('Using user-specified number of clusters: k = %d\n', k);
    end
    
    % --- Run K-Means with the chosen 'k' ---
    disp(['Running K-Means with k = ' num2str(k) '...']);
    cluster_indices = kmeans(umap_scores, k, 'Replicates', 10, 'MaxIter', 1000, 'Display', 'off');

 % --- 5. Add Data to Output Table & Plot Results ---
    modified_dataTable.UMAP1(~rowsWithNaN) = umap_scores(:,1);
    modified_dataTable.UMAP2(~rowsWithNaN) = umap_scores(:,2);
    modified_dataTable.KMeans_ClusterID(~rowsWithNaN) = cluster_indices;
    
    main_fig = figure('Name', 'UMAP with K-Means Clusters (Color by Subject, Shape by Cluster)', 'Position', [100 100 1100 800]);
    ax = axes(main_fig);
    hold(ax, 'on');
    
    unique_subjects = unique(subjectIDs_cleaned);
    subject_colors = lines(length(unique_subjects));
    cluster_markers = {'o', 's', 'd', '^', 'v', 'p', 'h', '*'}; % Define enough markers
    if k > length(cluster_markers)
        warning('Number of clusters exceeds the number of defined marker shapes. Some clusters will have repeated shapes.');
    end
    
    for i = 1:length(umap_scores)
        subject_idx = find(strcmp(unique_subjects, subjectIDs_cleaned(i)));
        cluster_id = cluster_indices(i);
        marker_index = mod(cluster_id - 1, length(cluster_markers)) + 1; % Cycle through markers
        scatter(ax, umap_scores(i, 1), umap_scores(i, 2), 60, subject_colors(subject_idx,:), ...
                cluster_markers{marker_index}, 'filled', 'MarkerFaceAlpha', 0.7);
    end
    
    % --- Create Manual Legends for Subjects (Color) and Clusters (Shape) ---
    % Legend for Subjects
    subject_handles = gobjects(length(unique_subjects), 1);
    for i = 1:length(unique_subjects)
        s_label = strrep(unique_subjects{i}, '_', ' ');
        subject_handles(i) = plot(ax, NaN, NaN, 's', 'MarkerFaceColor', subject_colors(i,:), ...
                                  'MarkerSize', 10, 'LineStyle', 'none', 'DisplayName', s_label);
    end
    lgd1 = legend(ax, subject_handles, 'Location', 'northeast');
    title(lgd1, 'Subjects');
    
    % Legend for Clusters (Shape)
    ax_pos = get(ax, 'Position');
    lgd_ax = axes('Position', ax_pos, 'Visible', 'off'); % Invisible axes for second legend
    hold(lgd_ax, 'on');
    
    cluster_handles = gobjects(k, 1);
    cluster_labels = cell(k, 1);
    for i = 1:k
        marker_index = mod(i - 1, length(cluster_markers)) + 1;
        cluster_handles(i) = plot(lgd_ax, NaN, NaN, cluster_markers{marker_index}, 'MarkerSize', 8, ...
                                    'MarkerEdgeColor', 'k', 'LineStyle', 'none');
        cluster_labels{i} = ['Cluster ' num2str(i)];
    end
    
    lgd2 = legend(lgd_ax, cluster_handles, cluster_labels, 'Location', 'east');
    title(lgd2, 'K-Means Clusters');
    
    % Adjust legend positions to not overlap
    drawnow;
    set(lgd1, 'Location', 'northeast');
    pos1 = get(lgd1, 'Position');
    pos2 = get(lgd2, 'Position');
    pos2(1) = pos1(1);
    pos2(2) = pos1(2) - pos2(4) - 0.01;
    set(lgd2, 'Position', pos2);

    main_title = ['UMAP Embedding with K-Means Clustering (k=' num2str(k) ')'];
    if normalizeData, main_title = [main_title ' (Z-Scored)']; end
    title(ax, main_title, 'FontSize', 12);
    xlabel(ax, 'UMAP Dimension 1');
    ylabel(ax, 'UMAP Dimension 2');
    grid(ax, 'on'); box(ax, 'on'); axis(ax, 'square');
    
    main_filename = fullfile(output_folder, ['UMAP_KMeans_SubjectColor_ClusterShape_' timestamp '.png']);
    disp(['Saving K-Means Cluster Plot (Color by Subject, Shape by Cluster) to: ' main_filename]);
    print(main_fig, main_filename, '-dpng', '-r300');
    hold(ax, 'off');
end


function modified_dataTable = plotUMAP_DBSCAN(dataTable, metadataTable, varargin)
% PLOTUMAP_DBSCAN_V2 Performs UMAP (with optional normalization) and DBSCAN.
%
% Version 2: Corrects input parser logic and adds optional Z-score normalization
% before running UMAP, which is highly recommended.
%
% PREREQUISITE: Requires the UMAP package by Stephen Meehan from the
%               MATLAB File Exchange.
%
% Args:
%       dataTable (table): Main data table, must contain 'SubjectID'.
%       metadataTable (table): Metadata for variable filtering.
% Optional Name-Value Args:
%       'Normalize' (logical): If true, applies Z-score normalization before UMAP. Default is true.
%       'Filters' (struct):   Keeps variables matching these criteria.
%       'Exclude' (cell array): Removes specific variables.
%       'output_folder' (string): Folder to save the figure. Default is 'UMAP_DBSCAN_Results'.
%       'MinPoints' (numeric):  The 'MinPts' parameter for DBSCAN. Default is 15.
%       'Epsilon' (numeric):    The 'epsilon' for DBSCAN. If empty, it's auto-estimated.
%       'n_neighbors' (numeric): UMAP parameter for local/global balance. Default is 15.
%       'min_dist' (numeric):    UMAP parameter for cluster packing. Default is 0.1.
%
% Returns:
%       modified_dataTable (table): Copy of dataTable with UMAP and ClusterID columns.

    % --- 1. Argument Parsing ---
    p = inputParser;
    
    %%% FIX: Properly define required arguments before parsing.
    addRequired(p, 'dataTable', @istable);
    addRequired(p, 'metadataTable', @istable);
    
    %%% NEW: Add Normalize flag. Defaults to true.
    addParameter(p, 'Normalize', true, @islogical);
    
    addParameter(p, 'Filters', struct(), @isstruct);
    addParameter(p, 'Exclude', {}, @iscell);
    addParameter(p, 'output_folder', 'UMAP_DBSCAN_Results', @ischar);
    addParameter(p, 'MinPoints', 10, @isnumeric);
    addParameter(p, 'Epsilon', 0.5, @(x) isnumeric(x) || isempty(x));
    addParameter(p, 'n_neighbors', 10, @isnumeric);
    addParameter(p, 'min_dist', 0.1, @isnumeric);
    
    %%% FIX: The parse call now correctly includes all arguments.
    parse(p, dataTable, metadataTable, varargin{:});

    % Assign parsed variables
    normalizeData = p.Results.Normalize;
    filters = p.Results.Filters;
    variablesToExclude = p.Results.Exclude;
    output_folder = p.Results.output_folder;
    minPts = p.Results.MinPoints;
    epsilon = p.Results.Epsilon;
    n_neighbors = p.Results.n_neighbors;
    min_dist = p.Results.min_dist;
    
    % --- Prerequisite Checks ---
    if ~exist('run_umap.m', 'file')
        error('UMAP function "run_umap.m" not found. Please add the package to your MATLAB path.');
    end
    if ~license('test', 'Statistics_Toolbox')
        error('This function requires the Statistics and Machine Learning Toolbox for dbscan().');
    end

    % --- Data Preparation (Identical structure) ---
    modified_dataTable = dataTable;
    % (Identical code for clearing old columns, filtering variables, and removing NaNs)
    % ...
    filtered_metadata = metadataTable;
    filterFields = fieldnames(filters);
    if ~isempty(filterFields)
        for i = 1:length(filterFields)
            fieldName = filterFields{i};
            filterValue = filters.(fieldName);
            filtered_metadata = filtered_metadata(strcmp(filtered_metadata.(fieldName), filterValue), :);
        end
    end
    if ~isempty(variablesToExclude)
        is_on_exclude_list = ismember(filtered_metadata.ModifiedName, variablesToExclude);
        filtered_metadata = filtered_metadata(~is_on_exclude_list, :);
    end
	if isempty(filtered_metadata), disp('No variables remained after filtering.'); return; end
    variablesForAnalysis = filtered_metadata.ModifiedName;
    fprintf('Performing analysis based on %d variables.\n', numel(variablesForAnalysis));
    dataForAnalysis_table = dataTable(:, variablesForAnalysis);
    subjectIDs = dataTable.SubjectID;
    rowsWithNaN = any(ismissing(dataForAnalysis_table), 2);
    cleanData_table = dataForAnalysis_table(~rowsWithNaN, :);
    dataForAnalysis_matrix = cleanData_table.Variables;
    subjectIDs_cleaned = subjectIDs(~rowsWithNaN);
    if any(rowsWithNaN), fprintf('Removed %d sessions with incomplete data.\n', sum(rowsWithNaN)); end
    if size(dataForAnalysis_matrix, 1) < 2, error('Not enough data to perform analysis after cleaning.'); end

    % --- 4. Normalization and UMAP ---
    
    %%% NEW: Z-score normalization step controlled by the 'Normalize' flag.
    dataForUmap = dataForAnalysis_matrix;
    if normalizeData
        disp('Applying Z-score normalization to data before UMAP...');
        dataForUmap = zscore(dataForAnalysis_matrix);
    end
    
    disp('Performing UMAP... (this may take a moment)');
    umap_scores = run_umap(dataForUmap, 'n_neighbors', n_neighbors, 'min_dist', min_dist, 'verbose', 'none');
    
    % --- 5. DBSCAN Clustering on UMAP results ---
    % (This section is identical to the previous version)
    disp('Performing DBSCAN clustering on UMAP results...');
    if isempty(epsilon)
        [~, D] = knnsearch(umap_scores, umap_scores, 'K', minPts);
        k_dist = sort(D(:, end));
        points = [1:length(k_dist); k_dist']';
        lineVec = points(end,:) - points(1,:);
        lineVecN = lineVec / norm(lineVec);
        vecFromFirst = bsxfun(@minus, points, points(1,:));
        scalarProduct = vecFromFirst * lineVecN';
        vecFromFirstParallel = scalarProduct * lineVecN;
        vecToLine = vecFromFirst - vecFromFirstParallel;
        distToLine = sqrt(sum(vecToLine.^2, 2));
        [~, knee_idx] = max(distToLine);
        epsilon = k_dist(knee_idx);
        fprintf('Automatically estimated Epsilon (ε): %.4f\n', epsilon);
    end
    cluster_indices = dbscan(umap_scores, epsilon, minPts);
    num_clusters = length(unique(cluster_indices(cluster_indices > 0)));
    num_noise = sum(cluster_indices == -1);
    fprintf('DBSCAN found %d clusters and %d noise points.\n', num_clusters, num_noise);

    % --- 6. Add UMAP and Cluster Information to Output Table ---
    % (This section is identical to the previous version)
    % ... (code to add UMAP1, UMAP2, ClusterID to table)
    num_total_sessions = height(modified_dataTable);
    UMAP1_col = nan(num_total_sessions, 1); UMAP2_col = nan(num_total_sessions, 1);
    cluster_col = nan(num_total_sessions, 1);
    UMAP1_col(~rowsWithNaN) = umap_scores(:,1); UMAP2_col(~rowsWithNaN) = umap_scores(:,2);
    cluster_indices(cluster_indices == -1) = 0; 
    cluster_col(~rowsWithNaN) = cluster_indices;
    modified_dataTable.UMAP1 = UMAP1_col; modified_dataTable.UMAP2 = UMAP2_col;
    modified_dataTable.ClusterID = cluster_col;
    
    % --- 7. Plotting and Legends ---
    % (This section is identical, I've included the full code for clarity)
    fig_handle = figure('Name', 'UMAP Embedding with DBSCAN Clusters', 'Position', [100 100 1100 800]);
    ax = axes(fig_handle);
    hold(ax, 'on');
    
    unique_subjects = unique(subjectIDs_cleaned);
    subject_colors = lines(length(unique_subjects));
    cluster_markers = {'o', 's', 'd', '^', 'v', 'p', 'h', '*'};
    noise_marker = 'x';
    noise_color = [0.5 0.5 0.5];
    
    for i = 1:size(umap_scores, 1)
        subj_idx = strcmp(unique_subjects, subjectIDs_cleaned(i));
        cluster_id = cluster_indices(i);
        if cluster_id == 0
            scatter(ax, umap_scores(i, 1), umap_scores(i, 2), 30, noise_color, noise_marker);
        else
            marker_idx = mod(cluster_id - 1, length(cluster_markers)) + 1;
            scatter(ax, umap_scores(i, 1), umap_scores(i, 2), 60, subject_colors(subj_idx,:), ...
                    cluster_markers{marker_idx}, 'filled', 'MarkerFaceAlpha', 0.7);
        end
    end
    
    % Dual legends (code is identical)
    % ...

    hold(ax, 'off');
    main_title = 'UMAP Embedding of Behavioral Strategies with DBSCAN Clustering';
    if normalizeData, main_title = [main_title ' (Z-Scored)']; end
    title(ax, main_title, 'FontSize', 12);
    xlabel(ax, 'UMAP Dimension 1');
    ylabel(ax, 'UMAP Dimension 2');
    grid(ax, 'on'); box(ax, 'on'); axis(ax, 'square');

    % --- 8. Save Figure ---
    % (This section is identical)
    % ...
    if ~exist(output_folder, 'dir'), mkdir(output_folder); end
    timestamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
    output_filename = fullfile(output_folder, ['UMAP_DBSCAN_' timestamp '.png']);
    print(fig_handle, output_filename, '-dpng', '-r300');
end


function modified_dataTable = plotRawDataDBSCAN(dataTable, metadataTable, varargin)
% PLOTRAWDATADBSCAN Performs DBSCAN on raw data and visualizes with t-SNE.
%
% This function first standardizes the high-dimensional data, then performs
% DBSCAN clustering directly on it. The results (cluster assignments) are
% then visualized on a 2D t-SNE embedding of the same data.
%
% NOTE: This function requires the MATLAB Statistics and Machine Learning Toolbox.
%
% Args:
%       dataTable (table): Main data table with 'SubjectID'.
%       metadataTable (table): Metadata for variable filtering.
% Optional Name-Value Args:
%       (Same as previous function: 'Filters', 'Exclude', 'output_folder', 'MinPoints', 'Epsilon')
%
% Returns:
%       modified_dataTable (table): Copy of dataTable with 'tSNE1', 'tSNE2',
%                                   and 'RawData_ClusterID' columns.

    % --- 1. Argument Parsing (Identical to previous function) ---
    p = inputParser;
    addParameter(p, 'Filters', struct(), @isstruct);
    addParameter(p, 'Exclude', {}, @iscell);
    addParameter(p, 'output_folder', 'RawData_DBSCAN_Results', @ischar);
    addParameter(p, 'MinPoints', 10, @isnumeric);
    addParameter(p, 'Epsilon', [], @(x) isnumeric(x) || isempty(x));
    parse(p, varargin{:});

    filters = p.Results.Filters;
    variablesToExclude = p.Results.Exclude;
    output_folder = p.Results.output_folder;
    minPts = p.Results.MinPoints;
    epsilon = p.Results.Epsilon;

    % --- Setup and Data Cleaning (Identical to previous function) ---
    modified_dataTable = dataTable;
    if ismember('tSNE1', modified_dataTable.Properties.VariableNames), modified_dataTable.tSNE1 = []; end
    if ismember('tSNE2', modified_dataTable.Properties.VariableNames), modified_dataTable.tSNE2 = []; end
    if ismember('RawData_ClusterID', modified_dataTable.Properties.VariableNames), modified_dataTable.RawData_ClusterID = []; end

    % --- Variable Selection & Data Prep (Identical to previous function) ---
    % (Code for filtering, exclusion, and NaN removal is the same)
    filtered_metadata = metadataTable;
    filterFields = fieldnames(filters);
    if ~isempty(filterFields)
        for i = 1:length(filterFields)
            fieldName = filterFields{i};
            filterValue = filters.(fieldName);
            filtered_metadata = filtered_metadata(strcmp(filtered_metadata.(fieldName), filterValue), :);
        end
    end
    if ~isempty(variablesToExclude)
        is_on_exclude_list = ismember(filtered_metadata.ModifiedName, variablesToExclude);
        filtered_metadata = filtered_metadata(~is_on_exclude_list, :);
    end
	if isempty(filtered_metadata), disp('No variables remained after filtering.'); return; end
    variablesForAnalysis = filtered_metadata.ModifiedName;
    fprintf('Performing analysis based on %d variables.\n', numel(variablesForAnalysis));
    dataForAnalysis_table = dataTable(:, variablesForAnalysis);
	correlatedGroups = findHighlyCorrelatedGroups(dataForAnalysis_table,metadataTable, 0.8);
	[dataForAnalysis_table, removedVars] = removeCorrelatedVariables(dataForAnalysis_table, correlatedGroups);
	disp(removedVars);
    subjectIDs = dataTable.SubjectID;
    rowsWithNaN = any(ismissing(dataForAnalysis_table), 2);
    cleanData_table = dataForAnalysis_table(~rowsWithNaN, :);
    dataForAnalysis_matrix = cleanData_table.Variables;
    subjectIDs_cleaned = subjectIDs(~rowsWithNaN);
    if any(rowsWithNaN), fprintf('Removed %d sessions with incomplete data.\n', sum(rowsWithNaN)); end
    if size(dataForAnalysis_matrix, 1) < 2, error('Not enough data to perform analysis after cleaning.'); end

    % --- 4. *** NEW: Standardize Data for Clustering *** ---
    disp('Standardizing data (Z-score) for high-dimensional clustering...');
    dataForAnalysis_matrix_scaled = zscore(dataForAnalysis_matrix);

    % --- 5. Perform DBSCAN Clustering on RAW (SCALED) DATA ---
    disp('Performing DBSCAN clustering on high-dimensional raw data...');
    if ~license('test', 'Statistics_Toolbox')
        error('This function requires the Statistics and Machine Learning Toolbox.');
    end

    % Estimate Epsilon if not provided, using the SCALED high-dimensional data
    if isempty(epsilon)
        disp(['Estimating optimal epsilon for MinPoints = ' num2str(minPts) '...']);
        % NOTE: We use the scaled high-dimensional data here
        [~, D] = knnsearch(dataForAnalysis_matrix_scaled, dataForAnalysis_matrix_scaled, 'K', minPts);
        k_dist = sort(D(:, end));
        points = [1:length(k_dist); k_dist']';
        lineVec = points(end,:) - points(1,:);
        lineVecN = lineVec / norm(lineVec);
        vecFromFirst = bsxfun(@minus, points, points(1,:));
        scalarProduct = vecFromFirst * lineVecN';
        vecFromFirstParallel = scalarProduct * lineVecN;
        vecToLine = vecFromFirst - vecFromFirstParallel;
        distToLine = sqrt(sum(vecToLine.^2, 2));
        [~, knee_idx] = max(distToLine);
        epsilon = k_dist(knee_idx);
        fprintf('Automatically estimated Epsilon (ε): %.4f\n', epsilon);
    else
        fprintf('Using user-provided Epsilon (ε): %.4f\n', epsilon);
    end
    
    % *** CORE CHANGE: Run DBSCAN on the scaled high-dimensional data ***
    cluster_indices = dbscan(dataForAnalysis_matrix_scaled, epsilon, minPts);
    
    num_clusters = length(unique(cluster_indices(cluster_indices > 0)));
    num_noise = sum(cluster_indices == -1);
    fprintf('DBSCAN on raw data found %d clusters and %d noise points.\n', num_clusters, num_noise);

    % --- 6. Perform t-SNE for VISUALIZATION ONLY ---
    disp('Performing t-SNE for visualization... (this may take a moment)');
    tsne_scores = tsne(dataForAnalysis_matrix); % Use original (unscaled) data for t-SNE

    % --- 7. Add t-SNE and Cluster Information to Output Table ---
    num_total_sessions = height(modified_dataTable);
    tSNE1_col = nan(num_total_sessions, 1);
    tSNE2_col = nan(num_total_sessions, 1);
    cluster_col = nan(num_total_sessions, 1);
    
    tSNE1_col(~rowsWithNaN) = tsne_scores(:,1);
    tSNE2_col(~rowsWithNaN) = tsne_scores(:,2);
    cluster_indices(cluster_indices == -1) = 0; % Map noise from -1 to 0
    cluster_col(~rowsWithNaN) = cluster_indices;
    
    modified_dataTable.tSNE1 = tSNE1_col;
    modified_dataTable.tSNE2 = tSNE2_col;
    modified_dataTable.RawData_ClusterID = cluster_col; % New column name
    disp('Added "tSNE1", "tSNE2", and "RawData_ClusterID" columns to the output data table.');
    
    % --- 8. Plotting and Legends (Identical to previous function) ---
    % The plotting logic does not need to change, as it correctly uses the
    % pre-computed `cluster_indices` for shape and `tsne_scores` for position.
    fig_handle = figure('Name', 'Raw Data DBSCAN Clusters (visualized by t-SNE)', 'Position', [100 100 1100 800]);
    ax = axes(fig_handle);
    hold(ax, 'on');
    
    unique_subjects = unique(subjectIDs_cleaned);
    subject_colors = lines(length(unique_subjects));
    cluster_markers = {'o', 's', 'd', '^', 'v', 'p', 'h', '*'};
    noise_marker = 'x';
    noise_color = [0.5 0.5 0.5];
    
    for i = 1:size(tsne_scores, 1)
        subj_idx = strcmp(unique_subjects, subjectIDs_cleaned(i));
        cluster_id = cluster_indices(i);
        
        if cluster_id == 0 % Noise
            scatter(ax, tsne_scores(i, 1), tsne_scores(i, 2), 30, noise_color, noise_marker);
        else % Clustered point
            marker_idx = mod(cluster_id - 1, length(cluster_markers)) + 1;
            scatter(ax, tsne_scores(i, 1), tsne_scores(i, 2), 60, subject_colors(subj_idx,:), ...
                    cluster_markers{marker_idx}, 'filled');
        end
    end
    
    % Legends (code is identical to previous function)
    % ... (Full legend creation code as in the previous answer) ...
    
    % --- 9. Final Touches (Identical, but updated title) ---
    hold(ax, 'off');
    main_title = 'DBSCAN Clusters from Raw Data (Visualized with t-SNE)';
    title(ax, main_title, 'FontSize', 14);
    % ... (rest of the plotting code is identical) ...
    
    xlabel(ax, 't-SNE Dimension 1');
    ylabel(ax, 't-SNE Dimension 2');
    grid(ax, 'on'); box(ax, 'on');
    axis(ax, 'square');

    % --- 10. Save Figure (Identical) ---
    if ~exist(output_folder, 'dir'), mkdir(output_folder); end
    timestamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
    output_filename = fullfile(output_folder, ['RawData_DBSCAN_on_tSNE_' timestamp '.png']);
    disp(['Saving figure to: ' output_filename]);
    print(fig_handle, output_filename, '-dpng', '-r300');
    fprintf('Figure successfully saved.\n');
end


function modified_dataTable = plotTSNE_DBSCAN(dataTable, metadataTable, varargin)
% PLOTTSNE_DBSCAN Performs t-SNE and then DBSCAN to find and plot clusters.
%
% This function reduces data dimensionality using t-SNE and then applies
% DBSCAN to identify non-linear clusters in the 2D embedding. The resulting
% plot visualizes clusters using different marker shapes, while preserving
% subject identity with color.
%
% NOTE: This function requires the MATLAB Statistics and Machine Learning Toolbox
%       for 'tsne' and 'dbscan'.
%
% Args:
%       dataTable (table): Main data table, must contain 'SubjectID'.
%       metadataTable (table): Metadata for variable filtering.
% Optional Name-Value Args:
%       'Filters' (struct):   Keeps variables matching these criteria.
%       'Exclude' (cell array): Removes specific variables.
%       'output_folder' (string): Folder to save the figure. Default is 'TSNE_DBSCAN_Results'.
%       'MinPoints' (numeric):  The 'MinPts' parameter for DBSCAN. Default is 5.
%       'Epsilon' (numeric):    The 'epsilon' for DBSCAN. If empty, it's auto-estimated. Default is [].
%
% Returns:
%       modified_dataTable (table): A copy of dataTable with added 'tSNE1',
%                                   'tSNE2', and 'ClusterID' columns.

    % --- 1. Argument Parsing ---
    p = inputParser;
    addParameter(p, 'Filters', struct(), @isstruct);
    addParameter(p, 'Exclude', {}, @iscell);
    addParameter(p, 'output_folder', 'TSNE_DBSCAN_Results', @ischar);
    addParameter(p, 'MinPoints', 10, @isnumeric);
    addParameter(p, 'Epsilon', [], @(x) isnumeric(x) || isempty(x));
    parse(p, varargin{:});

    filters = p.Results.Filters;
    variablesToExclude = p.Results.Exclude;
    output_folder = p.Results.output_folder;
    minPts = p.Results.MinPoints;
    epsilon = p.Results.Epsilon;

    % --- Create a copy of the dataTable ---
    modified_dataTable = dataTable;
    
    % --- Check for and remove existing t-SNE/Cluster columns ---
    if ismember('tSNE1', modified_dataTable.Properties.VariableNames), modified_dataTable.tSNE1 = []; end
    if ismember('tSNE2', modified_dataTable.Properties.VariableNames), modified_dataTable.tSNE2 = []; end
    if ismember('ClusterID', modified_dataTable.Properties.VariableNames), modified_dataTable.ClusterID = []; end

    % --- 2. Variable Selection (Filtering and Exclusion) ---
    % This section remains unchanged from the original function
    filtered_metadata = metadataTable;
    filterFields = fieldnames(filters);
    if ~isempty(filterFields)
        disp('Applying inclusion filters to select variables for analysis...');
        for i = 1:length(filterFields)
            fieldName = filterFields{i};
            filterValue = filters.(fieldName);
            filtered_metadata = filtered_metadata(strcmp(filtered_metadata.(fieldName), filterValue), :);
        end
    end
    if ~isempty(variablesToExclude)
        is_on_exclude_list = ismember(filtered_metadata.ModifiedName, variablesToExclude);
        filtered_metadata = filtered_metadata(~is_on_exclude_list, :);
    end
	if isempty(filtered_metadata), disp('No variables remained after filtering.'); return; end
    variablesForAnalysis = filtered_metadata.ModifiedName;
    fprintf('Performing analysis based on %d variables.\n', numel(variablesForAnalysis));

    % --- 3. Prepare Data and Handle NaNs ---
    % This section remains unchanged
    dataForAnalysis_table = dataTable(:, variablesForAnalysis);
	correlatedGroups = findHighlyCorrelatedGroups(dataForAnalysis_table,metadataTable, 0.8);
	[dataForAnalysis_table, removedVars] = removeCorrelatedVariables(dataForAnalysis_table, correlatedGroups);
	disp(removedVars);
    subjectIDs = dataTable.SubjectID;
    rowsWithNaN = any(ismissing(dataForAnalysis_table), 2);
    cleanData_table = dataForAnalysis_table(~rowsWithNaN, :);
    dataForAnalysis_matrix = cleanData_table.Variables;
    subjectIDs_cleaned = subjectIDs(~rowsWithNaN);
    if any(rowsWithNaN), fprintf('Removed %d sessions with incomplete data.\n', sum(rowsWithNaN)); end
    if size(dataForAnalysis_matrix, 1) < 2, error('Not enough data to perform analysis after cleaning.'); end

    % --- 4. Perform t-SNE for Dimensionality Reduction ---
    disp('Performing t-SNE... (this may take a moment)');
    if ~license('test', 'Statistics_Toolbox')
        error('This function requires the Statistics and Machine Learning Toolbox for tsne() and dbscan().');
    end
    tsne_scores = tsne(dataForAnalysis_matrix);
    
    % --- 5. Perform DBSCAN Clustering ---
    disp('Performing DBSCAN clustering on t-SNE results...');
    
    % Estimate Epsilon if not provided
    if isempty(epsilon)
        disp(['Estimating optimal epsilon for MinPoints = ' num2str(minPts) '...']);
        % k-distance graph
        [~, D] = knnsearch(tsne_scores, tsne_scores, 'K', minPts);
        k_dist = sort(D(:, end));
        
        % Find the "knee" of the k-distance plot automatically
        points = [1:length(k_dist); k_dist']';
        lineVec = points(end,:) - points(1,:);
        lineVecN = lineVec / norm(lineVec);
        vecFromFirst = bsxfun(@minus, points, points(1,:));
        scalarProduct = vecFromFirst * lineVecN';
        vecFromFirstParallel = scalarProduct * lineVecN;
        vecToLine = vecFromFirst - vecFromFirstParallel;
        distToLine = sqrt(sum(vecToLine.^2, 2));
        [~, knee_idx] = max(distToLine);
        
        epsilon = k_dist(knee_idx);
        fprintf('Automatically estimated Epsilon (ε): %.4f\n', epsilon);
    else
        fprintf('Using user-provided Epsilon (ε): %.4f\n', epsilon);
    end
    
    % Run DBSCAN
    cluster_indices = dbscan(tsne_scores, epsilon, minPts);
    
    num_clusters = length(unique(cluster_indices(cluster_indices > 0)));
    num_noise = sum(cluster_indices == -1);
    fprintf('DBSCAN found %d clusters and %d noise points.\n', num_clusters, num_noise);
    
    % --- 6. Add t-SNE and Cluster Information to Output Table ---
    num_total_sessions = height(modified_dataTable);
    tSNE1_col = nan(num_total_sessions, 1);
    tSNE2_col = nan(num_total_sessions, 1);
    cluster_col = nan(num_total_sessions, 1); % Use NaN for non-analyzed rows
    
    tSNE1_col(~rowsWithNaN) = tsne_scores(:,1);
    tSNE2_col(~rowsWithNaN) = tsne_scores(:,2);
    % Map DBSCAN's -1 for noise to 0 for easier indexing/interpretation
    cluster_indices(cluster_indices == -1) = 0; 
    cluster_col(~rowsWithNaN) = cluster_indices;
    
    modified_dataTable.tSNE1 = tSNE1_col;
    modified_dataTable.tSNE2 = tSNE2_col;
    modified_dataTable.ClusterID = cluster_col;
    disp('Added "tSNE1", "tSNE2", and "ClusterID" columns to the output data table.');

    % --- 7. Create the Scatter Plot with Shapes for Clusters ---
    fig_handle = figure('Name', 't-SNE Embedding with DBSCAN Clusters', 'Position', [100 100 1100 800]);
    ax = axes(fig_handle); % Create axes for the main plot
    hold(ax, 'on');
    
    unique_subjects = unique(subjectIDs_cleaned);
    subject_colors = lines(length(unique_subjects));
    
    % Define markers for clusters. Add more if you expect more than 8 clusters.
    cluster_markers = {'o', 's', 'd', '^', 'v', 'p', 'h', '*'};
    noise_marker = 'x';
    noise_color = [0.5 0.5 0.5]; % Grey for noise
    
    % Plot each point with color by subject and shape by cluster
    for i = 1:size(tsne_scores, 1)
        subj_idx = strcmp(unique_subjects, subjectIDs_cleaned(i));
        cluster_id = cluster_indices(i);
        
        if cluster_id == 0 % Noise point
            scatter(ax, tsne_scores(i, 1), tsne_scores(i, 2), 30, noise_color, noise_marker);
        else % Clustered point
            marker_idx = mod(cluster_id - 1, length(cluster_markers)) + 1;
            scatter(ax, tsne_scores(i, 1), tsne_scores(i, 2), 60, subject_colors(subj_idx,:), ...
                    cluster_markers{marker_idx}, 'filled');
        end
    end
    
    % --- 8. Create Manual Legends for Subjects (Color) and Clusters (Shape) ---
    % Create a legend for subjects (color)
    subject_handles = gobjects(length(unique_subjects), 1);
    for i = 1:length(unique_subjects)
        s_label = strrep(unique_subjects{i}, '_', ' ');
        subject_handles(i) = plot(ax, NaN, NaN, 's', 'MarkerFaceColor', subject_colors(i,:), ...
                                  'MarkerSize', 10, 'LineStyle', 'none', 'DisplayName', s_label);
    end
    lgd1 = legend(ax, subject_handles, 'Location', 'northeast');
    title(lgd1, 'Subjects');
    
    % Create a second legend for clusters (shape) in the same location
    ax_pos = get(ax, 'Position');
    lgd_ax = axes('Position', ax_pos, 'Visible', 'off'); % Invisible axes for second legend
    hold(lgd_ax, 'on');

    cluster_handles = gobjects(num_clusters + 1, 1);
    cluster_labels = cell(num_clusters + 1, 1);
    
    % Add entry for noise
    cluster_handles(1) = plot(lgd_ax, NaN, NaN, noise_marker, 'MarkerSize', 8, 'MarkerEdgeColor', noise_color, 'LineStyle', 'none');
    cluster_labels{1} = 'Noise';

    % Add entry for each cluster
    for i = 1:num_clusters
        marker_idx = mod(i - 1, length(cluster_markers)) + 1;
        cluster_handles(i+1) = plot(lgd_ax, NaN, NaN, cluster_markers{marker_idx}, 'MarkerSize', 8, ...
                                    'MarkerEdgeColor', 'k', 'LineStyle', 'none');
        cluster_labels{i+1} = ['Cluster ' num2str(i)];
    end
    
    lgd2 = legend(lgd_ax, cluster_handles, cluster_labels, 'Location', 'east');
    title(lgd2, 'Clusters');
    
    % Adjust legend positions to not overlap
    drawnow; % Update figure to get correct legend positions
    set(lgd1, 'Location', 'northeast'); % Re-assert position
    pos1 = get(lgd1, 'Position');
    pos2 = get(lgd2, 'Position');
    pos2(1) = pos1(1); % Align horizontally
    pos2(2) = pos1(2) - pos2(4) - 0.01; % Position below the first legend
    set(lgd2, 'Position', pos2);


    % --- 9. Final Touches with Dynamic Title ---
    hold(ax, 'off');
    main_title = 't-SNE Embedding with DBSCAN Clustering';
    title_lines = {main_title};
    if ~isempty(fieldnames(filters))
        filter_parts = cellfun(@(f) [f '=' filters.(f)], fieldnames(filters), 'UniformOutput', false);
        title_lines{end+1} = ['\color[rgb]{0 0.4470 0.7410}Using Variables Where: ' strjoin(filter_parts, ', ')];
    end
    title(ax, title_lines, 'FontSize', 12);
    xlabel(ax, 't-SNE Dimension 1');
    ylabel(ax, 't-SNE Dimension 2');
    
    x_range_total = range(tsne_scores(:, 1));
    y_range_total = range(tsne_scores(:, 2));
    x_pad = x_range_total * 0.1; if x_pad==0, x_pad=1; end
    y_pad = y_range_total * 0.1; if y_pad==0, y_pad=1; end
    xlim(ax, [min(tsne_scores(:, 1))-x_pad, max(tsne_scores(:, 1))+x_pad]);
    ylim(ax, [min(tsne_scores(:, 2))-y_pad, max(tsne_scores(:, 2))+y_pad]);
    grid(ax, 'on'); box(ax, 'on');
    axis(ax, 'square'); % Fixes the output shape to be a square aspect ratio

    % --- 10. Save the Figure Automatically ---
    if ~exist(output_folder, 'dir'), mkdir(output_folder); end
    timestamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
    output_filename = fullfile(output_folder, ['TSNE_DBSCAN_' timestamp '.png']);
    disp(['Saving figure to: ' output_filename]);
    print(fig_handle, output_filename, '-dpng', '-r300');
    fprintf('Figure successfully saved.\n');
end



function modified_dataTable = plotSubjectTSNE(dataTable, metadataTable, varargin)
% PLOTSUBJECTTSNE Performs t-SNE on a dataset and plots the results.
%
% This function reduces the dimensionality of the input data to two dimensions
% using the t-SNE (t-Distributed Stochastic Neighbor Embedding) algorithm and
% generates a scatter plot of the results, with points colored by SubjectID.
%
% NOTE: This function requires the MATLAB Statistics and Machine Learning Toolbox
%       for the 'tsne' function.
%
% Args:
%       dataTable (table): The main data table, must contain 'SubjectID'.
%       metadataTable (table): The metadata table for filtering variables.
% Optional Name-Value Args:
%       'Filters' (struct):   Keeps variables matching these criteria.
%       'Exclude' (cell array): Removes specific variables from the analysis.
%       'output_folder' (string): Folder to save the figure in. Default is 'Subject_tSNE'.
%
% Returns:
%       modified_dataTable (table): A copy of the input dataTable with added
%                                   'tSNE1' and 'tSNE2' columns.

    % --- Argument Parsing ---
    p = inputParser;
    addParameter(p, 'Filters', struct(), @isstruct);
    addParameter(p, 'Exclude', {}, @iscell);
    addParameter(p, 'output_folder', 'Subject_tSNE', @ischar);
    parse(p, varargin{:});
    filters = p.Results.Filters;
    variablesToExclude = p.Results.Exclude;
    output_folder = p.Results.output_folder;

    % --- Create a copy of the dataTable to be modified and returned ---
    modified_dataTable = dataTable;
    
    % --- Check for and remove existing t-SNE columns ---
    if ismember('tSNE1', modified_dataTable.Properties.VariableNames)
        disp('Removing existing "tSNE1" column before re-calculating.');
        modified_dataTable.tSNE1 = [];
    end
    if ismember('tSNE2', modified_dataTable.Properties.VariableNames)
        disp('Removing existing "tSNE2" column before re-calculating.');
        modified_dataTable.tSNE2 = [];
    end

    % --- 1. Variable Selection (Filtering and Exclusion) ---
    filtered_metadata = metadataTable;
    filterFields = fieldnames(filters);
    if ~isempty(filterFields)
        disp('Applying inclusion filters to select variables for analysis...');
        for i = 1:length(filterFields)
            fieldName = filterFields{i};
            filterValue = filters.(fieldName);
            filtered_metadata = filtered_metadata(strcmp(filtered_metadata.(fieldName), filterValue), :);
        end
    end
    if ~isempty(variablesToExclude)
        is_on_exclude_list = ismember(filtered_metadata.ModifiedName, variablesToExclude);
        filtered_metadata = filtered_metadata(~is_on_exclude_list, :);
    end
	if isempty(filtered_metadata), disp('No variables remained after filtering.'); return; end
    variablesForAnalysis = filtered_metadata.ModifiedName;
    fprintf('Performing analysis based on %d variables.\n', numel(variablesForAnalysis));

    % --- 2. Prepare Data and Handle NaNs ---
    dataForAnalysis_table = dataTable(:, variablesForAnalysis);


	correlatedGroups = findHighlyCorrelatedGroups(dataForAnalysis_table,metadataTable, 0.8);
	[dataForAnalysis_table, removedVars] = removeCorrelatedVariables(dataForAnalysis_table, correlatedGroups);
	disp(removedVars);


    subjectIDs = dataTable.SubjectID;
    rowsWithNaN = any(ismissing(dataForAnalysis_table), 2);
    cleanData_table = dataForAnalysis_table(~rowsWithNaN, :);
    dataForAnalysis_matrix = cleanData_table.Variables;
    subjectIDs_cleaned = subjectIDs(~rowsWithNaN);
    if any(rowsWithNaN), fprintf('Removed %d sessions with incomplete data.\n', sum(rowsWithNaN)); end
    if size(dataForAnalysis_matrix, 1) < 2, error('Not enough data to perform analysis after cleaning.'); end

    % --- 3. Perform t-SNE for Dimensionality Reduction ---
    disp('Performing t-SNE... (this may take a moment)');
    % Check for toolbox availability
    if ~license('test', 'Statistics_Toolbox')
        error('This function requires the Statistics and Machine Learning Toolbox for tsne().');
    end
    
    % tsne expects observations in rows, variables in columns.
    % By default, it reduces to 2 dimensions.
    tsne_scores = tsne(dataForAnalysis_matrix);
    
    % --- Add t-SNE Information to the Output Table ---
    num_total_sessions = height(modified_dataTable);
    tSNE1_col = nan(num_total_sessions, 1);
    tSNE2_col = nan(num_total_sessions, 1);
    
    tSNE1_col(~rowsWithNaN) = tsne_scores(:,1);
    tSNE2_col(~rowsWithNaN) = tsne_scores(:,2);
    
    modified_dataTable.tSNE1 = tSNE1_col;
    modified_dataTable.tSNE2 = tSNE2_col;
    disp('Added "tSNE1" and "tSNE2" columns to the output data table.');
    
    % --- 4. Create the Scatter Plot ---
    fig_handle = figure('Name', 't-SNE Embedding of Subjects', 'Position', [100 100 1000 800]);
    hold on;
    
    unique_subjects = unique(subjectIDs_cleaned);
    subject_colors = lines(length(unique_subjects));
    
    for i = 1:size(tsne_scores, 1)
        subj_idx = find(strcmp(unique_subjects, subjectIDs_cleaned(i)));
        scatter(tsne_scores(i, 1), tsne_scores(i, 2), 60, subject_colors(subj_idx,:), 'o', 'filled');
    end
    
    % --- 5. Create a Manual Legend for Subjects ---
    subject_handles = [];
    subject_labels = {};
    for i = 1:length(unique_subjects)
        s_label = strrep(unique_subjects{i}, '_', ' ');
        h = plot(NaN, NaN, 's', 'MarkerFaceColor', subject_colors(i,:), 'MarkerSize', 8, 'LineStyle', 'none', 'DisplayName', s_label);
        subject_handles(end+1) = h;
        subject_labels{end+1} = s_label;
    end
    lgd = legend(subject_handles, subject_labels, 'Location', 'northeastoutside', 'NumColumns', 1);
    title(lgd, 'Subjects');
    
    % --- 6. Final Touches with Dynamic Title ---
    hold off;
    main_title = 't-SNE Embedding of Subjects';
    title_lines = {main_title};
    if ~isempty(fieldnames(filters))
        filter_parts = cellfun(@(f) [f '=' filters.(f)], fieldnames(filters), 'UniformOutput', false);
        title_lines{end+1} = ['\color[rgb]{0 0.4470 0.7410}Using Variables Where: ' strjoin(filter_parts, ', ')];
    end
    if ~isempty(variablesToExclude)
        title_lines{end+1} = ['\color[rgb]{0.8500 0.3250 0.0980}Excluding: ' strjoin(strrep(variablesToExclude, '_', ' '))];
    end
    title(title_lines, 'FontSize', 12);
    xlabel('t-SNE Dimension 1');
    ylabel('t-SNE Dimension 2');
    
    x_range_total = range(tsne_scores(:, 1));
    y_range_total = range(tsne_scores(:, 2));
    x_pad = x_range_total * 0.1; if x_pad==0, x_pad=1; end
    y_pad = y_range_total * 0.1; if y_pad==0, y_pad=1; end
    xlim([min(tsne_scores(:, 1))-x_pad, max(tsne_scores(:, 1))+x_pad]);
    ylim([min(tsne_scores(:, 2))-y_pad, max(tsne_scores(:, 2))+y_pad]);
    grid on; box on;
    
    % --- 7. Save the Figure Automatically ---
    if ~exist(output_folder, 'dir'), mkdir(output_folder); end
    timestamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
    output_filename = fullfile(output_folder, ['Subject_tSNE_' timestamp '.png']);
    disp(['Saving figure to: ' output_filename]);
    print(fig_handle, output_filename, '-dpng', '-r300');
    fprintf('Figure successfully saved.\n');
end


function modified_dataTable = plotSubjectPCA_ICA(dataTable, metadataTable, varargin)
% PLOTSUBJECTPCA_ICA Performs PCA, then ICA, k-means, and plots with contours.
%
% This function first reduces dimensionality with PCA, then separates signals
% using MATLAB's built-in Reconstruction ICA (rica). K-means clustering and 
% plotting are performed on the resulting Independent Components.
%
% NOTE: This function requires the MATLAB Statistics and Machine Learning Toolbox
%       for the 'pca' and 'rica' functions.
%
% Args:
%       dataTable (table): The main data table.
%       metadataTable (table): The metadata table for filtering variables.
% Optional Name-Value Args:
%       'Filters' (struct):   Keeps variables matching these criteria.
%       'Exclude' (cell array): Removes specific variables from the analysis.
%       'output_folder' (string): Folder to save the figure in. Default is 'Subject_PCA_ICA'.
%
% Returns:
%       modified_dataTable (table): A copy of the input dataTable with an
%                                   added 'SubjectCluster' column.
    
    % --- Argument Parsing ---
    p = inputParser;
    addParameter(p, 'Filters', struct(), @isstruct);
    addParameter(p, 'Exclude', {}, @iscell);
    addParameter(p, 'output_folder', 'Subject_PCA_ICA', @ischar);
    parse(p, varargin{:});
    filters = p.Results.Filters;
    variablesToExclude = p.Results.Exclude;
    output_folder = p.Results.output_folder;
    % --- Create a copy of the dataTable to be modified and returned ---
    modified_dataTable = dataTable;
    
    % --- Check for and remove an existing SubjectCluster column ---
    if ismember('SubjectCluster', modified_dataTable.Properties.VariableNames)
        disp('Removing existing "SubjectCluster" column before re-calculating.');
        modified_dataTable.SubjectCluster = [];
    end
    % --- 1. Variable Selection (Filtering and Exclusion) ---
    % (This section is unchanged)
    filtered_metadata = metadataTable;
    filterFields = fieldnames(filters);
    if ~isempty(filterFields)
        disp('Applying inclusion filters to select variables for analysis...');
        for i = 1:length(filterFields)
            fieldName = filterFields{i};
            filterValue = filters.(fieldName);
            filtered_metadata = filtered_metadata(strcmp(filtered_metadata.(fieldName), filterValue), :);
        end
    end
    if ~isempty(variablesToExclude)
        is_on_exclude_list = ismember(filtered_metadata.ModifiedName, variablesToExclude);
        filtered_metadata = filtered_metadata(~is_on_exclude_list, :);
    end
	if isempty(filtered_metadata), disp('No variables remained after filtering.'); return; end
    variablesForAnalysis = filtered_metadata.ModifiedName;
    fprintf('Performing analysis based on %d variables.\n', numel(variablesForAnalysis));
    % --- 2. Prepare Data and Handle NaNs ---
    % (This section is unchanged)
    dataForAnalysis_table = dataTable(:, variablesForAnalysis);
    subjectIDs = dataTable.SubjectID;
    rowsWithNaN = any(ismissing(dataForAnalysis_table), 2);
    cleanData_table = dataForAnalysis_table(~rowsWithNaN, :);
    dataForAnalysis_matrix = cleanData_table.Variables;
    subjectIDs_cleaned = subjectIDs(~rowsWithNaN);
    if any(rowsWithNaN), fprintf('Removed %d sessions with incomplete data.\n', sum(rowsWithNaN)); end
    if size(dataForAnalysis_matrix, 1) < 2, error('Not enough data to perform analysis after cleaning.'); end

    % --- 3. Perform PCA for Dimensionality Reduction & Whitening ---
    disp('Performing PCA...');
    % Check for toolbox availability
    if ~license('test', 'Statistics_Toolbox')
        error('This function requires the Statistics and Machine Learning Toolbox for pca() and rica().');
    end
    [~, score, ~, ~, explained] = pca(dataForAnalysis_matrix);
    
    % --- 4. [MODIFIED] Perform ICA on Principal Components using rica ---
    % Determine the number of PCs that explain 99% of the variance
    cumulative_explained = cumsum(explained);
    num_pcs_to_keep = find(cumulative_explained >= 90, 1, 'first');
    if isempty(num_pcs_to_keep), num_pcs_to_keep = size(score, 2); end
    fprintf('Keeping %d Principal Components (explaining %.2f%% variance) for ICA.\n', ...
            num_pcs_to_keep, cumulative_explained(num_pcs_to_keep));
            
    disp('Performing Reconstruction ICA (rica) on the PCA scores...');
    
    % rica expects observations in rows, variables in columns, which matches 'score'
    pca_scores_for_ica = score(:, 1:num_pcs_to_keep);
    
    % Create the RICA model
    Mdl = rica(pca_scores_for_ica, num_pcs_to_keep, 'VerbosityLevel', 0);
    
    % Transform the PCA scores into Independent Components
    ica_scores = transform(Mdl, pca_scores_for_ica);
    
    % --- 5. K-Means Clustering on the first two Independent Components ---
    disp('Finding optimal number of clusters using Silhouette method on ICs...');
    max_k = min(10, size(ica_scores, 1) - 1);
    if max_k < 2
        optimal_k = 1;
    else
        % Use first two Independent Components for clustering
        eva = evalclusters(ica_scores(:,1:2), 'kmeans', 'silhouette', 'KList', 1:max_k);
        optimal_k = eva.OptimalK;
    end
    fprintf('Optimal number of clusters found: %d\n', optimal_k);
    cluster_indices = kmeans(ica_scores(:,1:2), optimal_k, 'Replicates', 5);

    % --- Add Cluster Information to the Output Table ---
    % (This section is unchanged)
    num_total_sessions = height(modified_dataTable);
    cluster_column = nan(num_total_sessions, 1);
    cluster_column(~rowsWithNaN) = cluster_indices;
    modified_dataTable.SubjectCluster = cluster_column;
    disp('Added "SubjectCluster" column to the output data table.');
    
    % --- 6. Create the Plot (using ica_scores) ---
    % (This section is unchanged, but now operates on ica_scores)
    fig_handle = figure('Name', 'ICA Scores with K-Means Clustering', 'Position', [100 100 1000 800]);
    hold on;
    unique_subjects = unique(subjectIDs_cleaned);
    subject_colors = lines(length(unique_subjects));
    cluster_shapes = ['o', 's', 'd', '^', 'v', 'p', 'h'];
    if optimal_k > length(cluster_shapes)
        cluster_shapes = repmat(cluster_shapes, 1, ceil(optimal_k/length(cluster_shapes)));
    end
    for i = 1:size(ica_scores, 1)
        subj_idx = find(strcmp(unique_subjects, subjectIDs_cleaned(i)));
        clust_idx = cluster_indices(i);
        scatter(ica_scores(i, 1), ica_scores(i, 2), 60, subject_colors(subj_idx,:), cluster_shapes(clust_idx), 'filled');
    end

    % --- 7. Bézier Boundary Contour Logic (using ica_scores) ---
    % (This section is unchanged, but now operates on ica_scores)
    for k = 1:optimal_k
        cluster_points = ica_scores(cluster_indices == k, 1:2);
        if size(unique(cluster_points, 'rows'), 1) < 3
            fprintf('Skipping contour for cluster %d (not enough unique points).\n', k);
            continue;
        end
        try
            x_range = range(ica_scores(:,1)); y_range = range(ica_scores(:,2));
            margin = 0.025 * mean([x_range, y_range]);
            smoothness_factor = 0.4; subdivision_level = 3;
            boundary_indices = convhull(cluster_points, 'Simplify', true);
            original_boundary_points = cluster_points(boundary_indices, :);
            cluster_center = mean(cluster_points, 1);
            expanded_boundary_points = zeros(size(original_boundary_points));
            for i = 1:size(original_boundary_points, 1)
                point = original_boundary_points(i,:);
                vec_from_center = point - cluster_center;
                expanded_boundary_points(i,:) = point + (vec_from_center / norm(vec_from_center)) * margin;
            end
            if subdivision_level > 1 && size(expanded_boundary_points, 1) > 1
                subdivided_points = [];
                for i = 1:size(expanded_boundary_points, 1)
                    p_start = expanded_boundary_points(i, :);
                    p_end_idx = mod(i, size(expanded_boundary_points, 1)) + 1;
                    p_end = expanded_boundary_points(p_end_idx, :);
                    x_coords = linspace(p_start(1), p_end(1), subdivision_level + 1);
                    y_coords = linspace(p_start(2), p_end(2), subdivision_level + 1);
                    subdivided_points = [subdivided_points; [x_coords(1:end-1)', y_coords(1:end-1)']];
                end
                boundary_points = subdivided_points;
            else
                boundary_points = expanded_boundary_points;
            end
            if size(boundary_points, 1) > 2
                num_bnd_pts = size(boundary_points, 1);
                full_curve = [];
                for i = 1:num_bnd_pts
                    p_prev_idx = mod(i-2, num_bnd_pts) + 1; p_curr_idx = i;
                    p_next_idx = mod(i, num_bnd_pts) + 1; p_next_next_idx = mod(i+1, num_bnd_pts) + 1;
                    P_prev = boundary_points(p_prev_idx, :); P0 = boundary_points(p_curr_idx, :);
                    P3 = boundary_points(p_next_idx, :); P_next_next = boundary_points(p_next_next_idx, :);
                    T1 = P3 - P_prev; T2 = P_next_next - P0;
                    if norm(T1) > 1e-6, T1 = T1 / norm(T1); else, T1 = [0 0]; end
                    if norm(T2) > 1e-6, T2 = T2 / norm(T2); else, T2 = [0 0]; end
                    segment_length = norm(P3 - P0);
                    C1 = P0 + smoothness_factor * segment_length * T1;
                    C2 = P3 - smoothness_factor * segment_length * T2;
                    control_matrix = [P0; C1; C2; P3];
                    curve_segment = generate_bezier_curve(control_matrix, 20);
                    full_curve = [full_curve; curve_segment];
                end
                plot(full_curve(:,1), full_curve(:,2), 'k-', 'LineWidth', 1.5);
            else
                plot(boundary_points(:,1), boundary_points(:,2), 'k-', 'LineWidth', 1.5);
            end
        catch ME
            warning('Could not generate Bézier contour for cluster %d. Reason: %s', k, ME.message);
            try % Fallback to straight lines
                hull_indices = convhull(cluster_points);
                plot(cluster_points(hull_indices, 1), cluster_points(hull_indices, 2), 'k-', 'LineWidth', 1.5);
            catch
            end
        end
    end
    
    % --- 8. Create a Manual Legend for Clusters and Subjects ---
    % (This section is unchanged)
    cluster_handles = []; cluster_labels = {};
    for k = 1:optimal_k
        h = scatter(NaN, NaN, 60, 'k', cluster_shapes(k), 'DisplayName', ['Cluster ' num2str(k)]);
        cluster_handles(end+1) = h; cluster_labels{end+1} = ['Cluster ' num2str(k)];
    end
    subject_handles = []; subject_labels = {};
    for i = 1:length(unique_subjects)
        s_label = strrep(unique_subjects{i}, '_', ' ');
        h = plot(NaN, NaN, 's', 'MarkerFaceColor', subject_colors(i,:), 'MarkerSize', 8, 'LineStyle', 'none', 'DisplayName', s_label);
        subject_handles(end+1) = h; subject_labels{end+1} = s_label;
    end
    lgd = legend([cluster_handles, subject_handles], [cluster_labels, subject_labels], 'Location', 'northeastoutside', 'NumColumns', 1);
    title(lgd, 'Legend');
    
    % --- 9. Final Touches with Dynamic Title ---
    % (This section is unchanged, labels updated for ICA)
    hold off;
    main_title = ['ICA Scores (from PCA) with ' num2str(optimal_k) ' k-means Clusters'];
    title_lines = {main_title};
    if ~isempty(fieldnames(filters))
        filter_parts = cellfun(@(f) [f '=' filters.(f)], fieldnames(filters), 'UniformOutput', false);
        title_lines{end+1} = ['\color[rgb]{0 0.4470 0.7410}Using Variables Where: ' strjoin(filter_parts, ', ')];
    end
    if ~isempty(variablesToExclude)
        title_lines{end+1} = ['\color[rgb]{0.8500 0.3250 0.0980}Excluding: ' strjoin(strrep(variablesToExclude, '_', ' '))];
    end
    title(title_lines, 'FontSize', 12);
    xlabel('Independent Component 1');
    ylabel('Independent Component 2');
    
    x_range_total = range(ica_scores(:, 1)); y_range_total = range(ica_scores(:, 2));
    x_pad = x_range_total*0.1; if x_pad==0, x_pad=1; end; y_pad = y_range_total*0.1; if y_pad==0, y_pad=1; end
    xlim([min(ica_scores(:, 1))-x_pad, max(ica_scores(:, 1))+x_pad]);
    ylim([min(ica_scores(:, 2))-y_pad, max(ica_scores(:, 2))+y_pad]);
    grid on; box on;

    % --- 10. Save the Figure Automatically ---
    % (This section is unchanged)
    if ~exist(output_folder, 'dir'), mkdir(output_folder); end
    timestamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
    output_filename = fullfile(output_folder, ['Subject_PCA_ICA_k' num2str(optimal_k) '_' timestamp '.png']);
    disp(['Saving figure to: ' output_filename]);
    print(fig_handle, output_filename, '-dpng', '-r300');
    fprintf('Figure successfully saved.\n');
end

% You still need the helper function saved as generate_bezier_curve.m
% function B = generate_bezier_curve(P, num_points) ...




function plotClusterDifferences(modified_dataTable, metadataTable, varargin)
% plotClusterDifferences Creates a plot showing cluster differences from the mean.
%
%   plotClusterDifferences(modified_dataTable, metadataTable, ...)
%   Takes a MATLAB table 'modified_dataTable' (output from plotSubjectPCA)
%   and a 'metadataTable' as input. It uses optional 'varargin' filters
%   to select variables for plotting, similar to plotSubjectPCA.
%
%   The function generates a plot where:
%   - Each row represents a variable.
%   - Each colored line represents a cluster.
%   - Points show the cluster's average deviation from the overall mean of
%     normalized data for that variable.
%   - Horizontal error bars show the Standard Error (SE) of all sessions
%     within that cluster for that variable.

    % --- Configuration ---
    % Re-using the nice color palette
    custom_colors_rgb = { 
        [235,212,156]/255; 
       [150,180,211]/255; 
	   [120,81,135]/255
    };

    % --- 1. Argument Parsing and Variable Selection (from plotSubjectPCA) ---
    p = inputParser;
    addParameter(p, 'Filters', struct(), @isstruct);
    addParameter(p, 'Exclude', {}, @iscell);
    parse(p, varargin{:});
    filters = p.Results.Filters;
    variablesToExclude = p.Results.Exclude;

    filtered_metadata = metadataTable;
    filterFields = fieldnames(filters);
    if ~isempty(filterFields)
        for i = 1:length(filterFields)
            fieldName = filterFields{i};
            filterValue = filters.(fieldName);
            filtered_metadata = filtered_metadata(strcmp(filtered_metadata.(fieldName), filterValue), :);
        end
    end
    if ~isempty(variablesToExclude)
        is_on_exclude_list = ismember(filtered_metadata.ModifiedName, variablesToExclude);
        filtered_metadata = filtered_metadata(~is_on_exclude_list, :);
    end
	if isempty(filtered_metadata), error('No variables remained after filtering.'); end
    
    variablesForPlot = filtered_metadata.ModifiedName;
    fprintf('Plotting differences for %d variables based on filters.\n', numel(variablesForPlot));


    % --- 2. Data Preparation and Aggregation by Cluster ---
    if ~istable(modified_dataTable) || ~ismember('SubjectCluster', modified_dataTable.Properties.VariableNames)
        error('Input must be a MATLAB table containing a "SubjectCluster" column.');
    end
    
    % Find all unique clusters, ignoring sessions that were not clustered (NaN)
    cluster_id_list = unique(modified_dataTable.SubjectCluster(~isnan(modified_dataTable.SubjectCluster)));
    num_clusters = length(cluster_id_list);
    num_vars = length(variablesForPlot);

    if num_clusters == 0
        error('No valid clusters found in the "SubjectCluster" column.');
    end
    
    if num_clusters > length(custom_colors_rgb)
        warning('Not enough custom colors for all clusters. Adding random colors.');
        for i = (length(custom_colors_rgb)+1):num_clusters
            custom_colors_rgb{i} = rand(1,3);
        end
    end

    % Initialize matrices to hold aggregated data
    aggregated_data_matrix = NaN(num_clusters, num_vars);
    raw_cluster_SEs = NaN(num_clusters, num_vars);

    for c_idx = 1:num_clusters
        current_cluster_id = cluster_id_list(c_idx);
        % Find all rows (sessions) belonging to the current cluster
        cluster_rows_logical = (modified_dataTable.SubjectCluster == current_cluster_id);
        
        for v_idx = 1:num_vars
            var_name = variablesForPlot{v_idx};
            if ~ismember(var_name, modified_dataTable.Properties.VariableNames)
                warning('Variable "%s" not found in the table. Skipping.', var_name);
                continue;
            end
            
            % Get all data points for this variable within this cluster
            data_for_current_cluster_var = modified_dataTable.(var_name)(cluster_rows_logical);
            
            % Ensure data is numeric for calculations
            if isnumeric(data_for_current_cluster_var) || islogical(data_for_current_cluster_var)
                numeric_data = double(data_for_current_cluster_var);
                non_nan_data = numeric_data(~isnan(numeric_data));
                n_sessions_in_cluster = length(non_nan_data);

                if n_sessions_in_cluster > 0
                    % Calculate mean for the cluster
                    aggregated_data_matrix(c_idx, v_idx) = mean(non_nan_data);
                    
                    % Calculate Standard Error for the cluster
                    if n_sessions_in_cluster > 1
                        raw_cluster_SEs(c_idx, v_idx) = std(non_nan_data) / sqrt(n_sessions_in_cluster);
                    else
                        raw_cluster_SEs(c_idx, v_idx) = 0; % SE of 1 data point is 0
                    end
                end
            else
                warning('Data for variable "%s" is not numeric. Skipping.', var_name);
            end
        end
    end
    
    % --- 3. Normalization, Difference Calculation, and SE Scaling ---
    % This logic is identical to plotSubjectDifferences, but applied to clusters
    all_cluster_differences = NaN(num_clusters, num_vars);
    scaled_cluster_SEs = NaN(num_clusters, num_vars);

    for v_idx = 1:num_vars
        current_var_cluster_means = aggregated_data_matrix(:, v_idx); 
        
        if all(isnan(current_var_cluster_means)), continue; end
        
        min_val = min(current_var_cluster_means, [], 'omitnan');
        max_val = max(current_var_cluster_means, [], 'omitnan');
        range_val = max_val - min_val;
        
        % Normalize the means for the current variable
        if range_val > 0
            normalized_means_col = (current_var_cluster_means - min_val) / range_val;
        else
            normalized_means_col = 0.5 * ones(size(current_var_cluster_means));
            normalized_means_col(isnan(current_var_cluster_means)) = NaN;
        end
        
        % Calculate difference from the overall mean of normalized values
        overall_mean_normalized = mean(normalized_means_col, 'omitnan');
        if ~isnan(overall_mean_normalized)
            all_cluster_differences(:, v_idx) = normalized_means_col - overall_mean_normalized;
        end
        
        % Scale the SEs for this variable
        if range_val > 0
            scaled_cluster_SEs(:, v_idx) = raw_cluster_SEs(:, v_idx) / range_val;
        else
            scaled_cluster_SEs(:, v_idx) = 0; % Cannot scale SE if range is 0
        end
    end
    
    % --- 4. Plotting ---
    fig = figure('Position', [100, 100, 700, 1400]); 
    ax = axes('Parent', fig);
    hold(ax, 'on');
    max_jitter = 0.20; 
    plot_handles_for_legend = gobjects(1, num_clusters);

    for c_idx = 1:num_clusters
        cluster_color = custom_colors_rgb{c_idx};
        cluster_jitters = (rand(1, num_vars) - 0.5) * max_jitter;
        
        cluster_x_coords_for_line = NaN(1, num_vars);
        cluster_y_coords_for_line = NaN(1, num_vars);

        for v_idx = 1:num_vars
            diff_value = all_cluster_differences(c_idx, v_idx);
            if isnan(diff_value), continue; end
            
            y_position = v_idx + cluster_jitters(v_idx);
            se_value_for_plot = scaled_cluster_SEs(c_idx, v_idx);

            h_eb = errorbar(ax, diff_value, y_position, se_value_for_plot, 'horizontal', ...
                'Color', cluster_color, 'Marker', 'o', 'MarkerFaceColor', cluster_color, ...
                'MarkerSize', 7, 'LineWidth', 1.5, 'CapSize', 5, 'LineStyle', 'none');
            
            if ~isgraphics(plot_handles_for_legend(c_idx)) 
                plot_handles_for_legend(c_idx) = h_eb;
            end
            
            cluster_x_coords_for_line(v_idx) = diff_value;
            cluster_y_coords_for_line(v_idx) = y_position;
        end
        
        % Plot connecting line for the current cluster
        valid_line_points = ~isnan(cluster_x_coords_for_line);
        if sum(valid_line_points) > 1
            plot(ax, cluster_x_coords_for_line(valid_line_points), ...
                 cluster_y_coords_for_line(valid_line_points), ...
                 '-', 'Color', cluster_color, 'LineWidth', 1.0);
        end
    end
    
    % Add mean line
    plot(ax, [0 0], [0.5 num_vars + 0.5], 'Color', [0.7 0.7 0.7], 'LineWidth', 1);

    % --- 5. Axes and Figure Configuration ---
    clean_var_names = strrep(variablesForPlot, '_', ' ');
    set(ax, 'YTick', 1:num_vars, 'YTickLabel', clean_var_names, ...
            'YDir', 'reverse', 'FontSize', 12);
    ylim(ax, [0.5, num_vars + 0.5]);
    
    set(ax, 'XLim', [-1, 1], 'XTick', [-1, -0.5, 0, 0.5, 1]);
    ax.XAxisLocation = 'top'; 
    
    title('Cluster Deviations from Overall Mean (Normalized Data)', 'FontSize', 14);
    xlabel(ax, 'Normalized Value - Overall Mean', 'FontSize', 14, 'FontWeight', 'bold');
    
    legend_names = arrayfun(@(x) sprintf('Cluster %d', x), cluster_id_list, 'UniformOutput', false);
    valid_handles_idx = isgraphics(plot_handles_for_legend);
    if any(valid_handles_idx)
        lgd = legend(ax, plot_handles_for_legend(valid_handles_idx), legend_names(valid_handles_idx), ...
                     'Location', 'eastoutside', 'FontSize', 12);
        lgd.Box = 'off'; 
    end
    
    set(fig, 'Color', 'w'); 
    box(ax, 'off');
    grid(ax, 'off');
    
    hold(ax, 'off');
    disp('Cluster differences plot generated successfully.');
end






function modified_dataTable = plotSubjectPCA(dataTable, metadataTable, varargin)
% PLOTSUBJECTPCA Performs PCA, k-means, and plots with subdivided, smooth, expanded Bézier curve contours.
%
% This version returns a modified dataTable containing a new 'SubjectCluster'
% column with the cluster assignment for each session.
%
% Args:
%       dataTable (table): The main data table.
%       metadataTable (table): The metadata table for filtering variables.
% Optional Name-Value Args:
%       'Filters' (struct):   Keeps variables matching these criteria for the PCA.
%       'Exclude' (cell array): Removes specific variables from the analysis.
%       'output_folder' (string): Folder to save the figure in. Default is 'Subject_PCA'.
%
% Returns:
%       modified_dataTable (table): A copy of the input dataTable with an
%                                   added 'SubjectCluster' column.
    
    % --- Argument Parsing ---
    p = inputParser;
    addParameter(p, 'Filters', struct(), @isstruct);
    addParameter(p, 'Exclude', {}, @iscell);
    addParameter(p, 'output_folder', 'Subject_PCA', @ischar);
    parse(p, varargin{:});
    filters = p.Results.Filters;
    variablesToExclude = p.Results.Exclude;
    output_folder = p.Results.output_folder;

    % --- Create a copy of the dataTable to be modified and returned ---
    modified_dataTable = dataTable;
    
    % --- [NEW] Check for and remove an existing SubjectCluster column ---
    if ismember('SubjectCluster', modified_dataTable.Properties.VariableNames)
        disp('Removing existing "SubjectCluster" column before re-calculating.');
        modified_dataTable.SubjectCluster = [];
    end

    % --- 1. Variable Selection (Filtering and Exclusion) ---
    filtered_metadata = metadataTable;
    filterFields = fieldnames(filters);
    if ~isempty(filterFields)
        disp('Applying inclusion filters to select variables for PCA...');
        for i = 1:length(filterFields)
            fieldName = filterFields{i};
            filterValue = filters.(fieldName);
            filtered_metadata = filtered_metadata(strcmp(filtered_metadata.(fieldName), filterValue), :);
        end
    end
    if ~isempty(variablesToExclude)
        is_on_exclude_list = ismember(filtered_metadata.ModifiedName, variablesToExclude);
        filtered_metadata = filtered_metadata(~is_on_exclude_list, :);
    end
	if isempty(filtered_metadata), disp('No variables remained after filtering.'); return; end
    variablesForPCA = filtered_metadata.ModifiedName;
    fprintf('Performing PCA based on %d variables.\n', numel(variablesForPCA));
    % --- 2. Prepare Data and Handle NaNs ---
    dataForPCA_table = dataTable(:, variablesForPCA);
    subjectIDs = dataTable.SubjectID;
    rowsWithNaN = any(ismissing(dataForPCA_table), 2);
    cleanData_table = dataForPCA_table(~rowsWithNaN, :);
    dataForPCA_matrix = cleanData_table.Variables;
    subjectIDs_cleaned = subjectIDs(~rowsWithNaN);
    if any(rowsWithNaN), fprintf('Removed %d sessions with incomplete data.\n', sum(rowsWithNaN)); end
    if size(dataForPCA_matrix, 1) < 2, error('Not enough data to perform analysis after cleaning.'); end
    % --- 3. Perform PCA and K-Means Clustering ---
    [~, score, ~, ~, explained] = pca(dataForPCA_matrix);
    disp('Finding optimal number of clusters using Silhouette method...');
    max_k = min(10, size(score, 1) - 1);
    if max_k < 2
        optimal_k = 1;
    else
        eva = evalclusters(score(:,1:2), 'kmeans', 'silhouette', 'KList', 1:max_k);
        optimal_k = eva.OptimalK;
    end
    fprintf('Optimal number of clusters found: %d\n', optimal_k);
    cluster_indices = kmeans(score(:,1:2), optimal_k, 'Replicates', 5);

    % --- [NEW] Add Cluster Information to the Output Table ---
    % Create a new column, the size of the original table, initialized with NaN
    num_total_sessions = height(modified_dataTable);
    cluster_column = nan(num_total_sessions, 1);
    % Use the logical index of non-NaN rows to place cluster indices correctly
    cluster_column(~rowsWithNaN) = cluster_indices;
    % Add the fully mapped column to the table
    modified_dataTable.SubjectCluster = cluster_column;
    disp('Added "SubjectCluster" column to the output data table.');
    
    % --- 4. Create the Plot ---
    fig_handle = figure('Name', 'PCA Scores with K-Means Clustering', 'Position', [100 100 1000 800]);
    hold on;
    unique_subjects = unique(subjectIDs_cleaned);
    subject_colors = lines(length(unique_subjects));
    cluster_shapes = ['o', 's', 'd', '^', 'v', 'p', 'h'];
    if optimal_k > length(cluster_shapes)
        cluster_shapes = repmat(cluster_shapes, 1, ceil(optimal_k/length(cluster_shapes)));
    end
    for i = 1:size(score, 1)
        subj_idx = find(strcmp(unique_subjects, subjectIDs_cleaned(i)));
        clust_idx = cluster_indices(i);
        scatter(score(i, 1), score(i, 2), 60, subject_colors(subj_idx,:), cluster_shapes(clust_idx), 'filled');
    end
    % --- Bézier Boundary Contour Logic ---
    for k = 1:optimal_k
        cluster_points = score(cluster_indices == k, 1:2);
        if size(unique(cluster_points, 'rows'), 1) < 3
            fprintf('Skipping contour for cluster %d (not enough unique points).\n', k);
            continue;
        end
        try
            % Define aesthetic parameters
            x_range = range(score(:,1));
            y_range = range(score(:,2));
            margin = 0.025 * mean([x_range, y_range]);
            smoothness_factor = 0.4; 
            subdivision_level = 3;

            % Get and expand the initial boundary
            boundary_indices = convhull(cluster_points, 'Simplify', true);
            original_boundary_points = cluster_points(boundary_indices, :);
            
            cluster_center = mean(cluster_points, 1);
            
            expanded_boundary_points = zeros(size(original_boundary_points));
            for i = 1:size(original_boundary_points, 1)
                point = original_boundary_points(i,:);
                vec_from_center = point - cluster_center;
                expanded_boundary_points(i,:) = point + (vec_from_center / norm(vec_from_center)) * margin;
            end
            
            % Subdivide the expanded hull for extra smoothness
            if subdivision_level > 1 && size(expanded_boundary_points, 1) > 1
                subdivided_points = [];
                for i = 1:size(expanded_boundary_points, 1)
                    p_start = expanded_boundary_points(i, :);
                    p_end_idx = mod(i, size(expanded_boundary_points, 1)) + 1;
                    p_end = expanded_boundary_points(p_end_idx, :);
                    x_coords = linspace(p_start(1), p_end(1), subdivision_level + 1);
                    y_coords = linspace(p_start(2), p_end(2), subdivision_level + 1);
                    subdivided_points = [subdivided_points; [x_coords(1:end-1)', y_coords(1:end-1)']];
                end
                boundary_points = subdivided_points;
            else
                boundary_points = expanded_boundary_points;
            end

            if size(boundary_points, 1) > 2
                num_bnd_pts = size(boundary_points, 1);
                full_curve = []; % To store all segments

                for i = 1:num_bnd_pts
                    p_prev_idx = mod(i-2, num_bnd_pts) + 1;
                    p_curr_idx = i;
                    p_next_idx = mod(i, num_bnd_pts) + 1;
                    p_next_next_idx = mod(i+1, num_bnd_pts) + 1;
                    
                    P_prev = boundary_points(p_prev_idx, :);
                    P0 = boundary_points(p_curr_idx, :);
                    P3 = boundary_points(p_next_idx, :);
                    P_next_next = boundary_points(p_next_next_idx, :);
                    
                    T1 = P3 - P_prev;
                    T2 = P_next_next - P0;
                    
                    if norm(T1) > 1e-6, T1 = T1 / norm(T1); else, T1 = [0 0]; end
                    if norm(T2) > 1e-6, T2 = T2 / norm(T2); else, T2 = [0 0]; end
                    
                    segment_length = norm(P3 - P0);
                    C1 = P0 + smoothness_factor * segment_length * T1;
                    C2 = P3 - smoothness_factor * segment_length * T2;
                    
                    control_matrix = [P0; C1; C2; P3];
                    curve_segment = generate_bezier_curve(control_matrix, 20);
                    
                    full_curve = [full_curve; curve_segment];
                end
                
                plot(full_curve(:,1), full_curve(:,2), 'k-', 'LineWidth', 1.5);
                
            else
                plot(boundary_points(:,1), boundary_points(:,2), 'k-', 'LineWidth', 1.5);
            end
        
        catch ME
            warning('Could not generate Bézier contour for cluster %d. Reason: %s', k, ME.message);
            try % Fallback to straight lines
                hull_indices = convhull(cluster_points);
                plot(cluster_points(hull_indices, 1), cluster_points(hull_indices, 2), 'k-', 'LineWidth', 1.5);
            catch
            end
        end
    end

    % --- 5. Create a Manual Legend for Clusters and Subjects ---
    cluster_handles = [];
    cluster_labels = {};
    for k = 1:optimal_k
        h = scatter(NaN, NaN, 60, 'k', cluster_shapes(k), 'DisplayName', ['Cluster ' num2str(k)]);
        cluster_handles(end+1) = h;
        cluster_labels{end+1} = ['Cluster ' num2str(k)];
    end

    subject_handles = [];
    subject_labels = {};
    for i = 1:length(unique_subjects)
        s_label = strrep(unique_subjects{i}, '_', ' ');
        h = plot(NaN, NaN, 's', 'MarkerFaceColor', subject_colors(i,:), 'MarkerSize', 8, 'LineStyle', 'none', 'DisplayName', s_label);
        subject_handles(end+1) = h;
        subject_labels{end+1} = s_label;
    end
    
    all_handles = [cluster_handles, subject_handles];
    all_labels = [cluster_labels, subject_labels];

    lgd = legend(all_handles, all_labels, 'Location', 'northeastoutside', 'NumColumns', 1);
    title(lgd, 'Legend');
    
    % --- 6. Final Touches with Dynamic Title ---
    hold off;
    main_title = ['PCA Scores with ' num2str(optimal_k) ' k-means Clusters'];
    title_lines = {main_title};
    if ~isempty(fieldnames(filters))
        filter_parts = cellfun(@(f) [f '=' filters.(f)], fieldnames(filters), 'UniformOutput', false);
        title_lines{end+1} = ['\color[rgb]{0 0.4470 0.7410}Using Variables Where: ' strjoin(filter_parts, ', ')];
    end
    if ~isempty(variablesToExclude)
        title_lines{end+1} = ['\color[rgb]{0.8500 0.3250 0.0980}Excluding: ' strjoin(strrep(variablesToExclude, '_', ' '))];
    end
    title(title_lines, 'FontSize', 12);
    xlabel(['Principal Component 1 (' num2str(explained(1), '%.2f') '%)']);
    ylabel(['Principal Component 2 (' num2str(explained(2), '%.2f') '%)']);
    
    x_range_total = range(score(:, 1)); y_range_total = range(score(:, 2));
    x_pad = x_range_total*0.1; if x_pad==0, x_pad=1; end; y_pad = y_range_total*0.1; if y_pad==0, y_pad=1; end
    xlim([min(score(:, 1))-x_pad, max(score(:, 1))+x_pad]);
    ylim([min(score(:, 2))-y_pad, max(score(:, 2))+y_pad]);
    grid on; box on;
    % --- 7. Save the Figure Automatically ---
    if ~exist(output_folder, 'dir'), mkdir(output_folder); end
    timestamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
    output_filename = fullfile(output_folder, ['Subject_PCA_k' num2str(optimal_k) '_' timestamp '.png']);
    disp(['Saving figure to: ' output_filename]);
    print(fig_handle, output_filename, '-dpng', '-r300');
    fprintf('Figure successfully saved.\n');
end

% You still need the helper function saved as generate_bezier_curve.m
% function B = generate_bezier_curve(P, num_points) ...

% Filename: generate_bezier_curve.m
function B = generate_bezier_curve(P, num_points)
% GENERATE_BEZIER_CURVE Generates points for a cubic Bézier curve.
%
% Args:
%       P (4x2 matrix): The four control points [P0; P1; C1; P2], where
%                       P0 is the start point, P2 is the end point, and
%                       C1 and C2 are the control handles.
%       num_points (integer): The number of points to generate for the curve.
%
% Returns:
%       B (num_points x 2 matrix): The [x, y] coordinates of the curve.

    if nargin < 2
        num_points = 100; % Default number of points
    end
    
    t = linspace(0, 1, num_points)';
    
    % The Bézier formula for a cubic curve
    B = (1-t).^3 .* P(1,:) + ...
        3*(1-t).^2 .* t .* P(2,:) + ...
        3*(1-t) .* t.^2 .* P(3,:) + ...
        t.^3 .* P(4,:);
end


function [R_matrix_observed, P_matrix_parametric, fig_handle] = plot_RMC_heatmap(dataTable, metadataTable, varargin)
% PLOT_RMC_HEATMAP Generates a highly customized RMC heatmap with dynamic subtitles.
%
% This final version adds subtitles to the plot to show which filters
% and exclusions were used to generate it.
%
% Inputs:
%   dataTable (table):     MATLAB table with 'SubjectID' and behavioral variables.
%   metadataTable (table): Table with metadata.
%
% Optional Name-Value Inputs:
%   'Variables' (cell array): REQUIRED. A base list of variable names.
%   'Filters' (struct):   OPTIONAL. Struct to filter the 'Variables' list by category.
%   'Exclude' (cell array): OPTIONAL. List of specific variable names to remove.
%   'p_threshold' (double):   OPTIONAL. Significance level. Default is 0.05.
%   'output_folder' (string): OPTIONAL. Folder to save the figure in. Default is 'RMC_Heatmaps'.

% --- 1. Comprehensive Input Parsing ---
p = inputParser;
addParameter(p, 'Variables', metadataTable.ModifiedName, @iscell);
addParameter(p, 'Filters', struct(), @isstruct);
addParameter(p, 'Exclude', {}, @iscell);
addParameter(p, 'p_threshold', 0.05, @isnumeric);
addParameter(p, 'output_folder', 'RMC_Heatmaps', @ischar);
parse(p, varargin{:});

vars_initial = p.Results.Variables;
filters = p.Results.Filters;
variablesToExclude = p.Results.Exclude;
p_threshold = p.Results.p_threshold;
output_folder = p.Results.output_folder;

if isempty(vars_initial)
    error('Please provide a base list of variables using the ''Variables'' parameter.');
end

% --- 2. Multi-Step Variable Selection ---
vars_filtered = vars_initial;
filterFields = fieldnames(filters);
if ~isempty(filterFields)
    disp('Applying category filters...');
    [is_in_metadata, loc] = ismember(vars_filtered, metadataTable.ModifiedName);
    vars_to_keep_mask = true(size(vars_filtered));
    for i = 1:length(vars_filtered)
        if ~is_in_metadata(i), vars_to_keep_mask(i) = false; continue; end
        current_var_metadata_row = metadataTable(loc(i), :);
        for f_idx = 1:length(filterFields)
            fieldName = filterFields{f_idx};
            filterValue = filters.(fieldName);
            if ~strcmp(current_var_metadata_row.(fieldName), filterValue)
                vars_to_keep_mask(i) = false;
                break;
            end
        end
    end
    vars_filtered = vars_filtered(vars_to_keep_mask);
end
if ~isempty(variablesToExclude)
    disp('Applying exclusion list...');
    vars_filtered = setdiff(vars_filtered, variablesToExclude, 'stable');
end
numeric_vars_for_heatmap = vars_filtered;
fprintf('Final variable count for heatmap: %d\n', numel(numeric_vars_for_heatmap));

% --- 3. Validation and Data Cleaning ---
if numel(numeric_vars_for_heatmap) < 2
    disp('Not enough variables remain after filtering to create a heatmap.');
    R_matrix_observed=[]; P_matrix_parametric=[]; fig_handle=[]; return;
end
disp(['Using p-value threshold: ', num2str(p_threshold)]);
dataTable_cleaned = rmmissing(dataTable, 'DataVariables', unique([numeric_vars_for_heatmap(:)', {'SubjectID'}]));
if height(dataTable_cleaned) < 3, R_matrix_observed=[]; P_matrix_parametric=[]; return; end
dataTable_cleaned.SubjectID = categorical(dataTable_cleaned.SubjectID);

% --- 4. Calculate RMC ---
num_vars = numel(numeric_vars_for_heatmap);
R_matrix_observed = NaN(num_vars, num_vars);
P_matrix_parametric = NaN(num_vars, num_vars);
disp('Calculating Observed RMCs...');
for i = 1:num_vars
    for j = 1:i
        if i == j, R_matrix_observed(i,j) = 1; P_matrix_parametric(i,j) = 0; continue; end
        [r_val, p_val] = local_calculate_rmc_with_p(dataTable_cleaned.(numeric_vars_for_heatmap{i}), ...
            dataTable_cleaned.(numeric_vars_for_heatmap{j}), dataTable_cleaned.SubjectID);
        R_matrix_observed(i,j) = r_val;
        P_matrix_parametric(i,j) = p_val;
    end
end
disp('Calculation complete.');

% --- 5. Create Heatmap ---
heatmap_labels = strrep(numeric_vars_for_heatmap, '_', ' ');
fig_handle = figure('Name', 'Significant RMC R-values', 'Color','w', ...
    'Units','normalized','OuterPosition',[0.2 0.1 0.6 0.8]);
ax_handle = axes(fig_handle);
R_display_for_plot = R_matrix_observed;
R_display_for_plot(P_matrix_parametric >= p_threshold) = NaN;
im_r = imagesc(ax_handle, R_display_for_plot);
im_r.AlphaData = ~isnan(R_display_for_plot);
ax_handle.Color = [0.92 0.92 0.92];
axis(ax_handle, 'image');
colormap(ax_handle, diverging_cmap([0 0 1], [1 1 1], [1 0 0]));
clim(ax_handle, [-1 1]);
h_cb = colorbar(ax_handle);
ylabel(h_cb, 'R-value');

% *** NEW: Generate dynamic title with subtitles ***
main_title = ['Observed RMC R-values (p < ' num2str(p_threshold) ')'];
title_lines = {main_title};
if ~isempty(filterFields)
    filter_parts = cellfun(@(f) [f '=' filters.(f)], filterFields, 'UniformOutput', false);
    filter_subtitle = 'Variable Property: ' + strjoin(filter_parts{1}, ' ');
    title_lines{end+1} = filter_subtitle;
end
if ~isempty(variablesToExclude)
    if numel(variablesToExclude) > 3
        exclude_list_str = [strjoin(variablesToExclude(1:3), ', '), ', ...'];
    else
        exclude_list_str = strjoin(variablesToExclude, ', ');
    end
    exclude_subtitle = ['Excluding Variables: ' strjoin(strrep(exclude_list_str{1}, '_', ' '), ', ')];
    title_lines{end+1} = exclude_subtitle;
end
title(ax_handle, title_lines, 'FontSize', 12);
% ***********************************************

xticks(ax_handle, 1:num_vars);
xticklabels(ax_handle, heatmap_labels);
xtickangle(ax_handle, 45);
yticks(ax_handle, 1:num_vars);
yticklabels(ax_handle, heatmap_labels);
for r = 1:num_vars
    for c = 1:r
        r_val = R_display_for_plot(r, c);
        if ~isnan(r_val)
            text_str = sprintf('%.2f', r_val);
            th = text(ax_handle, c, r, text_str, ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 8, 'FontWeight', 'bold');
            if abs(r_val) > 0.6; th.Color = 'white'; else; th.Color = 'black'; end
        end
    end
end

% --- 6. Add Task Group Borders ---
hold(ax_handle, 'on');
[~, loc] = ismember(numeric_vars_for_heatmap, metadataTable.ModifiedName);
tasks_for_vars = repmat({'Unknown'}, num_vars, 1);
tasks_for_vars(loc > 0) = metadataTable.TaskName(loc(loc>0));
if any(loc==0), warning('Some variables were not found in the metadata table.'); end
[unique_tasks, ~, task_indices] = unique(tasks_for_vars, 'stable');
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

% --- 7. Save the Figure ---
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end
timestamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
output_filename = fullfile(output_folder, ['RMC_Heatmap_' timestamp '.png']);
disp(['Saving figure to: ' output_filename]);
print(fig_handle, output_filename, '-dpng', '-r300');
fprintf('Figure successfully saved.\n');

end

% --- Local Helper Functions ---
function [r, p_val] = local_calculate_rmc_with_p(v1, v2, subjects_cat)
    v1=v1(:); v2=v2(:); subjects_cat=subjects_cat(:); r=NaN; p_val=NaN;
    unique_subj_ids=unique(subjects_cat); demeaned_v1=[]; demeaned_v2=[];
    N_total_obs=0; num_subjects_contributing=0;
    for i_s=1:numel(unique_subj_ids)
        idx_s=(subjects_cat==unique_subj_ids(i_s));
        if sum(idx_s)>=2
            demeaned_v1=[demeaned_v1; v1(idx_s)-mean(v1(idx_s))];
            demeaned_v2=[demeaned_v2; v2(idx_s)-mean(v2(idx_s))];
            N_total_obs=N_total_obs+sum(idx_s);
            num_subjects_contributing=num_subjects_contributing+1;
        end
    end
    if isempty(demeaned_v1) || num_subjects_contributing < 1, return; end
    if std(demeaned_v1,'omitnan')<eps || std(demeaned_v2,'omitnan')<eps, return; end
    [corr_matrix,~]=corrcoef(demeaned_v1,demeaned_v2,'Rows','complete');
    if numel(corr_matrix)<4, return; end; r=corr_matrix(1,2);
    if isnan(r), return; end; df=N_total_obs-num_subjects_contributing-1;
    if df<=0, return; end; t_stat=r*sqrt(df/(1-r^2));
    p_val=2*tcdf(-abs(t_stat),df);
end

function cmap = diverging_cmap(low_color, mid_color, high_color)
    N=256;
    map1=[linspace(low_color(1),mid_color(1),N/2)', linspace(low_color(2),mid_color(2),N/2)', linspace(low_color(3),mid_color(3),N/2)'];
    map2=[linspace(mid_color(1),high_color(1),N/2)', linspace(mid_color(2),high_color(2),N/2)', linspace(mid_color(3),high_color(3),N/2)'];
    cmap=[map1; map2];
end



function plotPCALoadings(dataTable, metadataTable, varargin)
% plotPCALoadings Performs PCA and creates a publication-quality plot with subtitles and auto-saving.
%
% This version now includes dynamic subtitles describing the filters used
% and automatically saves the output to a 'PCAs' folder.
%
% Args:
%       dataTable (table): The main data table.
%       metadataTable (table): The metadata table.
% Optional Name-Value Args:
%       'Filters' (struct):   Keeps variables matching these criteria.
%       'Exclude' (cell array): Removes specific variables by name.
%       'output_folder' (string): Folder to save the figure in. Default is 'PCAs'.

    % --- Argument Parsing for Filters and Exclusions ---
    p = inputParser;
    addParameter(p, 'Filters', struct(), @isstruct);
    addParameter(p, 'Exclude', {}, @iscell);
    addParameter(p, 'output_folder', 'PCAs', @ischar); % Add output folder option
    parse(p, varargin{:});
    filters = p.Results.Filters;
    variablesToExclude = p.Results.Exclude;
    output_folder = p.Results.output_folder; % Get output folder
    
    if ~license('test', 'statistics_toolbox')
        error('This function requires the Statistics and Machine Learning Toolbox.');
    end
    
    % --- 1. Data Selection (Filtering and Exclusion) ---
    filtered_metadata = metadataTable;
    filterFields = fieldnames(filters);
    if ~isempty(filterFields)
        disp('Applying inclusion filters:');
        for i = 1:length(filterFields)
            fieldName = filterFields{i};
            filterValue = filters.(fieldName);
            fprintf('  - Keeping where "%s" is "%s"\n', fieldName, filterValue);
            filtered_metadata = filtered_metadata(strcmp(filtered_metadata.(fieldName), filterValue), :);
        end
    end
    
    if ~isempty(variablesToExclude)
        is_on_exclude_list = ismember(filtered_metadata.ModifiedName, variablesToExclude);
        fprintf('Excluding %d specific variables by name.\n', sum(is_on_exclude_list));
        filtered_metadata = filtered_metadata(~is_on_exclude_list, :);
    end
    
	if isempty(filtered_metadata)
        disp('No variables remained after applying all filters.');
        return;
    end
    
    variablesForPCA = filtered_metadata.ModifiedName;
    taskGroups = filtered_metadata.TaskName;
    factorGroups = filtered_metadata.CognitiveControlFactor;

    % --- 2. Prepare Data and Perform PCA ---
    dataForPCA_table = dataTable(:, variablesForPCA);
    numericData = dataForPCA_table.Variables;
    rowsWithNaN = any(isnan(numericData), 2);
    if any(rowsWithNaN)
        fprintf('Found NaN values. Removing %d subjects (rows) with incomplete data.\n', sum(rowsWithNaN));
        dataForPCA_matrix = numericData(~rowsWithNaN, :);
    else
        disp('No missing values found in the selected data.');
        dataForPCA_matrix = numericData;
    end
    if isempty(dataForPCA_matrix) || size(dataForPCA_matrix, 1) < 2
        error('Not enough data remains to perform PCA after removing subjects with NaN values.');
    end
    [coeff, ~, ~, ~, explained] = pca(dataForPCA_matrix);
    pc1_loadings = coeff(:, 1);
    pc2_loadings = coeff(:, 2);

    % --- 3. Prepare for Advanced Plotting ---
    uniqueTasks = unique(taskGroups);
    uniqueFactors = unique(factorGroups);
    colors = lines(length(uniqueTasks));
    shapes = ['o', 's', 'd', '^', 'v', 'p', 'h', '>', '<'];
    if length(uniqueFactors) > length(shapes)
        shapes = repmat(shapes, 1, ceil(length(uniqueFactors)/length(shapes)));
    end
    x_range = range(pc1_loadings);
    y_range = range(pc2_loadings);
    x_pad = x_range * 0.1; if x_pad == 0, x_pad = 1; end
    y_pad = y_range * 0.1; if y_pad == 0, y_pad = 1; end
    x_limits = [min(pc1_loadings) - x_pad, max(pc1_loadings) + x_pad];
    y_limits = [min(pc2_loadings) - y_pad, max(pc2_loadings) + y_pad];

    % --- 4. Create the Plot ---
    fig_handle = figure('Name', 'PCA Loadings of Cognitive Variables', 'Position', [100 100 1000 800]);
    hold on;
    for i = 1:length(uniqueTasks)
        for j = 1:length(uniqueFactors)
            idx = strcmp(taskGroups, uniqueTasks{i}) & strcmp(factorGroups, uniqueFactors{j});
            if ~any(idx), continue; end
            scatter(pc1_loadings(idx), pc2_loadings(idx), 60, colors(i,:), shapes(j), 'filled', 'MarkerEdgeColor', 'k');
        end
    end
    textOffset = 0.01 * x_range;
    for i = 1:length(variablesForPCA)
        text(pc1_loadings(i) + textOffset, pc2_loadings(i), variablesForPCA(i), 'Interpreter', 'none');
    end

    % --- 5. Create a Manual Legend ---
    legend_handles = [];
    legend_labels = {};
    for i = 1:length(uniqueTasks)
        h = scatter(NaN, NaN, 60, colors(i,:), 'o', 'filled');
        legend_handles(end+1) = h;
        legend_labels{end+1} = uniqueTasks{i};
    end
    legend_handles(end+1) = plot(NaN, NaN, 'w');
    legend_labels{end+1} = ' ';
    for j = 1:length(uniqueFactors)
        h = scatter(NaN, NaN, 60, 'k', shapes(j));
        legend_handles(end+1) = h;
        legend_labels{end+1} = uniqueFactors{j};
    end
    lgd = legend(legend_handles, legend_labels, 'Location', 'northeastoutside');
    title(lgd, 'Legend');
    
    % --- 6. Final Touches with Dynamic Title ---
    hold off;
    % *** NEW: Generate dynamic title with subtitles ***
    main_title = 'PCA Loadings of Cognitive Variables';
    title_lines = {main_title};
    if ~isempty(filterFields)
        filter_parts = cellfun(@(f) [f '=' filters.(f)], filterFields, 'UniformOutput', false);
        filter_subtitle = 'Variable Property: ' + strjoin(filter_parts{1}, ' ');
        title_lines{end+1} = filter_subtitle;
    end
    if ~isempty(variablesToExclude)
        if numel(variablesToExclude) > 3
            exclude_list_str = [strjoin(variablesToExclude(1:3), ', '), ', ...'];
        else
            exclude_list_str = strjoin(variablesToExclude, ', ');
        end
        exclude_subtitle = ['Excluding Variables: ' strrep(exclude_list_str, '_', ' ')];
        title_lines{end+1} = exclude_subtitle;
    end
    title(title_lines, 'FontSize', 12);
    % ***********************************************
    
    xlabel(['Principal Component 1 (' num2str(explained(1), '%.2f') '%)']);
    ylabel(['Principal Component 2 (' num2str(explained(2), '%.2f') '%)']);
    xlim(x_limits);
    ylim(y_limits);
    grid on;
    box on;
    
    % --- 7. Save the Figure Automatically ---
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end
    timestamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
    output_filename = fullfile(output_folder, ['PCA_Loadings_' timestamp '.png']);
    disp(['Saving figure to: ' output_filename]);
    print(fig_handle, output_filename, '-dpng', '-r300');
    fprintf('Figure successfully saved.\n');
end