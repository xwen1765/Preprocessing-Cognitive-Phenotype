
% ---
% --- this script prpocesses Data from Session folders and saves trialdata,
% --- blockdata and framedata from each task in a matfile
% ---
addpath '/Users/wenxuan/Documents/MATLAB/Multitasking_analysis/fcn_local'
SessionSets = {};

SessionSets{end+1} = 'WM_AS_CR_FL_083_Bard';
SessionSets{end+1} = 'WM_AS_CR_FL_083_Frey';
SessionSets{end+1} = 'WM_AS_CR_FL_083_Igor'; 
SessionSets{end+1} = 'WM_AS_CR_FL_083_Reider'; 
SessionSets{end+1} = 'WM_AS_CR_FL_083_Sindri'; 
SessionSets{end+1} = 'WM_AS_CR_FL_083_Wotan';


doReadFrameData = 0;

% --- specify the session folder and the sessions to analyze
for iO = 1:length(SessionSets)
    FOLDER_SESSION = [];

    if strcmp(SessionSets{iO},'WM_AS_CR_FL_083_Bard')
        EXP_ID = 'WM_AS_CR_FL_083_Bard';
        Subject = 'Bard';
        FOLDER_DATA = '/Volumes/Womelsdorf Lab/DATA_kiosk/Bard/WM_AS_CR_FL_083';
        RESULT_FOLDER = [pwd filesep 'MUSEMAT01_WM_AS_CR_FL_083_Bard']; if ~exist(RESULT_FOLDER), mkdir(RESULT_FOLDER), end
		
	elseif strcmp(SessionSets{iO},'WM_AS_CR_FL_083_Frey')
        EXP_ID = 'WM_AS_CR_FL_083_Frey';
        Subject = 'Frey';
        FOLDER_DATA = '/Volumes/Womelsdorf Lab/DATA_kiosk/Frey/WM_AS_CR_FL_083';
        RESULT_FOLDER = [pwd filesep 'MUSEMAT01_WM_AS_CR_FL_083_Frey']; if ~exist(RESULT_FOLDER), mkdir(RESULT_FOLDER), end


	elseif strcmp(SessionSets{iO},'WM_AS_CR_FL_083_Igor')
        EXP_ID = 'WM_AS_CR_FL_083_Igor';
        Subject = 'Igor';
        FOLDER_DATA ='/Volumes/Womelsdorf Lab/DATA_kiosk/Igor/WM_AS_CR_FL_083'
        RESULT_FOLDER = [pwd filesep 'MUSEMAT01_WM_AS_CR_FL_083_Igor']; if ~exist(RESULT_FOLDER), mkdir(RESULT_FOLDER), end

	elseif strcmp(SessionSets{iO},'WM_AS_CR_FL_083_Reider')
        EXP_ID = 'WM_AS_CR_FL_083_Reider';
        Subject = 'Reider';
        FOLDER_DATA = '/Volumes/Womelsdorf Lab/DATA_kiosk/Reider/WM_AS_CR_FL_083'
        RESULT_FOLDER = [pwd filesep 'MUSEMAT01_WM_AS_CR_FL_083_Reider']; if ~exist(RESULT_FOLDER), mkdir(RESULT_FOLDER), end

	elseif strcmp(SessionSets{iO},'WM_AS_CR_FL_083_Sindri')
        EXP_ID = 'WM_AS_CR_FL_083_Sindri';
        Subject = 'Sindri';
        FOLDER_DATA = '/Volumes/Womelsdorf Lab/DATA_kiosk/Sindri/WM_AS_CR_FL_083'
        RESULT_FOLDER = [pwd filesep 'MUSEMAT01_WM_AS_CR_FL_083_Sindri']; if ~exist(RESULT_FOLDER), mkdir(RESULT_FOLDER), end

	elseif strcmp(SessionSets{iO},'WM_AS_CR_FL_083_Wotan')
        EXP_ID = 'WM_AS_CR_FL_083_Wotan';
        Subject = 'Wotan';
        FOLDER_DATA = '/Volumes/Womelsdorf Lab/DATA_kiosk/Wotan/WM_AS_CR_FL_083'
        RESULT_FOLDER = [pwd filesep 'MUSEMAT01_WM_AS_CR_FL_083_Wotan']; if ~exist(RESULT_FOLDER), mkdir(RESULT_FOLDER), end


    end

    % --- --- --- --- --- --- ---
    % --- collect all files in the Datafolder if they were not specified
    % --- --- --- --- --- --- ---
    if isempty(FOLDER_SESSION)
        dirInfo = dir(FOLDER_DATA);
        FOLDER_SESSION = {dirInfo([dirInfo.isdir]).name};
        if isempty(dirInfo), sprintf('could not open %s (FOLDER_DATA)\n',FOLDER_DATA), continue, end
        FOLDER_SESSION = FOLDER_SESSION(~ismember(FOLDER_SESSION, {'.', '..'}));
        % --- for processing the latest data first...
        FOLDER_SESSION  = fliplr(FOLDER_SESSION);
    end

    % --- --- --- --- --- --- ---
    % --- determine which sessions to analyze
    % --- --- --- --- --- --- ---
    
    % --- start with last one added
    %FOLDER_SESSION = fliplr(FOLDER_SESSION)';

    for iSession = 1:length(FOLDER_SESSION)
        iSessionName = FOLDER_SESSION{iSession};
        iSessionDataFolder = [ FOLDER_DATA  filesep  iSessionName ];
        iResultFile = ['DAT01_' Subject '_' EXP_ID '_' iSessionName];
        iResultFileWithPath = [RESULT_FOLDER filesep iResultFile];

        % --- --- --- --- --- --- --- --- --- --- ---
        % --- pre-process M-USE session folder with all tasks
        % --- --- --- --- --- --- --- --- --- --- ---
        iDataFolder = [ iSessionDataFolder ];
        if ~exist([iSessionDataFolder filesep 'ProcessedData/'])
            Import_MUSE_Session_RawData('dataFolder', [iDataFolder],'gazeArgs','cancel','serialDataArgs','cancel');
        end

        % --- --- --- --- --- --- ---
        % --- find the tasks available in this session folder
        % --- --- --- --- --- --- ---
        taskNames = {};
        %
        dirInfo = dir(iSessionDataFolder)
        folderNames = {dirInfo([dirInfo.isdir]).name};
        if isempty(dirInfo), sprintf('could not open %s\n',iDataFolderSession), continue, end
        folderNames = folderNames(~ismember(folderNames, {'.', '..'})); % Removes '.' (current directory) and '..' (parent directory) from the list
        pattern = '^Task\d{4}_\w+';% Regular expression pattern to match TaskXXXX_Name
        taskNames = regexp(folderNames, pattern, 'match');% Use cellfun and regexp to extract task names
        taskNames = vertcat(taskNames{:});  % Convert cell array of cell arrays to a single cell array

        disp('...collecting data of these tasks:');
        disp(taskNames);

        if isempty(taskNames), 
            disp('empty taskNames, returning'), continue
        end
        
        dat = [];
        dat.sessionFolder  = iSessionDataFolder;
        dat.sessionName    = iSessionName;
        dat.sessionTasks   = taskNames;
        dat.subject         = Subject;
        dat.expID           = EXP_ID;

        dat.taskLabel =[];
        dat.trialData = [];
        dat.blockData = [];
        dat.frameData = [];

        dat.cfg_BlockDef = [];
        dat.cfg_StimDef = [];
        dat.cfg_TrialDef = [];
        dat.cfg_MazeDef = [];

        % --- --- --- --- --- --- --- --- --- --- ---
        % --- read  trial, block and framdata
        % --- --- --- --- --- --- --- --- --- --- ---
        for iT = 1:length(taskNames)

            dat.taskLabel{iT} = taskNames{iT};

            % --- --- --- --- --- --- --- --- --- --- ---
            % --- read  trial, block and framdata
            % --- --- --- --- --- --- --- --- --- --- ---
            preFixFolder = [iSessionDataFolder filesep 'ProcessedData' filesep taskNames{iT} filesep];

            GotData = 0;
            if exist([preFixFolder  'TrialData.mat'])
				try
                in = load([preFixFolder  'TrialData.mat']);
				catch
					disp('Unable to read data stream because the data contains a bad version or endian-key');
					continue;
				end
                dat.trialData{iT} = table2struct(in.trialData);
                GotData=1;
            end

            if exist([preFixFolder  'BlockData.mat'])
				try
                in = load([preFixFolder  'BlockData']);
				catch
					disp('Unable to read data stream because the data contains a bad version or endian-key');
					continue;
				end
                dat.blockData{iT}  = table2struct(in.blockData);
                GotData=1;
            end

            if doReadFrameData == 1 & exist([preFixFolder  'FrameData'])~=0
				try
                in = load([preFixFolder   'FrameData']);
				catch
					disp('Unable to read data stream because the data contains a bad version or endian-key');
					continue;
				end
                dat.frameData{iT}  = table2struct(in.frameData);
            end

            % --- do not add information when neither trial nor block data was found.
            if GotData == 0, continue, end

            % --- --- --- --- --- --- --- --- --- --- ---
            % --- read session configs for each task too.
            % --- --- --- --- --- --- --- --- --- --- ---
            configFolderInfo = dir([iSessionDataFolder filesep  'SessionConfigs' filesep]);
            folderNames = {configFolderInfo([configFolderInfo.isdir]).name};
            folderNames = folderNames(~ismember(folderNames, {'.', '..'}))'; % Removes '.' (current directory) and '..' (parent directory) from the list
            for iConfig=1:length(folderNames), if contains(taskNames{iT}, folderNames{iConfig})==1, break, end, end

            configFiles = dir([iSessionDataFolder filesep  'SessionConfigs' filesep folderNames{iConfig}]);
            for j=1:length(configFiles)
                if (configFiles(j).isdir) | strcmp(configFiles(j).name(j),'.' ),  continue, end
                if ~isempty(findstr(configFiles(j).name,'BlockDef'))
                    dat.cfg_BlockDef{iT} = readtable([configFiles(j).folder filesep configFiles(j).name], 'Delimiter', '\t');
                elseif ~isempty(findstr(configFiles(j).name,'StimDef'))
                    dat.cfg_StimDef{iT} = readtable([configFiles(j).folder filesep configFiles(j).name], 'Delimiter', '\t');
                elseif ~isempty(findstr(configFiles(j).name,'TrialDef'))
                    dat.cfg_TrialDef{iT} = readtable([configFiles(j).folder filesep configFiles(j).name], 'Delimiter', '\t');
                elseif ~isempty(findstr(configFiles(j).name,'MazeDef'))
                    dat.cfg_MazeDef{iT} = readtable([configFiles(j).folder filesep configFiles(j).name], 'Delimiter', '\t');
                end
            end
        end

%dat.taskLabel

        % --- --- --- --- --- --- --- --- --- --- --- --- ---
        % --- save the preprocessed session data
        % --- --- --- --- --- --- --- --- --- --- --- --- ---
        save(iResultFileWithPath,'dat','-V7.3')
        sprintf('preprocessed sesssion %d of %d; saved dat in %s\n',iSession, length(FOLDER_SESSION), iResultFile),

    end

    sprintf('done with %d sessions.\n',length(iSession)),
end

return

