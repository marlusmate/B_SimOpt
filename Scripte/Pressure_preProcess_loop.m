%% Pressure Data Preprocessing
% Only select needed Sensordata (P4-8,14,15,17) & NO Timestamps (Nutzen der
% Timestamps noch nicht ganz klar, Matlab lädt timestamps falsch, können
% später noch hinzugefügt werden)
% Calibrate Data, match with sensor names

%% Get file list
% get file names
files = dir('RawData\Pressure\Raw');                % lade Namen der Dateien im Ordner
files = string({files.name})';
files = files(contains(files,'Test'));

%% Preprocess Loop over files

tags = {'P4-B14',	'P5-B13',	'P6-B12', 'P7-B11',	'P8-B10',	'P13-B20',	'P14-B08',	'P15-B09','P17-B05'}; % from Table 4 page 9 "Technical Report"
ts_file = cell(numel(files)+1,2);
ts_file{1,1} = "Timestamp";
ts_file{1,2} = "ExpNr";
l = 2;

for i = 1:numel(files)
    file_name = files(i);
    path = strcat('.\Pressure\Raw\', file_name);

    % Extrahiere Zeitpunkt Messung, sehr ugly
    fid = fopen(path);
    out = textscan(fid,'%s%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f','delimiter',',');
    ts= out{1}{3};
    par = extractBetween(file_name, "_", ".csv");
    ts_file{l,1} = ts;
    ts_file{l,2} = par;
    l= l+1;
    
    %
    % Bearbeite Datensheet
    p_data_uc = readtable(path, 'NumHeaderLines', 9);
    calibration_data = readtable(path, "Range", "C5:L6");   % Range irgendwie um 1 versetzt, wie als würde er bei 0 statt 1 anfangen zu zählen

    p_data_sensors = table2array(p_data_uc(:,3:12)).* table2array(calibration_data(2,:)) + table2array(calibration_data(1,:));
    p_data_sensors(:,9) = []; % droppen von P16-Sensor
    p_data_sensors = array2table(p_data_sensors, "VariableNames", tags);
    
    writetable(p_data_sensors, strcat('Pressure\', file_name))
    fprintf('###Prozessdatei %s erfoglreich sortiert, kalibriert und gespeichert###\n', file_name)
    %}
    fclose(fid);
end
ts_file{2,1} = '12:27';     %der File hakt irgendwie
writecell(ts_file, "ExpTimestamp.csv");
