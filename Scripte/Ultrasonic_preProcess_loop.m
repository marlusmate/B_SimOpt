%% Ultrasonic Data Preprocessing
% Only select needed Sensordata (Doppler frequency shift - Voltage (+-10V))
% & Timestamps/Messnummer (<=60s, 600.000 Messungen (10kHz))
% Drop thirt Column 

%% Get file list
% get file names
files = dir('RawData\Ultrasonic\\Raw');                % lade Namen der Dateien im Ordner
files = string({files.name})';
files = files(contains(files,'Test'));

%% Preprocess Loop over files

tags = {'ScanNr',	'voltage'}; % from page 18 "Technical Report"
for i = 1:numel(files)
    file_name = files(i);
    path = strcat('.\Ultrasonic\Raw\', file_name);
    
    ut_data_raw = csvread(path);
    ut_data_raw(:,3) = [];
    ut_data_raw(1,1) = 0.00001;     
    ut_data = ut_data_raw(1:600000,:); %cut of at 60s
    ut_data = array2table(ut_data, "VariableNames", tags);
    
    writetable(ut_data, strcat('Ultrasonic\', file_name))       %speicherts als csv
    fprintf('###Prozessdatei %s gekürzt und gespeichert###\n', file_name)
end

disp('>>>>>>Alle Dateien bearbeitet und gespeichert<<<<<<<')
