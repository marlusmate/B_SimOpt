%pd = readtable(".\Process\Raw\0912Testday4_09.csv", "NumHeaderLines", 4);
pd = readtable(".\Process\Raw\0912Testday4_09.csv", "NumHeaderLines", 3);
col = readtable(".\Process\Raw\0912Testday4_09.csv", "Range", [1, 1, 2, 31], "ExpectedNumVariables");

ts = readtable("ExpTimestamp.csv");
ts = ts{:,1};

[h_process, m_process] = hms(table2array(pd(:,1)));

for i = 1:numel(ts)
    hour = str2double(ts{i}(1:2));
    min = str2double(ts{i}(4:5));
    
    ind_h = h_process == hour;
    ind_m = m_process == min;
    ind = (ind_h+ind_m) == 2;
    
end