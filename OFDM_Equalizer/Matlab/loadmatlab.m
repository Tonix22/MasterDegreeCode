folder = '../Data';  % You specify this!
addpath(folder)  

fullMatFileName = fullfile(folder,  'v2v80211p_LOS.mat');
if ~exist(fullMatFileName, 'file')
  message = sprintf('%s does not exist', fullMatFileName);
  uiwait(warndlg(message));
else
  LOS = load(fullMatFileName);
  LOS = LOS.vectReal32b;
end


fullMatFileName = fullfile(folder,  'v2v80211p_NLOS.mat');
if ~exist(fullMatFileName, 'file')
  message = sprintf('%s does not exist', fullMatFileName);
  uiwait(warndlg(message));
else
  NLOS = load(fullMatFileName);
  NLOS = NLOS.vectReal32b;
end
