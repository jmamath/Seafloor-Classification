% Loading the echogram, in fact each ping has 2581 rows for the data from 2011
% and 2567 rows from the 2015's data, so we select 2550 rows for each data set
% so that we can feed both of them to the same model.
% Thus we ignore the first 31 rows for the 2011's dataset and the first 17 rows from 
% the 2015's dataset.

%% 1 - Loading the data

% 1.1 - Loading the echogram
cd /Volumes/AWA_bck/Preprocessed/system_2011
data = matfile('Dataset')
Echogram = data.Echogram18(32:end,:);

% 1.2 - Getting the labels
Bottom = data.Bottom();
CleanBottom = data.CleanBottom(1,:);

%% 2 - Preprocessing the data
% Preprocessing the echogram removing echos with no bottom
% Getting the indices of echos with bottom
% This is a visual criterion, check the presentation file at the preprocessing slide.
index_Bottom = max(Echogram > -32); 
index_NoBottom = ~ index_Bottom;


EchogramWithBottom = Echogram(:,index_Bottom);
Bottom = Bottom(index_Bottom);
CleanBottom = CleanBottom(index_Bottom);

%% 3 - Labeling the data
% 1.1 CleanBottom approximately equal to Bottom i.e bottom clear
index_BottomClear = ((CleanBottom - Bottom).^2) <= 11 ;

% 1.2 CleanBottom far from Bottom i.e diffuse bottom
index_BottomDiffus = ~index_BottomClear;

class_BottomClear = index_BottomClear * 1;
class_BottomDiffus = index_BottomDiffus* 2;

label =  class_BottomClear + class_BottomDiffus;

save EchogramPreprocessed EchogramWithBottom label -v7.3
