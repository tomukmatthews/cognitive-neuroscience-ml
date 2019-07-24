function [RT,RT_tp,subj_score,subj_score_tp,time,Resp_lm,RT_lm,RT_lstd,id] = identify_alertness(EEG)
% Author: Part III Physics student
% Function which extracts the reaction times, local mean reaction times,
% subjective drowsiness scores, whether a response was recorded or missed
% such that the local mean reaction time can be scaled if so. Also extracts
% the indices of epochs with no RT's within so that these can be removed.
% Resp_lm demonstrates the number of missed trials (failed to react to a
% promt), the number corresponds the proportion of trials responded to, so
% a vlue of 1 corresponds to no missing responses.


n = EEG.trials;
m = size(EEG.data,2);
% Calculate the reaction times and the times at which they occur
% Generate empty arrays to fill
RT_tp = zeros(size(EEG.event,2),1);
RT = zeros(size(EEG.event,2),1);
Resp_tp = NaN(size(EEG.event,2),1);
Resp = NaN(size(EEG.event,2),1);
subj_score_tp = zeros(size(EEG.event,2),1);
subj_score = zeros(size(EEG.event,2),1);
time = zeros(n,1);
RT_lm = zeros(n,1);
RT_lstd = zeros(n,1);
Resp_lm = zeros(n,1);

for i = 1:1:(size(EEG.event,2)-2)

    eve_type = cell2mat({EEG.event(i).type});
    eve_type_next = cell2mat({EEG.event(i+1).type});
    eve_type_next2 = cell2mat({EEG.event(i+2).type});
%   Chck1 and chck2 return booleans for whether the event is a
%   white block and whether the next event is a reaction
    chck0 = strcmpi({'4','8','6','10'}, eve_type);
    chck1 = strcmpi({'16'}, eve_type_next);
    chck2 = strcmpi({'16'}, eve_type_next2);
    time_delay = 0;
    
    if any(chck0 ~= 0)
        
        if any(chck1 ~= 0)
            time_delay = (EEG.event(i+1).latency - EEG.event(i).latency)/EEG.srate;
            Resp_tp(i) = (cell2mat({EEG.event(i+1).latency})-1)/EEG.srate;
            if time_delay < 3.5
                RT_tp(i) = Resp_tp(i);
                RT(i) = RT_tp(i) - (cell2mat({EEG.event(i).latency})-1)/EEG.srate;
                Resp(i) = 1;
            else
                Resp(i) = 0;
            end
        
        elseif any(chck2 ~= 0)
            time_delay = (EEG.event(i+2).latency - EEG.event(i).latency)/EEG.srate;
            Resp_tp(i) = (cell2mat({EEG.event(i+2).latency})-1)/EEG.srate;
            if time_delay < 3.5
                RT_tp(i) = Resp_tp(i);
                RT(i) = RT_tp(i) - (cell2mat({EEG.event(i).latency})-1)/EEG.srate;
                Resp(i) = 1;
            else
                Resp(i) = 0; 
            end
        end
        
    end

    % Find subjective drowsiness scores
    keyset = {'60','80','100','120','140','160','180','200','220'};
%   Create arbitrary values for subjective drowsiness that are on same
%   order magnitude as reaction times for comparison
    valueset = [0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3];
    M = containers.Map(keyset,valueset);

    if ismember(cell2mat({EEG.event(i).type}),keyset)
%       dscore is subjective drowsiness score, time and score of these 
%       events stored in these vectors
        subj_score(i) = M(cell2mat({EEG.event(i).type}));
        subj_score_tp(i) = (cell2mat({EEG.event(i).latency})-1)/EEG.srate;
    end
end

% Now remove any elements of the vectors that were not filled
RT_tp = RT_tp(RT_tp ~= 0);
RT = RT(RT ~= 0);
idResp = find(isnan(Resp));
Resp_tp(idResp) = [];
Resp(idResp) = [];
subj_score_tp = subj_score_tp(subj_score_tp ~= 0);
subj_score = subj_score(subj_score ~=0);

for i = 1:n

    time(i) = (i-1/2) * m / (EEG.srate);
%   Find mean and standard deviation of the reaction times conatained
%   within the i'th epoch of the data.
    t_start = (1+(i-1)*m)/EEG.srate;
    t_end = i*m/EEG.srate;
    RT_indices = find(RT_tp>t_start & RT_tp<t_end);
    Resp_indices = find(Resp_tp>t_start & Resp_tp<t_end);
    Resp_lm(i) = mean(Resp(Resp_indices));
    RT_lm(i) = mean(RT(RT_indices));
    RT_lstd(i) = std(RT(RT_indices));

end

% Next remove any elements which are NaN
id1 = find(isnan(RT_lm));
id2 = find(isnan(RT_lstd));
id = unique([id1 id2]);



end
