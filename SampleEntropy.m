% Author: Tom Matthews, Part III Cambridge Physics student
% Function to extract the sample entropy from every epoch as a feature,
% outputs the feature if its correlation with reaction times are
% statistically significant according to signf_thresh.

function SmpEntropy = SampleEntropy(EEG,id,RT_lm,signf_thresh)


Ntrls = 1:EEG.trials;
SmpEntropy = [];

dim = 2;        % Embedding dimension
r = 0.2;        % Tolerance

for idx = Ntrls
    data = squeeze(EEG.data(1,:,idx));
    SmpEntropy(idx,1) = SampEn(data,dim,r);
end

SmpEntropy(id)=[];

[rho, pval] = corr(RT_lm,SmpEntropy,'type','Spearman');

if pval > signf_thresh
    SmpEntropy = [];
end

end

