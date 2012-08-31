clear all;close all;clc;
load dataSet



elements = min(sum(TrainDataTargets,2));



EditedTrainData = [];
EditedTrainDataTargets = [];

for i=1:12
    imalndex = find(TrainDataTargets(i,:), elements);
    EditedTrainData = [EditedTrainData TrainData(:,imalndex)];
    EditedTrainDataTargets = [ EditedTrainDataTargets TrainDataTargets(:, imalndex)];
end


suffle = randperm(12*elements);


EditedTrainData = EditedTrainData(:, suffle);

EditedTrainDataTargets = EditedTrainDataTargets(:, suffle);




[TrainRemoved, settings] = removeconstantrows(EditedTrainData);

%[TestRemoved, settings] = removeconstantrows('apply', TestData, settings);

[TrainRemoved, ps] = processpca(TrainRemoved, 0.001);

%[TestRemoved, ps] = processpca('apply', TestRemoved, ps);


