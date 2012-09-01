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

[TestRemoved, settings] = removeconstantrows('apply', TestData, settings);



[TrainRemoved, ps] = processpca(TrainRemoved, 0.001);

[TestRemoved, ps] = processpca('apply', TestRemoved, ps);

%evaled_DATA = [];
for i = 1:6,
    for j = 1:6,
        j*5
        net = newff(TrainRemoved, EditedTrainDataTargets, [5*i 5*j]);
        net.divideParam.trainRatio = 0.8;
        net.divideParam.valRatio = 0.2;
        net.divideParam.testRatio = 0;

        net.trainParam.epochs = 300;
        net = train(net, TrainRemoved, EditedTrainDataTargets);
        TestDataOutput = sim(net, TestRemoved);
        a = eval_Accuracy_Precision_Recall(TestDataOutput, TestDataTargets);
        evaled_DATA(i,j,:) = a;
    end
end









