clear all;close all;clc;
load dataSet

%% Define Number of each class
elements = min(sum(TrainDataTargets,2));
bar(sum(TrainDataTargets,2));
%% Same number of elemets for each class
EditedTrainData = [];
EditedTrainDataTargets = [];

for i=1:12
    imalndex = find(TrainDataTargets(i,:), elements);
    EditedTrainData = [EditedTrainData TrainData(:,imalndex)];
    EditedTrainDataTargets = [ EditedTrainDataTargets TrainDataTargets(:, imalndex)];
end
%% Shuffle the data
suffle = randperm(12*elements);
EditedTrainData = EditedTrainData(:, suffle);
EditedTrainDataTargets = EditedTrainDataTargets(:, suffle);

%% Remove Constant Rows of MPEG-7 charasteristics of Train and Data
[TrainRemoved, settings] = removeconstantrows(EditedTrainData);
TestRemoved = removeconstantrows('apply', TestData, settings);

%% Remove Correlated components (Rows) of Train and Data
[TrainRemoved, ps] = processpca(TrainRemoved, 0.001);
TestRemoved = processpca('apply', TestRemoved, ps);

trainFunctions = {'traingdx', 'trainlm', 'traingd', 'traingda'};
%% One hidden Layer
evaled_DATA_Single = zeros(4,6);
for k = 1:4,
    for i = 1:6,
        fprintf('method is %s, NFL is %d \n',char(trainFunctions(k)),i*5);
        net = newff(TrainRemoved, EditedTrainDataTargets, 5*i, {'tansig' 'purelin'} , char(trainFunctions(k)));
        net.divideParam.trainRatio = 0.8;
        net.divideParam.valRatio = 0.2;
        net.divideParam.testRatio = 0;

        net.trainParam.epochs = 300;
        net = train(net, TrainRemoved, EditedTrainDataTargets);
        TestDataOutput = sim(net, TestRemoved);
        [Acc,~,~] = eval_Accuracy_Precision_Recall(TestDataOutput, TestDataTargets);
        evaled_DATA_Single(k,i) = Acc;
    end
end
save evaled_DATA_Single;

load evaled_DATA_Single;
[~ , ind1] = max(evaled_DATA_Single(:));
[bFunc1, bFL1] = ind2sub(size(evaled_DATA_Single), ind1);
fprintf('\n \n Best method is %s \n first layer: %d \n with Accuracy: %f\n', char(trainFunctions(bFunc1)), bFL1*5,evaled_DATA_Single(bFunc1,bFL1));

%% Two Hidden Layers
evaled_DATA_Double = zeros(4,6,6);
for k = 1:4,
    for i = 1:6,
        for j = 1:6,
            fprintf('method is %s, NFL is %d, NSL is %d \n',char(trainFunctions(k)),i*5,j*5);
            net = newff(TrainRemoved, EditedTrainDataTargets, [5*i 5*j], {'tansig' 'tansig' 'purelin'} , char(trainFunctions(k)));
            
            net.divideParam.trainRatio = 0.8;
            net.divideParam.valRatio = 0.2;
            net.divideParam.testRatio = 0;
            net.trainParam.epochs = 300;
            
            net = train(net, TrainRemoved, EditedTrainDataTargets);
            TestDataOutput = sim(net, TestRemoved);
            [Acc,~,~] = eval_Accuracy_Precision_Recall(TestDataOutput, TestDataTargets);
            evaled_DATA_Double(k,i,j) = Acc;
            
        end
    end
end

save evaled_DATA_Double;

load evaled_DATA_Double;
[~ , ind2] = max(evaled_DATA_Double(:));
[bFunc2, bFL2, bSL2] = ind2sub(size(evaled_DATA_Double), ind2);
fprintf('\n \n Best method is %s \n first layer: %d snd layer: %d \n with Accuracy: %f\n', char(trainFunctions(bFunc2)), bFL2*5, bSL2*5,evaled_DATA_Double(bFunc2,bFL2,bSL2));

