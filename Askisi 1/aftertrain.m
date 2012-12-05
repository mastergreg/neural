clc;
load evaled_DATA_Single;
[Max1 , ind1] = max(evaled_DATA_Single(:));
[bFunc1, bFL1] = ind2sub(size(evaled_DATA_Single), ind1);
fprintf('\n \n Best method for Single hidden layer is %s \n first layer: %d \n with Accuracy: %f\n', char(trainFunctions(bFunc1)), bFL1*5,evaled_DATA_Single(bFunc1,bFL1));

load evaled_DATA_Double;
[Max2 , ind2] = max(evaled_DATA_Double(:));
[bFunc2, bFL2, bSL2] = ind2sub(size(evaled_DATA_Double), ind2);
fprintf('\n Best method for double is %s \n first layer: %d snd layer: %d \n with Accuracy: %f\n', char(trainFunctions(bFunc2)), bFL2*5, bSL2*5,evaled_DATA_Double(bFunc2,bFL2,bSL2));

if Max1>=Max2,
    bFunc = bFunc1;
    bFL = bFL1;
    fprintf('\n Best method: Single Hidden Layer with method " %s " \n First Layer: %d Accuracy: %f \n', char(trainFunctions(bFunc)), bFL*5,Max1);
else
    bFunc = bFunc2;
    bFL = bFL2;
    bSL = bSL2;
    fprintf('\n Best method: Double Hidden Layer with method " %s " \n First Layer: %d \n Second Layer: %d \n Accuracy: %f \n', char(trainFunctions(bFunc)), bFL*5,bSL,Max2);
end

%% Activation Functions for Best method
ActFunct = {'hardlim', 'tansig', 'logsig', 'purelin'};
for k =1:4
            fprintf('Act Funct is %s, NFL is %d, NSL is %d \n',char(ActFunct(k)),bFL*5,bSL*5);
            net = newff(TrainRemoved, EditedTrainDataTargets, [5*bFL 5*bSL], {char(ActFunct(k)) char(ActFunct(k)) char(ActFunct(k))} , char(trainFunctions(bFunc)));
            
            net.divideParam.trainRatio = 0.8;
            net.divideParam.valRatio = 0.2;
            net.divideParam.testRatio = 0;
            net.trainParam.epochs = 300;
            
            net = train(net, TrainRemoved, EditedTrainDataTargets);
            TestDataOutput = sim(net, TestRemoved);
            [Acc,~,~] = eval_Accuracy_Precision_Recall(TestDataOutput, TestDataTargets);
             Acc
end
