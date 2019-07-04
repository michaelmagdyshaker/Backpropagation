from builtins import list

import numpy as np
import random
import math

def ReadAndShuffle():
    # firstC=C1
    # secondC=C2
    # ThirdC=C3
    # train="True"
    File = open("IrisData.txt")
    Lines = File.readlines()
    i = 0
    ListOfRecords = list()
    # firstThirty=0
    for EachLine in Lines:
        if i == 0:
            i = i + 1
        else:
            X1 = (EachLine.split(','))[0]
            X2 = (EachLine.split(','))[1]
            X3 = (EachLine.split(','))[2]
            X4 = (EachLine.split(','))[3]
            Label = (EachLine.split(','))[4]
            if Label == 'Iris-setosa\n':
                Cluster = "C1"
            if Label == 'Iris-versicolor\n':
                Cluster = 'C2'
            if Label == 'Iris-virginica\n':
                Cluster = 'C3'

            Record = dict(X1=X1, X2=X2, X3=X3, X4=X4, Label=Cluster, Index=i)
            ListOfRecords.append(Record)
            mod=i%50
            i = i + 1
            Line=Record["X1"]+','+ Record["X2"]+','+ Record["X3"]+','+ Record["X4"]+','+ Record["Label"]+'\n'#  to change the format of the dict
            if ((mod <31 and mod !=0 )):
                with open("TrainFile.txt", "a") as f:
                    f.writelines(Line)
            elif(mod >=31 or mod ==0  ):
                with open("TestFile.txt", "a") as f:
                    f.writelines(Line)
    with open("TrainFile.txt") as f:
        lines = f.readlines()
    random.shuffle(lines)
    with open("TrainFile.txt", "w") as f:
        f.writelines(lines)
    with open("TestFile.txt") as f:
           lines = f.readlines()
    random.shuffle(lines)
    with open("TestFile.txt", "w") as f:
            f.writelines(lines)

def ChooseFeatures (Filename):

    FirstFeatureList=list()
    SecodFeatureList=list()
    ThirdFeatureList=list()
    FourthFeatureList=list()
    ThisnewLabel=[]
    LabelList=list()
    File=open(Filename)
    Lines = File.readlines()
    for eachline in Lines:
        X1 = (eachline.split(','))[0]
        X2 = (eachline.split(','))[1]
        X3 = (eachline.split(','))[2]
        X4 = (eachline.split(','))[3]
        ThisLabel=(eachline.split(','))[4]
        if ThisLabel=="C1" + "\n":
            ThisnewLabel=([0,0,1])
        elif ThisLabel=="C2" + "\n":
            ThisnewLabel=([0,1,0])
        elif ThisLabel=="C3" + "\n":
            ThisnewLabel=([1,0,0])
        FirstFeatureList.append(X1)
        SecodFeatureList.append(X2)
        ThirdFeatureList.append(X3)
        FourthFeatureList.append(X4)

        LabelList.append(ThisnewLabel)

    # FourthFeatureList = np.array(FourthFeatureList)
    LabelList = np.array(LabelList)
    return (FirstFeatureList,SecodFeatureList,ThirdFeatureList,FourthFeatureList,LabelList)

def InputMatrixConstructor(F1,F2,F3,F4,BooleanBias):
    Input_Matrix=np.empty([90,5],dtype=float)
    if BooleanBias == True:
        Bias = 1
    elif BooleanBias == False:
        Bias=0
    for i in range(0,90):
        Input_Matrix[i][0] = Bias
        Input_Matrix[i][1] = F1[i]
        Input_Matrix[i][2] = F2[i]
        Input_Matrix[i][3] = F3[i]
        Input_Matrix[i][4] = F4[i]

    return Input_Matrix

def WeightMatrixConstructor(RowsNum):
    WeightMatrix=np.random.rand(RowsNum,1)
    return WeightMatrix

def Reshape(F1List,F2List,F3List,F4List):
    F1List = np.array(F1List)
    F2List = np.array(F2List)
    F3List = np.array(F3List)
    F4List = np.array(F4List)
    F1List = F1List.reshape(90, 1)
    F2List = F2List.reshape(90, 1)
    F3List = F3List.reshape(90, 1)
    F4List = F4List.reshape(90, 1)
    return (F1List,F2List,F3List,F4List)

def Sigmoid(Sum):
    SigomidOutput=(1/(1+math.exp(2*Sum)))
    return SigomidOutput

def Tangent(Sum):
    TangentOutput=((1-math.exp(-2*Sum))/(1+math.exp(-2*Sum)))
    return TangentOutput

def DashSigmoid(Sum):
    OutputDash=float((Sum*(1-Sum)))
    return OutputDash

def DashTangent(Sum):
    OutputDash=((1-Sum)*(1+Sum))
    return OutputDash

def UpdateWeight(Weight, Error, LearninRate, Input):
    Weight = np.array(Weight)
    UpdatedWeight = np.empty([len(Weight), 1])
    for i in range(0, len(Weight)):
        UpdatedWeight[i][0] = (Weight[i][0] + (LearninRate * Error * Input[0][i]))

    return UpdatedWeight

def ForwardPropagation(InputMatrix,WeightMat,ActivatioinFunction):
    Sum=np.dot(InputMatrix,WeightMat)
    if(ActivatioinFunction=="Sigmoid"):
        Sum=Sigmoid(Sum)
    elif(ActivatioinFunction=="Hyperbolic_Tangent"):
        Sum=Tangent(Sum)
    return Sum

def BackWardPropagation(WeightMat,ErrorListMat,ActivatioinFunction,inputvalue):
    BackSum=np.dot(ErrorListMat,WeightMat)
    if(ActivatioinFunction=="Sigmoid"):
        BackSum=BackSum*(DashSigmoid(inputvalue))
    elif(ActivatioinFunction=="Hyperbolic_Tangent"):
        BackSum=BackSum*(DashTangent(inputvalue))
    return BackSum

def UpdateWeightDash(Weight,Error,LearninRate,Input,Bais):
    baisinput=list()
    Weight=np.array(Weight)
    UpdatedWeight =np.empty([len(Weight),1])
    # if Bais==True:
    #     baisinput.append(1)
    # elif Bais==False:
    #     baisinput.append(0)
    # baisinput.append(Input)
    for i in range(0,len(Weight)):
        UpdatedWeight[i][0]=(Weight[i][0]+(LearninRate*Error*Input[i]))

    return UpdatedWeight

def AnotherInputMatrixConstructor(F1,F2,F3,F4,BooleanBias):
    AnotherInputMatrix=np.empty([60,5],dtype=float)
    if BooleanBias == True:
        Bias = 1
    elif BooleanBias == False:
        Bias=0
    for i in range(0,60):
        AnotherInputMatrix[i][0] = Bias
        AnotherInputMatrix[i][1] = F1[i]
        AnotherInputMatrix[i][2] = F2[i]
        AnotherInputMatrix[i][3] = F3[i]
        AnotherInputMatrix[i][4] = F4[i]
    return AnotherInputMatrix

def OutputConstructor(Output1,Output2,Output3):
    PredictedOutput=list()
    Vote = max(Output1, Output2, Output3)
    if (Vote == Output1):
        Output1 = 1
        Output2 = 0
        Output3 = 0
    elif (Vote == Output2):
        Output1 = 0
        Output2 = 1
        Output3 = 0
    else:
        Output1 = 0
        Output2 = 0
        Output3 = 1
    PredictedOutput.append(Output1)
    PredictedOutput.append(Output2)
    PredictedOutput.append(Output3)
    PredictedOutput=np.array(PredictedOutput)
    return PredictedOutput

def BackPropagationAlgorithm(InputMatrix,Network_Array,Hidden_input,eta_input,epoch_input,BiasBoolian,activation_function_input,TargetList):
    WeightListForOneNode=list()
    WeightListForOneLayer=list()
    Netlist=list()
    vlist=list()
    inverseweightforonenode=list()#### s
    CurrentWeights=list()
    ErrorBack=list()
    errorbackfinal=list()
    VirtualList=list()# ba5od feha values al sum lkol al nodes allly fe al hdden layers w basfrha tany
    ErrorList=list()
    SecondWeightList=[]

    for i in range (0,epoch_input):
        for j in range (0,90):
            Netlist=list()
            for hid in range(0,Hidden_input+1):
                InputRecordMat = np.empty([1, 5])
                InputRecordMat[0][0] = InputMatrix[j][0]
                InputRecordMat[0][1] = InputMatrix[j][1]
                InputRecordMat[0][2] = InputMatrix[j][2]
                InputRecordMat[0][3] = InputMatrix[j][3]
                InputRecordMat[0][4] = InputMatrix[j][4]
                if (hid == 0):
                    if (BiasBoolian):
                        vlist.append(1)
                    else:
                        vlist.append(0)
                    for neu in range(0,Network_Array[hid]):
                        if (j==0):
                            WeightforInputMatrix = WeightMatrixConstructor(5)
                        else:
                            # updated weight############################################################################################################

                            WeightforInputMatrix=UpdateWeight(WeightListForOneLayer[hid][neu],errorbackfinal[Hidden_input-hid-1][Network_Array[hid]-neu-1],eta_input,InputRecordMat)

                        Sum = ForwardPropagation(InputRecordMat, WeightforInputMatrix, activation_function_input)
                        vlist.append(Sum)
                        WeightListForOneNode.append(WeightforInputMatrix)
                    SecondWeightList.append(WeightListForOneNode)

                    Netlist.append(vlist)
                    vlist=list()
                    WeightListForOneNode=list()
                    VirtualList=list()
                if(hid>0 and hid!=(Hidden_input)):
                    if (BiasBoolian):
                        VirtualList.append(1)
                    else:
                        VirtualList.append(0)
                    for neu in range(0, Network_Array[hid]):
                        if (j == 0):
                            WeightforInputMatrix = WeightMatrixConstructor(len(Netlist[hid - 1]))
                        else:
                            WeightforInputMatrix = UpdateWeightDash(WeightListForOneLayer[hid][neu],errorbackfinal[Hidden_input-hid- 1][0][Network_Array[hid]-neu-1],eta_input,Netlist[hid-1],BiasBoolian)
                        Sum = ForwardPropagation(Netlist[hid-1], WeightforInputMatrix, activation_function_input)
                        VirtualList.append(Sum)
                        WeightListForOneNode.append(WeightforInputMatrix)
                    SecondWeightList.append(WeightListForOneNode)
                    WeightListForOneNode = list()
                    Netlist.append(VirtualList)
                    VirtualList=list()

                if (hid==(Hidden_input)):
                    if (BiasBoolian):
                       VirtualList.append(1)
                    else:
                       VirtualList.append(0)
                    for neu in range(0,3):
                        if(j==0):
                            WeightforInputMatrix = WeightMatrixConstructor(len(Netlist[hid-1]))
                        else:
                            WeightforInputMatrix=UpdateWeightDash(WeightListForOneLayer[hid][neu],errorbackfinal[Hidden_input-hid-1][3-neu-1],eta_input,Netlist[hid-1],BiasBoolian)
                        FinalSum = ForwardPropagation(Netlist[hid-1], WeightforInputMatrix, activation_function_input)
                        #VirtualList.append(FinalSum)
                        WeightListForOneNode.append(WeightforInputMatrix)
                        if(activation_function_input=="Sigmoid"):
                            Error = ((TargetList[j][neu] - FinalSum) * DashSigmoid(FinalSum))
                        else:
                            Error = ((TargetList[j][neu]-FinalSum ) * DashTangent(FinalSum))

                        ErrorList.append(Error)
                    SecondWeightList.append(WeightListForOneNode)
                    WeightListForOneNode = list()
            errorbackfinal=list()
####################################################################backward after getting last  error

            for invhid in range(Hidden_input, 0, -1):
                if (invhid == Hidden_input):
                    for invneu in range(Network_Array[invhid-1], 1, -1):
                        for wigt in range((len(SecondWeightList[invhid])-1),0,-1):
                            for invneuwi in range(0,invneu+1):
                                temp=SecondWeightList[invhid][invneuwi][wigt]
                                inverseweightforonenode.append(temp)

                            errorForonenode=BackWardPropagation(inverseweightforonenode,ErrorList,activation_function_input,Netlist[invhid-1][invneu])
                            ErrorBack.append(errorForonenode)
                            inverseweightforonenode=list()
                    errorbackfinal.append(ErrorBack)

                    ErrorBack=list()
                elif(invhid<Hidden_input):
                    for invneu in range(Network_Array[invhid-1], 0, -1):
                        for wigt in range((len(SecondWeightList[invhid])-1),0,-1):
                            for invneuwi in range(0,Network_Array[invhid-1]-1):
                                temp=SecondWeightList[invhid][invneuwi][invneu]
                                inverseweightforonenode.append(temp)
                            errorbackfinal[wigt - 1]=np.array(errorbackfinal[wigt-1])
                            x,y=np.shape(errorbackfinal[wigt-1])
                            z=max(x,y)
                            errorbackfinal[wigt - 1] = errorbackfinal[wigt-1].reshape(1,z)
                            # inverseweightforonenode=np.array(inverseweightforonenode)
                            errorForonenode=BackWardPropagation(inverseweightforonenode,errorbackfinal[wigt-1],activation_function_input,Netlist[invhid-1][invneu])
                            ErrorBack.append(errorForonenode)
                            inverseweightforonenode=list()
                    errorbackfinal.append(ErrorBack)
                    errorbackfinal[len(errorbackfinal)- 1] = np.array(errorbackfinal[len(errorbackfinal)-1])
                    # x, y = np.shape(errorbackfinal[len(errorbackfinal)-1])
                    # z = max(x, y)
                    # errorbackfinal[len(errorbackfinal) - 1]=errorbackfinal[len(errorbackfinal)-1].reshape(1, z)


                    ErrorBack = list()
                    ErrorList = list()
                    WeightListForOneLayer=SecondWeightList
                    SecondWeightList=[]
                    # WeightNPArray=np.array(WeightListForOneLayer)
                    # WeightListForOneLayer=np.array(WeightListForOneLayer)
                    # W
                    # WeightListForOneLayer=WeightListForOneLayer[j*(Hidden_input+1):j+Hidden_input+1]
    return WeightListForOneLayer

def TesBackProbagation(TestInputMatrix,FinalWeight,TestLabels,Network_Array,Hidden_input,activation_function_input,BoolianBias):
    C1Right = 0
    C1Wrong = 0
    C2Right = 0
    C2Wrong = 0
    C3Right = 0
    C3Wrong = 0
    Accuracy = 0

    PredictedOutput=list()
    Templist=list()
    for Sample in range(0 , 60):
        NetValues = list()
        for HidL in range(0, Hidden_input+1):
            InputRecordMat = np.empty([1, 5])
            InputRecordMat[0][0] = TestInputMatrix[Sample][0]
            InputRecordMat[0][1] = TestInputMatrix[Sample][1]
            InputRecordMat[0][2] = TestInputMatrix[Sample][2]
            InputRecordMat[0][3] = TestInputMatrix[Sample][3]
            InputRecordMat[0][4] = TestInputMatrix[Sample][4]
            if (HidL==0):
                if (BoolianBias):
                    Templist.append(1)
                else:
                    Templist.append(0)
                for neu in range ( 0 , Network_Array[HidL]):
                    Sum = ForwardPropagation(InputRecordMat,FinalWeight[HidL][neu], activation_function_input)
                    Templist.append(Sum)
                NetValues.append(Templist)
                Templist=list()
            if ( HidL > 0 and HidL != Hidden_input):
                if (BoolianBias):
                    Templist.append(1)
                else:
                    Templist.append(0)
                for neu in range (0, Network_Array[HidL]):
                    Sum = ForwardPropagation(NetValues, FinalWeight[HidL][neu], activation_function_input)
                    Templist.append(Sum)
                NetValues = list()
                NetValues.append(Templist)
                Templist=[]
            if (HidL == Hidden_input):
                Output1 = ForwardPropagation(NetValues, FinalWeight[HidL][0], activation_function_input)
                Output2 =ForwardPropagation(NetValues, FinalWeight[HidL][1], activation_function_input)
                Output3 =ForwardPropagation(NetValues, FinalWeight[HidL][2], activation_function_input)
                PredictedOutput=OutputConstructor(Output1,Output2,Output3)
                if (np.array_equal(PredictedOutput,TestLabels[Sample]) and np.array_equal(TestLabels[Sample],([1, 0, 0]))):
                    Accuracy += 1
                    C1Right += 1
                elif (np.array_equal(PredictedOutput,TestLabels[Sample]) and np.array_equal(TestLabels[Sample],([0, 1, 0]))):
                    Accuracy += 1
                    C2Right += 1
                elif (np.array_equal(PredictedOutput,TestLabels[Sample]) and np.array_equal(TestLabels[Sample],([0, 0, 1]))):
                    Accuracy += 1
                    C3Right += 1
                elif (not(np.array_equal(PredictedOutput,TestLabels[Sample]))and np.array_equal(TestLabels[Sample],([1, 0, 0]))):
                    C1Wrong += 1
                elif(not(np.array_equal(PredictedOutput,TestLabels[Sample]))and np.array_equal(TestLabels[Sample],([0, 1, 0]))):
                    C2Wrong += 1
                elif (not(np.array_equal(PredictedOutput,TestLabels[Sample]))and np.array_equal(TestLabels[Sample],([0, 0, 1]))):
                    C3Wrong += 1
    Confmat=np.empty([3,2])
    Confmat[0][0]=C1Right
    Confmat[0][1]=C1Wrong
    Confmat[1][0]=C2Right
    Confmat[1][1]=C2Wrong
    Confmat[2][0]=C3Right
    Confmat[2][1]=C3Wrong
    return ((Accuracy/60)*100),Confmat

def main(hidden_input, neurons_input, eta_input, epoch_input, BiasBoolian, activation_function_input):
    # Epochs=10############################################################## update this!!!!!
    open("TrainFile.txt", 'w').close()
    open("TestFile.txt", 'w').close()
    Network_Array = neurons_input

    # For Train
    ReadAndShuffle()
    F1List, F2List, F3List, F4List, TrainLabelList = ChooseFeatures("TrainFile.txt")
    F1List, F2List, F3List, F4List = Reshape(F1List, F2List, F3List, F4List)
    InputMatrix = InputMatrixConstructor(F1List, F2List, F3List, F4List, BiasBoolian)
    FinalWeight = BackPropagationAlgorithm(InputMatrix, Network_Array, hidden_input, eta_input, epoch_input,
                                           BiasBoolian, activation_function_input, TrainLabelList)
    FinalWeight = np.array(FinalWeight)
    # FinalWeight=FinalWeight[]

    # For Test
    F1T, F2T, F3T, F4T, TestLabelList = ChooseFeatures("TestFile.txt")
    TestInputMatrix = AnotherInputMatrixConstructor(F1T, F2T, F3T, F4T, BiasBoolian)
    Accuracy, ConfusionMatrix = TesBackProbagation(TestInputMatrix, FinalWeight, TestLabelList, Network_Array,
                                                   hidden_input, activation_function_input, BiasBoolian)
    print('Accuracy=', int(Accuracy), '%', '\n', 'Confusion Matrix is:\n', ConfusionMatrix, '\n')

# main(2,3,0.1,10,True,"Sigmoid")
