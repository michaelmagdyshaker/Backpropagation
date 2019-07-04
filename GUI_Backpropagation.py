from tkinter import *
import tkinter.ttk as ttk
import MainAlgo
top = Tk()
top.title("Multi Layer")
top.geometry("450x450")
def get_entries():
    #print (feature1_input,feature2_input,class1_input,class2_input)
    hidden_input=int(Hiddenlayer_textfield.get('1.0',END))
    neurons_input=str(neurons_textfield.get('1.0',END))
    lines = neurons_input.strip().split("\n")
    #print(lines)
    eta_input=float(eta_textfield.get('1.0',END))
    epoch_input=int(epoch_textfield.get('1.0',END))
    if (var1.get()==1):
        baisboolen=True
    elif(var2.get()==1):
        baisboolen=False
    activation_function_input=box.get()

    # print (hidden_input,neurons_input,eta_input,epoch_input,baisboolen,activation_function_input)

    Run(hidden_input,neurons_input,eta_input,epoch_input,baisboolen,activation_function_input)


#Label
HiddenLayer=Label(top,text="Enter Number of Hidden Layers",font=("Times New Roman",12),justify=LEFT)
HiddenLayer.grid(column=0,row=0)

#textbox
Hiddenlayer_textfield=Text(top,width=15,height=1)
Hiddenlayer_textfield.grid(column=1,row=0)



#Label
neurons=Label(top,text="Enter Number of Neuorns",font=("Times New Roman",12),justify=LEFT)
neurons.grid(column=0,row=1)

#textbox
neurons_textfield=Text(top,width=15,height=1)
neurons_textfield.grid(column=1,row=1)


def Run(hidden_input,neurons_input2,eta_input,epoch_input,baisboolen,activation_function_input):
    neurons_input2 = ([3, 2])
    hidden_input=2
    MainAlgo.main(hidden_input, neurons_input2, eta_input, epoch_input, baisboolen, activation_function_input)


#label for learning rate
eta=Label(top,text="Enter Learning Rate",font=("Times New Roman",12),justify=LEFT)
eta.grid(column=0,row=2)
#textbox
eta_textfield=Text(top,width=15,height=1)
eta_textfield.grid(column=1,row=2)


#label for number of epochs
epoch=Label(top,text="Enter Epochs Number",font=("Times New Roman",12),justify=LEFT)
epoch.grid(column=0,row=3)
#textbox
epoch_textfield=Text(top,width=15,height=1)
epoch_textfield.grid(column=1,row=3)


#label for bais
bais=Label(top,text="Select Bias Or No Bias",font=("Times New Roman",12),justify=LEFT)
bais.grid(column=0,row=4)
#checkbox
var1 = IntVar()
Checkbutton(top, text="Bias", variable=var1).grid(column=1,row=4)
var2 = IntVar()
Checkbutton(top, text="No Bias", variable=var2).grid(column=2,row=4)




activation_function=Label(top,text="Select activation function",font=("Times New Roman",12),justify=LEFT)
activation_function.grid(column=0,row=5)


#Combobox
box=StringVar()
activation_functioncb=ttk.Combobox(top,textvariable=box,values=("Sigmoid","Hyperbolic_Tangent"))
activation_functioncb.set("Sigmoid")
activation_functioncb.grid(column=1,row=5)



button=Button(top,text="RUN",command=get_entries)
button.grid(column=0,row=6)


top.mainloop()

