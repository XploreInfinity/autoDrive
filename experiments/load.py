import pickle
tdata=[]
with open('training_data-1.pkl','rb') as f:
    tdata=pickle.load(f)
print(tdata)
print(type(tdata))