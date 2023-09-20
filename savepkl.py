import pickle

def save(name,val):
    with open('./saved_data/'+name+'.pkl','wb') as file:
        pickle.dump(val,file)
def load(name):
    with open('./saved_data/'+name+'.pkl','rb') as file:
        m=pickle.load(file)
    return m

