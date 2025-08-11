import pickle as pkl

with open("data/sequence/CT_Shapes/CT-20_data.pkl", "rb") as f:
    data = pkl.load(f)
    
    onehot_fea = data['onehot_sequences']
    shape_fea = data['shapes']

    ex = shape_fea[0]

    print(ex)
    print(len(ex))