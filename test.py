import h5py

# Open the HDF5 file in read mode
with h5py.File('model/BCMA/cnn_bilstm/model.h5', 'r') as f:
    # List top-level keys (groups and datasets)
    for key in f.keys():
        print(f"{key} : {f[key]}")

# print(float('0.03673278252655676'))
