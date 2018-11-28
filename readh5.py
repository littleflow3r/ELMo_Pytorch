import h5py

data = h5py.File('output/avg', 'r')
fkey = list(data.keys())

for k in data.keys():
    print (k)
    print (data[k].value)



# Get the data
#data = list(f[a_group_key])
