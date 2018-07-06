import pickle

# create dictionary containing all your data
data = {'stim': np.array([1, 2, 3]), 'response': np.array([6, 2, 0])}

# save data in pickle format
with open('my_data.pickle', 'wb') as f:
    pickle.dump(data, f)

# open data from file
with open('my_data.pickle', 'rb') as f:
    new_data_variable = pickle.load(f)

#now the data will be in dictionary new_data_variable    
