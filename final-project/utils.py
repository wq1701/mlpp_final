def load_makematrix(path_in_str, split = False, test_ratio = 0.1, random_seed = 42): 
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    data_full = pd.read_csv(path_in_str)
    
    
    if split == True: 
        data_train, data_test = train_test_split(data_full, test_size = test_ratio, random_state = random_seed)
        data_train_matrix = data_train.pivot(index='user_id', columns='anime_id', values='rating')
    
    return data_full, data_train, data_test, data_train_matrix
    
    

