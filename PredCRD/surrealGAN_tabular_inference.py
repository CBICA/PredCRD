import numpy as np 
import pandas as pd
import pickle

import torch 
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from utils import get_surreal_GAN_loader, train_tabular_transformer, infer_tabular_transformer

### TabularTransformer - built for knowledge distillation approach
class TabularTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        transformer_layer = nn.TransformerEncoderLayer(d_model = d_model, nhead = nhead)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers = num_layers)
        self.fc = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        x = self.embedding(x)
        output = self.transformer_encoder(x)
        output = self.fc(output)
        
        return output 

# Train & Test the pipeline
def run_tabular_transformer_pipeline(model_dic_path='../roi_model', 
                                     data_path='../tabular_data/SurrealGAN-ALL.csv',
                                     nfolds=5):
    
    # model_dic_path = '../roi_model'
    
    ### read data
    data = pd.read_csv(data_path)
    #     - "train_test": test -> all istaging , train -> training for surrealGANs
    #     - feel free to change to "train" for small data 
    all_istaging_data = data[data['train_test'] == 'test']
    all_istaging_data['Sex'] = all_istaging_data['Sex'].apply(lambda x: 0 if x == 'M' else 1)
    
    transformer_result = []

    ### K-fold cross validation
    kf = KFold(n_splits=nfolds)
    MRID = all_istaging_data.MRID.unique()
    
    for i, (train_val_index, test_index) in enumerate(kf.split(MRID)):

        ### Train, Validation, Test creation 
        test_MRID = MRID[test_index]    
        train_val_MRID = MRID[train_val_index]
        train_MRID, val_MRID = train_test_split(train_val_MRID, test_size=0.2, random_state=42)

        train_df = all_istaging_data[all_istaging_data.MRID.isin(train_MRID)]
        val_df   = all_istaging_data[all_istaging_data.MRID.isin(val_MRID)]
        test_df  = all_istaging_data[all_istaging_data.MRID.isin(test_MRID)]

        ### ICV correction: divied by ICV, multiply by "MEAN of TRAIN ICV"
        col_name = train_df.filter(regex='^MUSE_(?:20[0-7]|1\d\d|[1-9]\d|[4-9])$').columns
        icv_mean_train = np.mean(train_df['DLICV'])

        train_df[col_name] = train_df.filter(regex='^MUSE_(?:20[0-7]|1\d\d|[1-9]\d|[4-9])$').div(train_df['DLICV'], axis = 0).mul(icv_mean_train, axis = 0)
        val_df[col_name]   = val_df.filter(regex='^MUSE_(?:20[0-7]|1\d\d|[1-9]\d|[4-9])$').div(val_df['DLICV'], axis = 0).mul(icv_mean_train, axis = 0)
        test_df[col_name]  = test_df.filter(regex='^MUSE_(?:20[0-7]|1\d\d|[1-9]\d|[4-9])$').div(test_df['DLICV'], axis = 0).mul(icv_mean_train, axis = 0)

        ### Standarize data 
        train_df_X = train_df.drop(columns = ['MRID','Study','train_test','r1','r2','r3','r4','r5']).reset_index(drop = True)
        val_df_X   = val_df.drop(columns = ['MRID','Study','train_test','r1','r2','r3','r4','r5']).reset_index(drop = True)
        test_df_X  = test_df.drop(columns =  ['MRID','Study','train_test','r1','r2','r3','r4','r5']).reset_index(drop = True)

        y_train = np.array(train_df[['r1','r2','r3','r4','r5']].reset_index(drop = True))
        y_val   = np.array(val_df[['r1','r2','r3','r4','r5']].reset_index(drop = True))
        y_test = np.array(test_df[['r1','r2','r3','r4','r5']].reset_index(drop = True))

        scaler = StandardScaler().fit(train_df_X)
        with open(f'{model_dic_path}/scaler.pkl','wb') as f:
            pickle.dump(scaler, f)

        X_train = scaler.transform(train_df_X)
        X_val   = scaler.transform(val_df_X)
        X_test  = scaler.transform(test_df_X)

        ### create dataloader for deep learning 
        train_loader = get_surreal_GAN_loader(X_train, y_train, batch_size = 32, shuffle = True)
        val_loader   = get_surreal_GAN_loader(X_val,   y_val,   batch_size = 32, shuffle = True)
        test_loader  = get_surreal_GAN_loader(X_test , y_test,  batch_size = 32, shuffle = False)

        ### define model, loss and optimizer 
        model = TabularTransformer(148, 32, 4, 4, 5).to(device)
        optimizer = optim.Adam(model.parameters(), lr = 3e-4, weight_decay = 5e-7)

        ### training 
        model, average_train_loss, average_val_loss = train_tabular_transformer(num_epochs = 100 , 
                                                                                model = model, 
                                                                                optimizer = optimizer, 
                                                                                train_loader = train_loader, 
                                                                                val_loader = val_loader, 
                                                                                model_dic_path = model_dic_path , 
                                                                                folder = i,
                                                                                device = device)
        
        np.save(f'{model_dic_path}/transformer_folder_{i}_train_loss.npy', average_train_loss)
        np.save(f'{model_dic_path}/transformer_folder_{i}_val_loss.npy', average_val_loss)
        
        ### testing 
        test_result, output_result = infer_tabular_transformer( model = model,
                                                                test_loader = test_loader, 
                                                                model_dic_path = model_dic_path, 
                                                                folder = i,
                                                                device = device)
  
        test_df[['R1','R2','R3','R4','R5']] = output_result
        test_df[['MRID','r1','r2','r3','r4','r5','R1','R2','R3','R4','R5']].to_csv(f'{model_dic_path}/transformer_Regressor_folder_{i}.csv', index = False)
        
        transformer_result.append(test_result)
    
    return transformer_result
# ###
# def test_tabular_transformer(input_path='../tabular_data/SurrealGAN-ALL.csv',
#                              output_path='../test_results',
#                              model_dic_path='../roi_model'):
    
    
#     ### read data
#     data = pd.read_csv(input_path)
#     #     - "train_test": test -> all istaging , train -> training for surrealGANs
#     #     - feel free to change to "train" for small data 
#     all_istaging_data = data[data['train_test'] == 'test']
#     all_istaging_data['Sex'] = all_istaging_data['Sex'].apply(lambda x: 0 if x == 'M' else 1)
    
#     MRID = all_istaging_data.MRID.unique()
    

#     ### Train, Validation, Test creation 
#     test_MRID = MRID

#     test_df  = all_istaging_data[all_istaging_data.MRID.isin(test_MRID)]

#     ### ICV correction: divied by ICV, multiply by "MEAN of TRAIN ICV"
#     col_name = train_df.filter(regex='^MUSE_(?:20[0-7]|1\d\d|[1-9]\d|[4-9])$').columns
#     icv_mean_train = np.mean(train_df['DLICV'])

#     test_df[col_name]  = test_df.filter(regex='^MUSE_(?:20[0-7]|1\d\d|[1-9]\d|[4-9])$').div(test_df['DLICV'], axis = 0).mul(icv_mean_train, axis = 0)

#     ### Standarize data 
#     test_df_X  = test_df.drop(columns =  ['MRID','Study','train_test','r1','r2','r3','r4','r5']).reset_index(drop = True)


#     y_test = np.array(test_df[['r1','r2','r3','r4','r5']].reset_index(drop = True))

#     scaler = StandardScaler().fit(train_df_X)

#     X_test  = scaler.transform(test_df_X)

#     ### create dataloader for deep learning 
#     test_loader  = get_surreal_GAN_loader(X_test , y_test,  batch_size = 32, shuffle = False)

#     ### define model, loss and optimizer 
#     model = TabularTransformer(148, 32, 4, 4, 5).to(device)
#     optimizer = optim.Adam(model.parameters(), lr = 3e-4, weight_decay = 5e-7)

    
#     ### testing 
#     test_result, output_result = infer_tabular_transformer( model = model,
#                                                             test_loader = test_loader, 
#                                                             model_dic_path = model_dic_path, 
#                                                             folder = i,
#                                                             device = device)

#     test_df[['R1','R2','R3','R4','R5']] = output_result
#     test_df[['MRID','r1','r2','r3','r4','r5','R1','R2','R3','R4','R5']].to_csv(f'{model_dic_path}/transformer_Regressor_folder_{i}.csv', index = False)
    
#     transformer_result.append(test_result)
    
#     return transformer_result

if __name__ == "__main__":
    
    # Use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Current Device is {device}')
    
    
    transformer_result = run_tabular_transformer_pipeline(model_dic_path='../roi_model', 
                                                          data_path='../tabular_data/SurrealGAN-ALL.csv', 
                                                          nfolds=5)

    ### get results
    transformer_final = np.mean(transformer_result)
    print(f'transformer Result: {transformer_result}, Average: {transformer_final}')

