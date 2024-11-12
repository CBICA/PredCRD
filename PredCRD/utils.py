import numpy as np 

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

### data loader 
def get_surreal_GAN_loader(X, y, batch_size = 32, shuffle = True):    
    X_tensor = torch.tensor(X.astype(np.float32))
    y_tensor = torch.tensor(y.astype(np.float32))

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
    
    return loader

def train_tabular_transformer(num_epochs, 
                              model, 
                              optimizer, 
                              train_loader, 
                              val_loader, 
                              model_dic_path, 
                              folder,
                              device):
    
    '''
        input: 
            num_epochs: number of epoch 
            model: TabularTransformer
            optimizer: optimizer for training (Adam)
            train_loader: training dataloader 
            val_loader: validation dataloader
            model_dic_path: where to save the model
            folder: 0,1,2,3,4 for 5 folder cross-validation
            device: cpu or cuda
        output:
            model, average_train_loss, average_val_loss
    '''

    loss_fn = nn.L1Loss(reduction='mean')
    average_train_loss = []
    average_val_loss = [] 
    
    min_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        step = 0 
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            step += inputs.shape[0]
            
        average_train_loss.append( train_loss / step)
            
        model.eval()
        test_loss = 0
        test_step = 0 
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

                test_loss += loss.item()
                test_step += inputs.shape[0]
                    
            average_val_loss.append(test_loss / test_step )

        print(f'average train loss: {average_train_loss[-1]}, average val loss : {average_val_loss[-1]}')
            
        if test_loss / test_step < min_val_loss:
            min_val_loss = test_loss / test_step
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, 
                        f'{model_dic_path}/ROI_Transformer_best_{folder}.pth')
            
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, 
               f'{model_dic_path}/ROI_Transformer_last_{folder}.pth')

            
    return model, average_train_loss, average_val_loss

def infer_tabular_transformer(model,
                              test_loader, 
                              model_dic_path, 
                              folder,
                              device):
    
    '''
        input: 
            model: initialize a TabularTransformer to load the weight
            test_loader: data loader for test dataset
            model_dic_path: path to store your model
            folder: 0,1,2,3,4 for 5 fold cross-validation
            device: cpu or cuda
        output:
            test_result_MAE, all_test_result in shape (number_of_test_data, 5)
    
    '''
    
    
    checkpoint = torch.load(f'{model_dic_path}/ROI_Transformer_best_{folder}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    test_loss = 0
    test_step = 0 

    loss_fn = nn.L1Loss(reduction='mean')

    output_result = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            output_result.append(outputs)
            loss = loss_fn(outputs, targets)

            test_loss += loss.item()
            test_step += inputs.shape[0]
                
        result = test_loss / test_step
        print(f'average test loss : {result}')
    
    return result, torch.concat(output_result,axis = 0).detach().cpu().numpy()


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

    # Use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Current Device is {device}')
    
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