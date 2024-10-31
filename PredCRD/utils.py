import numpy as np 

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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