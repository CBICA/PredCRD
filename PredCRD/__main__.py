import warnings
import argparse
import time
import os
from pathlib import Path

import numpy as np 
import pandas as pd
from joblib import dump, load

import torch 
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler

from .utils_surrealgan_prediction import get_surreal_GAN_loader_inference, inference

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=SyntaxWarning)

VERSION = '0.0.1'

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

def main() -> None:
    prog = "PredCRD"
    parser = argparse.ArgumentParser(
        prog=prog,
        description="PredCRD - Predict Continuous Representation of Disease.",
        usage="""
        PredCRD v{VERSION}
        Prediction of continuous representation of disease. Will use CUDA by default if available.

        Currently supporting:
         - SurrealGAN R-indices prediction from DLMUSE output ROIs

        Required arguments:
            [-i, --in_dir]   The filepath of the input DLMUSE ROI (.csv)
            [-o, --out_dir]  The filepath of the output CSV file
        Optional arguments:
            [-h, --help]    Show this help message and exit.
        EXAMPLE USAGE:
            PredCRD  -i           /path/to/input.csv
                     -o           /path/to/output.csv
                     -d           *Optional cuda|cpu
                     -m           *Optional /path/to/model.pth
                     -s           *Optional /path/to/scalar.pkl
                     -mt          *Optional /path/to/icv_mean.npy

        """.format(VERSION=VERSION),
        add_help=False
    )
    # Required Arguments
    parser.add_argument(
        "-i",
        type=str,
        required=True,
        help="[REQUIRED] input DLMUSE ROI file (.csv).",
    )
    parser.add_argument(
        "-o",
        type=str,
        required=True,
        help="[REQUIRED] output CSV file (.csv)",
    )
    parser.add_argument(
        "-d",
        type=str,
        required=False,
        default="cuda",
        help="[Optional] Device type. Choose between cuda and cpu.",
    )
    parser.add_argument(
        "-m",
        type=str,
        required=False,
        default=os.path.join(Path(__file__).parent.parent,"model/ROI_Transformer_best_2.pth"),
        help="[Optional] Model path. model/ROI_Transformer_best_2.pth by default.",
    )
    parser.add_argument(
        "-s",
        type=str,
        required=False,
        default=os.path.join(Path(__file__).parent.parent,"model/scaler_2.pkl"),
        help="[Optional] StandardScalar weight path. model/scaler_2.pkl by default.",
    )
    parser.add_argument(
        "-mt",
        type=str,
        required=False,
        default=os.path.join(Path(__file__).parent.parent,"model/transformer_folder_2_icv_mean_train.npy"),
        help="[Optional] Mean training ICV volume (.npy). model/transformer_folder_2_icv_mean_train.npy by default."
    )

    args = parser.parse_args()

    # Check for required args
    if not args.i or not args.o:
        parser.error("The following arguments are required: -i, -o")

    print("Input: %s" % args.i)
    print("output: %s" % args.o)

    # Set Device
    device = 'cuda' if args.d=="cuda" and torch.cuda.is_available() else 'cpu'
    print(f'Current Device is {device}')

    ## Inference
    print("Initiating the inference.")
    starttime = time.time()

    # Read the input csv into a dataframe
    test_df = pd.read_csv(args.i)
    # Encode sex to 0 and 1
    test_df['Sex'] = test_df['Sex'].apply(lambda x: 0 if x == 'M' else 1)
    # Load the mean ICV volume used for training of the model
    icv_mean_train = np.load(args.mt)[0]
    # Fetch column names containing MUSE_X
    col_name = test_df.filter(regex='^MUSE_(?:20[0-7]|1\d\d|[1-9]\d|[4-9])$').columns
    # ICV correction for column names containing MUSE_X
    test_df[col_name] = test_df.filter(regex='^MUSE_(?:20[0-7]|1\d\d|[1-9]\d|[4-9])$').div(test_df['DLICV'], axis = 0).mul(icv_mean_train, axis = 0)
    # Drop MRID for test_df_X
    test_df_X  = test_df.drop(columns=['MRID']).reset_index(drop = True) # include Age and Sex

    # Use the saved StandardScalar
    ## Load the saved Scaler
    sc=load(args.s) 
    ## Scale using the scaler
    X_inference  = sc.transform(test_df_X) 
    
    ### load model from checkpoint
    model_loaded = TabularTransformer(148, 32, 4, 4, 5).to(device)

    inference_loader  = get_surreal_GAN_loader_inference(X_inference, batch_size = 32)

    inference_result = inference(model = model_loaded,
                                 test_loader = inference_loader,
                                 model_dic_path = args.m,
                                 device = device)
    
    print("Time taken: ", str(time.time() - starttime))
    test_df[['R1','R2','R3','R4','R5']] = inference_result
    # # Save results only
    test_df[['MRID','R1','R2','R3','R4','R5']].to_csv(args.o, index = False)
    
    # Save all 
    # test_df.to_csv(args.o, index = False)

if __name__ == "__main__":
    main()