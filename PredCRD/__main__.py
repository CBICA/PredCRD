import warnings
import argparse

import numpy as np 
import pandas as pd
import pickle

import torch 
# import torch.nn as nn
# import torch.optim as optim

# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import StandardScaler

from .utils import get_surreal_GAN_loader, infer_tabular_transformer
from .utils import TabularTransformer

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

VERSION = '0.0.1'

def main():
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
            [-V, --version] Show program's version number and exit.
        EXAMPLE USAGE:
            PredCRD  -i           /path/to/input.csv
                     -o           /path/to/output.csv
                     -d           *Optional cuda/cpu
                     -m           *Optional /path/to/model.pth
                     -s           *Optional /path/to/scalar.pkl

        """.format(
            VERSION=VERSION
        ),
    )
    # Required Arguments
    parser.add_argument(
        "-i",
        "--in_file",
        type=str,
        required=True,
        help="[REQUIRED] input DLMUSE ROI file (.csv).",
    )
    parser.add_argument(
        "-o",
        "--out_file",
        type=str,
        required=True,
        help="[REQUIRED] output CSV file (.csv)",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        required=False,
        default="cuda",
        help="[Optional] Device type. Choose between cuda and cpu."
    )
    parser.add_argument(
        "-m",
        type=str,
        required=False,
        default="../model/ROI_Transformer_best_0.pth",
        help="[Optional] Model path. model/ROI_Transformer_best_0.pth by default.",
    )
    parser.add_argument(
        "-s",
        type=str,
        required=False,
        default="../model/scalar.pkl",
        help="[Optional] StandardScalar weight path. model/scalar.pkl by default.",
    )

    args = parser.parse_args()

    # Check for required args
    if not args.i or not args.o:
        parser.error("The following arguments are required: -i, -o")

    # Set Device
    device = 'cuda' if args.d=="cuda" and torch.cuda.is_available() else 'cpu'
    print(f'Current Device is {device}')

    # load the model architecture
    model = TabularTransformer(148, 32, 4, 4, 5).to(device)
    model_dic_path='../roi_model'

    # Load StandardScalar weights
    with open('model/scaler.pkl','rb') as f:
        scaler = pickle.load(f)

    # Test data
    X_test  = scaler.transform(test_df_X)
    y_test = np.array(test_df[['r1','r2','r3','r4','r5']].reset_index(drop = True))
    
    test_loader  = get_surreal_GAN_loader(X_test , y_test,  batch_size = 32, shuffle = False)

    test_result, output_result = infer_tabular_transformer(model = model,
                                                           model_dic_path = model_dic_path, 
                                                           test_loader = test_loader,
                                                           folder = 0,
                                                           device = device)

if __name__ == "__main__":
    main()