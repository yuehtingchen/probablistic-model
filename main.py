import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from argparse import ArgumentParser
import os

import Dataset, Output, Model



def main():
	# add argument parser
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to proband data file", required=True)
    parser.add_argument("--meta_data_columns", type=str, nargs='+', help="Columns to include in the meta data tensor", required=True)
    parser.add_argument("--output_path", type=str, help="Path to save the processed data", default="data/")
    parser.add_argument("--shift", type=bool, help="Whether to shift the data", default=True)
    parser.add_argument("--reference_data_path", type=str, help="Path to reference data file, without the _mean.csv or _std.csv suffix, and with _ln suffix if running with lognorm model", required=True)
    parser.add_argument("--switch", type=bool, help="Whether to switch the data", default=True)
    parser.add_argument("--model", type=str, help="Model to use", default="Normal", required=True)

    args = parser.parse_args()

    print("Loading data from: ", args.data_path)
    print("Saving data to: ", args.output_path)
    print("Meta data columns: ", args.meta_data_columns)

	# create file names
    today = date.today()
    save_dir = args.output_path + args.model + "/"
    os.makedirs(save_dir, exist_ok=True)
    normalization = today.strftime("%Y-%m-%d") + "_js"
    if args.switch:
        normalization += "_switch"
    if args.shift:
        normalization += "_shift"

	# create logger
    logger = Output.Logger(save_dir, normalization)

    # read in the data
    proband_data = pd.read_csv(args.data_path, delimiter="\t", index_col=0)
    meta_data_columns = args.meta_data_columns
    print("Loaded proband data with first 10 columns: ", proband_data.columns[:10])

    reference_path = args.reference_data_path
    if args.switch:
        reference_path += "_switch"
    if args.model == "LogNorm":
        reference_path += "_ln"

    clusters_mean = pd.read_csv(reference_path + "_mean.csv", index_col=0)
    clusters_std = pd.read_csv(reference_path + "_std.csv", index_col=0)
    print("Loaded reference data from: ", reference_path + "_mean.csv", reference_path + "_std.csv")


	# ensure cluster and proband data have the same genes
    assert clusters_mean.columns.all() == proband_data.columns.all()

	# create the proband data
    proband_data = Dataset.create_proband_data(proband_data, meta_data_columns)
    X = torch.cat([d.X for d in proband_data])
    print("Created proband data with shape: ", X.shape)


	# create the reference data
    # clusters with low std
    clusters_low_std = clusters_mean.index[clusters_mean.std(axis=1) < 1e-14]
    for c in clusters_low_std:
        clusters_mean.loc[c] += np.random.normal(0, 1e-13, clusters_mean.shape[1])

	# check that clusters with low std have been perturbed
    assert (clusters_mean.loc[clusters_low_std].std(axis=1) < 1e-14).sum() == 0

    clusters_corr = clusters_mean.T.corr()
    Z = {
	'mu': torch.tensor(np.vstack(clusters_mean.values).astype(np.float32)),
	'sigma': torch.tensor(np.vstack(clusters_std.values).astype(np.float32)),
	'corr': torch.tensor(clusters_corr.values.astype(np.float32))}
    print("Loaded reference data with shape: ", Z['mu'].shape, Z['sigma'].shape, Z['corr'].shape)

	# plot cluster and X
    Output.plot_exp_cluster(X, Z['mu'], f"{save_dir}{normalization}_mean_exp.png")

	# shift data by 10 if not shifting in model
    if not args.shift:
        X = X + 10
        Z['mu'] = Z['mu'] + 10


	# create the model
    model = Model.Model(logger, save_dir, normalization, args.shift, args.model, proband_data, X, Z)
    model.init_model()
    model.train(num_iterations=500, lr=0.05, clip_norm=20.0, lrd=0.999)

	# save model results
    model.save_clusters()
    model.save_log_prob()

    logger.close()




if __name__ == "__main__":
    np.random.seed(43)
    main()



