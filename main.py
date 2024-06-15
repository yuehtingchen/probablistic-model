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
    parser.add_argument("--data_path", type=str, help="Path to proband data file, will be concatenated with _gene.csv or _switch.csv if param is gene or switch respectively", default="data/exp_metadata_cleaned_js")
    parser.add_argument("--meta_data_columns", type=str, nargs='+', help="Columns to include in the meta data tensor", required=True)
    parser.add_argument("--output_path", type=str, help="Path to save the processed data", default="results/")
    parser.add_argument("--shift", help="Whether to shift the data", action="store_true")
    parser.add_argument("--reference_data_path", type=str, help="Path to single cell reference data file, will be concatenated with _gene.csv or _switch.csv if param is gene or switch respectively", required=True)
    parser.add_argument("--reference_cluster_assignment_path", type=str, help="Path to single cell reference data cluster assignments, first column cell identifier, second column cluster name. Will be concatenated with _gene.csv or _switch.csv if param is gene or switch respectively", required=True)
    parser.add_argument("--switch", help="Whether to use data represented in switches", action="store_true")
    parser.add_argument("--model", type=str, help="Model to use", default="Normal", required=True)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=829)
    parser.add_argument("--error", type=float, help="Error threshold", default=1e-4, required=False)

    args = parser.parse_args()

    print(f"Running model: {args.model}, switch: {args.switch}, shift: {args.shift}")

    # set random seed
    np.random.seed(43)

    # read in the data
    filename = args.data_path + '_switch.csv' if args.switch else args.data_path  + '_gene.csv'
    proband_data = pd.read_csv(filename, delimiter="\t", index_col=0)
    meta_data_columns = args.meta_data_columns
    print("Loaded proband data from file: ", filename)

    # ensure metadata columns match the data
    assert all([c in proband_data.columns for c in meta_data_columns])

    # change js to be in natural log if running lognorm model
    if args.model == "LogNorm":
        proband_data.iloc[:, len(meta_data_columns):] = np.log(2 ** (proband_data.iloc[:, len(meta_data_columns):].values - 10)) + 10
        print("Transformed data to natural log")

	# create file names
    filename_suff = args.data_path.split("/")[-1]
    today = date.today()
    save_dir = args.output_path + filename_suff + "/" + args.model + "/"
    os.makedirs(save_dir, exist_ok=True)
    normalization = today.strftime("%Y-%m-%d") + "_js"
    if args.switch:
        normalization += "_switch"
    if args.shift:
        normalization += "_shift"

	# create logger
    logger = Output.Logger(save_dir, normalization)
    # log model params
    logger.log(f"Model: {args.model}\n Switch: {args.switch}\n Shift: {args.shift}\n")
    logger.log(f"Data path: {args.data_path}\n Reference data path: {args.reference_data_path}\n")
    logger.log(f"Meta data columns: {meta_data_columns}\n")

    # load reference data
    if args.switch:
        reference_path = args.reference_data_path + "_switch.csv"
        reference_assignment_path = args.reference_cluster_assignment_path + "_switch.csv"
    else:
        reference_path = args.reference_data_path + "_gene.csv"
        reference_assignment_path = args.reference_cluster_assignment_path + "_switch.csv"

    ref_df = pd.read_csv(reference_path, index_col=0, delimiter="\t").T
    ref_assign_df = pd.read_csv(reference_assignment_path, delimiter="\t", index_col=0)

    if args.model == "LogNorm":
        # convert to natural log
        ref_df = np.log(2 ** (ref_df - 10)) + 10

    clusters_mean = ref_df.groupby(ref_assign_df['SEACell']).mean()
    clusters_std = ref_df.groupby(ref_assign_df['SEACell']).std()

	# ensure cluster and proband data have the same genes
    if clusters_mean.columns.all() != proband_data.columns.all():
        # extract metadata columns
        meta_data = proband_data[meta_data_columns]

        overlap = clusters_mean.columns.intersection(proband_data.columns)
        clusters_mean = clusters_mean[overlap]
        clusters_std = clusters_std[overlap]
        proband_data = proband_data[overlap]

        # add metadata columns back
        proband_data = pd.concat([meta_data, proband_data], axis=1)
    assert clusters_mean.columns.all() == proband_data.columns.all()

	# create the proband data
    dataset = Dataset.create_proband_data(proband_data, meta_data_columns)
    # shuffle the data
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    X = torch.cat([d['X'] for d in data_loader])


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

	# create the model
    model = Model.Model(logger, save_dir, normalization, args.shift, args.model, data_loader, Z)
    model.init_model()
    model.train(error=args.error, lr=0.01, clip_norm=20.0, lrd=0.999, batch_size=args.batch_size)

	# save model results
    model.plot_losses()
    model.save_clusters(index=clusters_mean.index)
    model.save_log_prob()
    model.save_AIC()
    # model.save_DIC()

    logger.close()




if __name__ == "__main__":
    np.random.seed(43)
    main()



