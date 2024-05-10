import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from argparse import ArgumentParser


def plot_clusters(cell_fractions, title):
	ax = sns.heatmap(cell_fractions, cmap = 'coolwarm', linewidth=0, yticklabels=True)
	ax.set_xlabel('States')
	ax.set_ylabel('Clusters')
	plt.title("Fraction of cell for each state ")
	plt.savefig(title, dpi=300, bbox_inches="tight")


def main():
	parser = ArgumentParser()
	parser.add_argument("--model_path", type=str, help="Path to model file", required=True)
	parser.add_argument("--output_path", type=str, help="Path to save the processed data", default=parser.model_path)
	parser.add_argument("--model_name", type=str, help="Model to use: Normal or LogNorm", required=True)
	parser.add_argument("--shift", type=bool, help="Whether to shift the data", default=False)





if __name__ == "__main__":
    main()