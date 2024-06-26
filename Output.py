import matplotlib.pyplot as plt
import seaborn as sns


def plot_exp_cluster(exp, cluster, title):
	all_clusters = cluster.mean(axis=0)
	all_exp = exp.mean(axis=0)

	plt.scatter(all_clusters, all_exp, s=10)
	plt.xlabel('Gene expression in clusters')
	plt.ylabel('Gene expression in samples')
	plt.title('Difference between gene expression in clusters and samples')
	plt.savefig(title, dpi=300, bbox_inches="tight")


def plot_cluster_gene_distribution(cluster, title):
	plt.clf()
	sns.heatmap(cluster, cmap='viridis')
	plt.title('Cluster gene distribution')
	plt.savefig(title, dpi=300, bbox_inches="tight")


class Logger():
	def __init__(self, output_path, normalization):
		self.output_path = output_path
		self.log_file = open(output_path + f"{normalization}_log.txt", "w")

	def log(self, message):
		print(message)
		self.log_file.write(message + "\n")

	def close(self):
		self.log_file.close()