from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
import pyro
from pyro import poutine

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import Dataset, LogNorm, Normal


NUM_STATES = 5

class Model():
	def __init__(self, logger, save_dir, normalization, shift, model_name, proband_data, X, Z):
		self.logger = logger
		self.save_dir = save_dir
		self.normalization = normalization
		self.shift = shift
		self.model_name = model_name
		self.proband_data = proband_data
		self.X = X
		self.Z = Z

		self.num_states = NUM_STATES
		self.num_cell_types = Z['mu'].shape[0]
		self.num_genes =  Z['mu'].shape[1]

	def hmm_expose(self, msg):
		return msg["name"].startswith("cell_prob") or msg["name"].startswith("S") or msg["name"].startswith("K")

	def init_model(self):
		if self.model_name == "Normal":
			self.model = Normal.NormalMM(num_states=self.num_states, num_cell_types=self.num_cell_types, num_genes=self.num_genes, shift=self.shift)
		elif self.model_name == "LogNorm":
			self.model = LogNorm.LogNorm(num_states=self.num_states, num_cell_types=self.num_cell_types, num_genes=self.num_genes, shift=self.shift)
		else:
			raise ValueError("Invalid model name")

		print("Initialize ", self.model, " model:")
		print("  num_states: ", self.num_states)
		print("  num_cell_types: ", self.num_cell_types)
		print("  num_genes: ", self.num_genes)
		print("  shift: ", self.shift)
		print()

		self.guide = pyro.infer.autoguide.AutoDelta(poutine.block(self.model.model, expose_fn=self.hmm_expose))


	def train(self, num_iterations=1000, lr=1e-4, clip_norm=5.0, lrd=0.999):
		pyro.clear_param_store()
		model_path = f'{self.save_dir}{self.normalization}_params'
		print("Training model, best model will be saved in ", model_path)

		optimizer = ClippedAdam({"lr": lr, "clip_norm": clip_norm, "lrd": lrd})
		svi = SVI(self.model.model, self.guide, optimizer, loss=Trace_ELBO())
		losses = []
		running_loss = 1e26

		for j in range(num_iterations):
			loss = svi.step(self.proband_data, self.Z, self.X) / (len(self.X)*self.X.shape[1])
			if j % 100 == 0:
				print("[iteration %04d] loss: %.4f" % (j + 1, loss))

				S = pyro.param("AutoDelta.S").detach()
				print("log prob:", self.model.log_probability(pyro.param("AutoDelta.cell_prob").detach(), self.proband_data, self.Z, self.X, S=S)[0] \
		  				/ (self.X.shape[0] * self.X.shape[1]))

			# save best model
			if loss < running_loss:
				running_loss = loss
				pyro.get_param_store().save(model_path)
			losses.append(loss)

		return svi, losses

	def save_clusters(self):
		# get the cell probabilities
		cell_prob = pyro.param("AutoDelta.cell_prob").detach().numpy()
		cell_fractions = pd.DataFrame(cell_prob).T

		# save the cell fractions
		cell_fractions.to_csv(f'{self.save_dir}cell_prob_{self.normalization}.csv')

		ax = sns.heatmap(cell_fractions, cmap = 'coolwarm', linewidth=0, yticklabels=True)
		ax.set_xlabel('States')
		ax.set_ylabel('Clusters')
		plt.title("Fraction of cell for each state ")

		title = f'{self.save_dir}{self.normalization}_cell_fractions.png'
		print("Saving plot to ", title)
		plt.savefig(title, dpi=300, bbox_inches="tight")

	def save_log_prob(self):
		S = pyro.param("AutoDelta.S").detach()
		cell_prob = pyro.param("AutoDelta.cell_prob").detach()
		log_prob = self.model.log_probability(cell_prob, self.proband_data, self.Z, self.X, S=S)[0] \
		  				/ (self.X.shape[0] * self.X.shape[1])

		self.logger.log(f"Log prob: {log_prob}")