from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
import pyro
from pyro import poutine

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import Dataset, LogNorm, Normal


NUM_STATES = 5

class Model():
	def __init__(self, logger, save_dir, normalization, shift, model_name, proband_data, Z):
		self.logger = logger
		self.save_dir = save_dir
		self.normalization = normalization
		self.shift = shift
		self.model_name = model_name
		self.proband_data = proband_data
		self.Z = Z

		self.num_states = NUM_STATES
		self.num_cell_types = Z['mu'].shape[0]
		self.num_genes =  Z['mu'].shape[1]

	def hmm_expose(self, msg):
		return msg["name"].startswith("cell_prob") or msg["name"].startswith("S") or msg["name"].startswith("K")

	def init_model(self):
		self.model = Normal.NormalMM(num_states=self.num_states, num_cell_types=self.num_cell_types, num_genes=self.num_genes, shift=self.shift)

		self.alt_model = LogNorm.LogNorm(num_states=self.num_states, num_cell_types=self.num_cell_types, num_genes=self.num_genes, shift=self.shift)

		if self.model_name == "LogNorm":
			tmp = self.model
			self.model = self.alt_model
			self.alt_model = tmp

		print("Initialize ", self.model, " model:")
		print("  num_states: ", self.num_states)
		print("  num_cell_types: ", self.num_cell_types)
		print("  num_genes: ", self.num_genes)
		print("  shift: ", self.shift)
		print()

		self.guide = pyro.infer.autoguide.AutoDelta(poutine.block(self.model.model, expose_fn=self.hmm_expose))


	def train(self, error=1e-4, lr=1e-4, clip_norm=5.0, lrd=0.999, batch_size=1):
		pyro.clear_param_store()
		model_path = f'{self.save_dir}{self.normalization}_params'
		print("Training model, best model will be saved in ", model_path)

		optimizer = ClippedAdam({"lr": lr, "clip_norm": clip_norm, "lrd": lrd})
		svi = SVI(self.model.model, self.guide, optimizer, loss=Trace_ELBO())
		losses = []
		running_loss = 1e26
		prev_logprob = 0
		j = 0

		while True:
			for data_batch in self.proband_data:
				X_batch = data_batch['X']
				loss = svi.step(data_batch, self.Z, X_batch) / (X_batch.shape[0] * X_batch.shape[1])

				# save best model
				if loss < running_loss:
					running_loss = loss
					pyro.get_param_store().save(model_path)
				losses.append(loss)

			if j % 100 == 0:
				print("[iteration %04d] loss: %.4f" % (j + 1, loss))

				S = pyro.param("AutoDelta.S").detach()
				log_prob = self.model.log_probability(cell_prob=pyro.param("AutoDelta.cell_prob").detach(), data=data_batch, Z_dist=self.Z, X=X_batch, S=S)[0] / (X_batch.shape[0] * X_batch.shape[1])
				print("log prob:", log_prob)

				if abs(log_prob - prev_logprob) < error:
					break
				prev_logprob = log_prob
			j += 1

		self.losses = losses

		return svi, losses

	def save_clusters(self, index):
		# get the cell probabilities
		cell_prob = pyro.param("AutoDelta.cell_prob").detach().numpy()
		cell_fractions = pd.DataFrame(cell_prob).T
		cell_fractions.index = index
		reorder_list = [f'SEACell-{i}' for i in range(0, len(index))]
		cell_fractions = cell_fractions.reindex(reorder_list)

		# save the cell fractions
		cell_fractions.to_csv(f'{self.save_dir}cell_prob_{self.normalization}.csv')

		# plot the cell fractions
		plt.clf()
		ax = sns.heatmap(cell_fractions, cmap = 'coolwarm', linewidth=0, yticklabels=True, vmin=0, vmax=1)
		ax.set_xlabel('States')
		ax.set_ylabel('Clusters')
		plt.title("Fraction of cell for each state ")

		title = f'{self.save_dir}{self.normalization}_cell_fractions.png'
		print("Saving plot to ", title)
		plt.savefig(title, dpi=300, bbox_inches="tight")

	def log_prob(self, cell_prob, S):
		# transform cell_prob to torch
		if not isinstance(cell_prob, torch.Tensor):
			cell_prob = torch.tensor(cell_prob, dtype=torch.float32)

		log_prob = []
		for data_batch in self.proband_data:
			X_batch = data_batch['X']
			log_prob.append(self.model.log_probability(cell_prob=cell_prob, data=data_batch, Z_dist=self.Z, X=X_batch, S=S)[0] / (X_batch.shape[0] * X_batch.shape[1]))

		return np.array(log_prob)

	def save_log_prob(self):
		S = pyro.param("AutoDelta.S").detach() if self.shift else 0
		cell_prob = pyro.param("AutoDelta.cell_prob").detach()
		log_prob = self.log_prob(cell_prob, S)

		self.logger.log(f"Log prob: {log_prob.mean()}")

	# approximate Akaike Information Criterion
	def save_AIC(self):
		S = pyro.param("AutoDelta.S").detach() if self.shift else 0
		cell_prob = pyro.param("AutoDelta.cell_prob").detach()
		log_prob = self.log_prob(cell_prob, S)

		# get number of parameters
		k = cell_prob.shape[0] * cell_prob.shape[1] + 1

		# calculate AIC
		AIC = -2 * log_prob.mean() + 2 * k

		self.logger.log(f'Number of parameters: {k}')
		self.logger.log(f'AIC: {AIC}')

	# approximate Deviance Information Criterion
	def save_DIC(self):
		predict = pyro.infer.Predictive(self.model.model, guide=self.guide, num_samples=100, return_sites=("cell_prob", "S"))
		for data_batch in self.proband_data:
			samples = {k: v.detach().cpu().numpy() for k, v in predict.forward(data_batch, self.Z, data_batch['X']).items()}

		pi_theta = np.array([self.log_prob(cell_prob, S) for cell_prob, S in zip(samples['cell_prob'], samples['S'])])
		D_bar = -2 * pi_theta.mean(axis=0)

		pi_theta_star = self.log_prob(samples['cell_prob'].mean(axis=0), samples['S'].mean(axis=0))
		D_theta_star = -2 * pi_theta_star
		self.logger.log(f'expected deviance: {D_bar}')
		self.logger.log(f'effective number of parameters: {D_bar - D_theta_star}')
		self.logger.log(f'DIC: {D_bar - D_theta_star + D_bar}')


	def plot_losses(self):
		plt.clf()
		plt.plot(self.losses)
		plt.xlabel('Iterations')
		plt.ylabel('Loss')
		plt.title('Loss over iterations')
		plt.savefig(f'{self.save_dir}{self.normalization}_loss.png', dpi=300, bbox_inches="tight")