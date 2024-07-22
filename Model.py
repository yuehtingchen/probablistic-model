from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
import pyro
from pyro import poutine

import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import seaborn as sns
import numpy as np
import pandas as pd

import Dataset, LogNorm, Normal


NUM_STATES = 5
CV_FOLDS = 5

class Model():
	def __init__(self, logger, save_dir, normalization, shift, model_name, cross_validation, proband_data, Z):
		self.logger = logger
		self.save_dir = save_dir
		self.normalization = normalization
		self.shift = shift
		self.model_name = model_name
		self.cross_validation = cross_validation
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
		if self.cross_validation:
			self.train_cv(error, lr, clip_norm, lrd, batch_size)
		else:
			self.train_(error, lr, clip_norm, lrd, batch_size)

	def train_(self, error=1e-4, lr=1e-4, clip_norm=5.0, lrd=0.999, batch_size=1):
		self.init_model()

		# shuffle the data
		data_loader = torch.utils.data.DataLoader(self.proband_data, batch_size=batch_size, shuffle=True)

		param_store = pyro.get_param_store()
		param_store.clear()
		model_path = f'{self.save_dir}{self.normalization}_params'
		print("Training model, best model will be saved in ", model_path)

		optimizer = ClippedAdam({"lr": lr, "clip_norm": clip_norm, "lrd": lrd})
		svi = SVI(self.model.model, self.guide, optimizer, loss=Trace_ELBO())
		losses = []
		running_loss = 1e26
		prev_logprob = 0
		j = 0

		while True:
			for data_batch in data_loader:
				X_batch = data_batch['X']
				loss = svi.step(data_batch, self.Z, X_batch) / (X_batch.shape[0] * X_batch.shape[1])

				# save best model
				if loss < running_loss:
					running_loss = loss
					param_store.save(model_path)
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

	def train_cv(self, error=1e-4, lr=1e-4, clip_norm=5.0, lrd=0.999, batch_size=1):
		kf = KFold(n_splits=CV_FOLDS, shuffle=True)
		train_losses = []

		for i, (train_index, test_index) in enumerate(kf.split(self.proband_data)):
			self.init_model()

			str = f"Training fold {i + 1}/{CV_FOLDS}"
			self.logger.log(str)

			train_loader = torch.utils.data.DataLoader(
				dataset=self.proband_data,
				batch_size=batch_size,
				sampler=torch.utils.data.SubsetRandomSampler(train_index),
			)
			test_loader = torch.utils.data.DataLoader(
				dataset=self.proband_data,
				batch_size=batch_size,
				sampler=torch.utils.data.SubsetRandomSampler(test_index),
			)

			param_store = pyro.get_param_store()
			param_store.clear()
			model_path = f'{self.save_dir}{self.normalization}_params_{i}'

			optimizer = ClippedAdam({"lr": lr, "clip_norm": clip_norm, "lrd": lrd})
			elbo = Trace_ELBO()
			svi = SVI(self.model.model, self.guide, optimizer, loss=elbo)
			train_losses = []
			test_losses = []
			running_loss = 1e26
			prev_logprob = 0
			j = 0

			while True:
				tmp_train_losses = []
				for data_batch in train_loader:
					X_batch = data_batch['X']
					loss = svi.step(data_batch, self.Z, X_batch) / (X_batch.shape[0] * X_batch.shape[1])
					tmp_train_losses.append(loss)
				train_losses.append(np.mean(tmp_train_losses))

				tmp_test_losses = []
				for data_batch in test_loader:
					with torch.no_grad():
						x_batch = data_batch['X']
						tmp_test_losses.append(elbo.loss(self.model.model, self.guide, data_batch, self.Z, x_batch) / (x_batch.shape[0] * x_batch.shape[1]))
				test_loss = np.mean(tmp_test_losses)

				# save best model
				if test_loss < running_loss:
					running_loss = test_loss
					param_store.save(model_path)
				test_losses.append(test_loss)

				if j % 100 == 0:
					print("[iteration %04d] train loss: %.4f, val loss: %.4f" % (j + 1, train_losses[-1], test_loss))

					S = pyro.param("AutoDelta.S").detach()
					log_prob = self.model.log_probability(cell_prob=pyro.param("AutoDelta.cell_prob").detach(), data=data_batch, Z_dist=self.Z, X=x_batch, S=S)[0] / (x_batch.shape[0] * x_batch.shape[1])
					print("log prob:", log_prob)

					if abs(log_prob - prev_logprob) < error:
						break
					prev_logprob = log_prob
				j += 1

			self.losses = train_losses
			self.test_losses = test_losses

	def save_clusters(self, index):
		if self.cross_validation:
			self.save_clusters_cv(index)
		else:
			self.save_clusters_(index)

	def save_clusters_(self, index, cv_fold=None):
		# get the cell probabilities
		cell_prob = pyro.param("AutoDelta.cell_prob").detach().numpy()
		cell_fractions = pd.DataFrame(cell_prob).T
		cell_fractions.index = index
		reorder_list = [f'SEACell-{i}' for i in range(0, len(index))]
		cell_fractions = cell_fractions.reindex(reorder_list)

		# save the cell fractions
		csv_dir = f'{self.save_dir}cell_prob_{self.normalization}{cv_fold}.csv'
		cell_fractions.to_csv(csv_dir)

		# plot the cell fractions
		plt.clf()
		ax = sns.heatmap(cell_fractions, cmap = 'coolwarm', linewidth=0, yticklabels=True, vmin=0, vmax=1)
		ax.set_xlabel('States')
		ax.set_ylabel('Clusters')
		plt.title("Fraction of cell for each state ")

		title = f'{self.save_dir}{self.normalization}_cell_fractions{cv_fold}.png'
		print("Saving plot to ", title)
		plt.savefig(title, dpi=300, bbox_inches="tight")

	def save_clusters_cv(self, index):
		for i in range(CV_FOLDS):
			model_path = f'{self.save_dir}{self.normalization}_params_{i}'
			pyro.get_param_store().load(model_path)

			self.save_clusters_(index, cv_fold=i)

	def log_prob(self, cell_prob, S):
		# transform cell_prob to torch
		if not isinstance(cell_prob, torch.Tensor):
			cell_prob = torch.tensor(cell_prob, dtype=torch.float32)

		data_loader = torch.utils.data.DataLoader(self.proband_data, batch_size=1, shuffle=False)

		log_prob = []
		for data_batch in data_loader:
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
		k += 1 if self.shift else 0

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
		if self.cross_validation:
			plt.plot(self.losses, label='train')
			plt.plot(self.test_losses, label='val')
			plt.legend()
		else:
			plt.plot(self.losses)
		plt.xlabel('Iterations')
		plt.ylabel('Loss')
		plt.title('Loss over iterations')
		plt.savefig(f'{self.save_dir}{self.normalization}_loss.png', dpi=300, bbox_inches="tight")