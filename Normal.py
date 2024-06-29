import pyro
import torch
from torch import nn
import pyro.distributions.constraints as constraints

class NormalMM(nn.Module):
	def __init__(self, num_states, num_cell_types, num_genes, shift=False, dayToState=None):
		super().__init__()
		self.num_states = num_states
		self.num_cell_types = num_cell_types
		self.num_genes = num_genes
		self.min_days = 1
		self.shift = shift

		if dayToState is not None:
			self.dayToState = dayToState

	def model(self, data, Z_dist, X=None):
		"""
		:param data: tensor of shape (num_samples)
		:param Z_dist: dictionary of mean and std tensors
		:param X: tensor of shape (num_samples, num_cell_types)
		"""

		# check that Z and X are the right shape
		assert Z_dist['mu'].shape[0] == self.num_cell_types
		assert Z_dist['mu'].shape[1] == self.num_genes
		assert Z_dist['mu'].shape[1] == X.shape[1]

		with pyro.plate("states", self.num_states):
			K = pyro.sample("K", pyro.distributions.Exponential(torch.ones(self.num_states)))
			cell_alpha = pyro.param("cell_alpha", torch.randn((self.num_states, self.num_cell_types), dtype=torch.float32), constraint=constraints.interval(0.0, 1e7))
			scaled_cell_alpha = pyro.deterministic("scaled_cell_alpha", torch.nn.functional.softmax(cell_alpha, dim=1) * K[:, None])
			cell_prob = pyro.sample("cell_prob", pyro.distributions.Dirichlet(scaled_cell_alpha))

		S = pyro.sample("S", pyro.distributions.Exponential(1.0))

		mu = torch.matmul(cell_prob, Z_dist['mu'])
		sigma = torch.matmul(cell_prob ** 2, Z_dist['sigma'])
		cov_terms = torch.stack([
					torch.matmul(cell_prob.T[i].unsqueeze(1), Z_dist['sigma'][i].unsqueeze(1).T) * torch.matmul(cell_prob.T[j].unsqueeze(1), Z_dist['sigma'][j].unsqueeze(1).T)
					for i in range(cell_prob.shape[1])
					for j in range(i + 1, cell_prob.shape[1])
				])
		sigma += 2 * cov_terms.sum(axis=0)

		for s in pyro.plate("samples", X.shape[0]):
			patient_data_day = data['day']
			patient_data_endo = data['endo']

			state = self.dayToState(patient_data_day[s], patient_data_endo[s])

			if self.shift:
				pyro.sample(f"X_{s}", pyro.distributions.Normal(mu[state] + S, sigma[state]).to_event(1), obs=X[s] if X is not None else None)
			else:
				pyro.sample(f"X_{s}", pyro.distributions.Normal(mu[state], sigma[state]).to_event(1), obs=X[s] if X is not None else None)


	def dayToState(self, day, endo=None):
		return int(day - self.min_days)

	def log_probability(self, cell_prob, data, Z_dist, X, S=0.0):
		if not self.shift:
			S = 0

		log_prob = 0
		prob_samples = []

		mu = torch.matmul(cell_prob, Z_dist['mu'])
		sigma = torch.matmul(cell_prob ** 2, Z_dist['sigma'])
		cov_terms = torch.stack([
					torch.matmul(cell_prob.T[i].unsqueeze(1), Z_dist['sigma'][i].unsqueeze(1).T) * torch.matmul(cell_prob.T[j].unsqueeze(1), Z_dist['sigma'][j].unsqueeze(1).T)
					for i in range(cell_prob.shape[1])
					for j in range(i + 1, cell_prob.shape[1])
				])
		sigma += 2 * cov_terms.sum(axis=0)

		for s in range(X.shape[0]):
			patient_data_day = data['day']
			patient_data_endo = data['endo']

			state = self.dayToState(patient_data_day[s], patient_data_endo[s])

			prob = pyro.distributions.Normal(mu[state] + S, sigma[state]).log_prob(X[s])
			log_prob += prob.sum()
			prob_samples.append(prob)

		return log_prob, torch.stack(prob_samples)

	# create to string method
	def __str__(self):
		return "Normal"
