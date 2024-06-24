import pyro
import torch
from torch import nn
import pyro.distributions.constraints as constraints


class LogNorm(nn.Module):
	def __init__(self, num_states, num_cell_types, num_genes, shift=False, dayToState=None, debug=False):
		super().__init__()
		self.min_days = 1
		self.debug = debug
		self.num_states = num_states
		self.num_cell_types = num_cell_types
		self.num_genes = num_genes

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

		S = pyro.sample("S", pyro.distributions.Exponential(1.0))

		with pyro.plate("states", self.num_states):
			K = pyro.sample("K", pyro.distributions.Exponential(torch.ones(self.num_states, dtype=torch.float64)))
			cell_alpha = pyro.param("cell_alpha", torch.randn((self.num_states, self.num_cell_types), dtype=torch.float64), constraint=constraints.interval(0.0, 1e10))
			scaled_cell_alpha = pyro.deterministic("scaled_cell_alpha", torch.nn.functional.softmax(cell_alpha, dim=1) * K[:, None])
			cell_prob = pyro.sample("cell_prob", pyro.distributions.Dirichlet(scaled_cell_alpha))

		mu, sigma = self.calculate_mu_sigma(Z_dist, cell_prob)

		for s in pyro.plate("samples", X.shape[0]):
			patient_data_day = data['day']
			patient_data_endo = data['endo']

			state = self.dayToState(patient_data_day[s], patient_data_endo[s])

			with pyro.plate("genes_{}".format(s), X.shape[1]):
				if self.shift:
					pyro.sample("X_{}".format(s), pyro.distributions.Normal(mu[state] - S, sigma[state]), obs=(X[s]) if X is not None else None)
				else:
					pyro.sample("X_{}".format(s), pyro.distributions.Normal(mu[state], sigma[state]), obs=(X[s]) if X is not None else None)


	def dayToState(self, day, endo=None):
		return int(day - self.min_days)


	def calculate_mu_sigma(self, Z_dist, cell_prob):
		mu = torch.zeros((self.num_states, self.num_genes), dtype=torch.float64)
		sigma = torch.zeros((self.num_states, self.num_genes), dtype=torch.float64)

		for i in range(self.num_states):
			mus = torch.zeros(self.num_genes, dtype=torch.float64)
			sigmas = torch.zeros(self.num_genes, dtype=torch.float64)
			for j in range(self.num_genes):
				t_mu, t_sigma = self.logNormSum(Z_dist['mu'][:, j], Z_dist['sigma'][:, j], Z_dist['corr'], cell_prob[i, :])
				mus[j] = t_mu
				sigmas[j] = t_sigma

			mu[i] = mus
			sigma[i] = sigmas

		return mu, sigma

	def logNormSum(self, mu, sigma, corr, w):
		"""
		Parameters
		----------
		mu : array_like
			Mean of the log-normal random variables.
		sigma : array_like
			Standard deviation of the log-normal random variables.
		corr : array_like
			Correlation matrix of the log-normal random variables.
		w : array_like
			Weights of the log-normal random variables.
		Return the mean and variance of the weighted sum of log-normal random variables.

		Reference: https://cran.r-project.org/web/packages/lognorm/vignettes/lognormalSum.html
		"""
		mu = mu.view(-1, 1)
		sigma = sigma.view(-1, 1)
		w = w.view(-1, 1)

		# Calculate the expected value of each log-normal random variable
		EV = torch.multiply(torch.exp(mu + sigma**2/2), w)
		S = EV.sum()

		# Calculate the variance of the weighted sum of log-normal random variables
		var = torch.multiply(torch.multiply(EV@EV.T, sigma@sigma.T), corr).sum(axis=0).sum(axis=0)/S**2

		mu_s = torch.log(S) - var/2

		return mu_s, torch.sqrt(var)


	def log_probability(self, cell_prob, data, Z_dist, X, S=0.0):
		if not self.shift:
			S = 0

		log_prob = 0
		prob_samples = []

		mu, sigma = self.calculate_mu_sigma(Z_dist, cell_prob)

		for s in range(X.shape[0]):
			patient_data_day = data['day']
			patient_data_endo = data['endo']

			state = self.dayToState(patient_data_day[s], patient_data_endo[s])
			prob = pyro.distributions.Normal(mu[state] - S, sigma[state]).log_prob(X[s])
			log_prob += prob.sum()
			prob_samples.append(prob)

		return log_prob, torch.stack(prob_samples)


	# create to string method
	def __str__(self):
		return "LogNorm"
