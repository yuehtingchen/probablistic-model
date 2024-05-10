# create custom pytorch dataset class
from torch.utils.data import Dataset
import torch
import numpy as np

class ProbandDataset(Dataset):
	def __init__(self, proband_data):
		self.proband_data = proband_data

	def __len__(self):
		return len(self.proband_data)

	def __getitem__(self, idx):
		return self.proband_data[idx]


class ProbandData():
	def __init__(self, day, endo, meta_data, seq_length, X=None):
		self.day = day
		self.endo = endo
		self.meta_data = meta_data
		self.seq_length = seq_length
		self.X = X


def create_proband_data(df, meta_data_columns):
	# check that the dataframe has patient, cycle, cycle.day, and endo columns
	assert df.index.name == 'Patient'
	assert 'Cycle' in df.columns
	assert 'Cycle.Day' in df.columns
	assert 'Endo.Case.Control' in df.columns

	# check that the meta_data_columns are in the dataframe
	for col in meta_data_columns:
		assert col in df.columns

	# sort the dataframe by patient, cycle, and cycle.day
	df.sort_index(inplace=True)
	df.sort_values(by=['Patient', 'Cycle', 'Cycle.Day'], inplace=True)

	# get patient data length per cycle
	seq_length = []

	prev_patient = None
	prev_cycle = None

	for index, row in df.iterrows():
		if index != prev_patient or row['Cycle'] != prev_cycle:
			seq_length.append(1)
		else:
			seq_length[-1] += 1

		prev_patient = index
		prev_cycle = row['Cycle']

	seq_length = torch.tensor(np.array(seq_length))

	# create patient data as tensors
	meta_data = torch.tensor(np.vstack(df[['Cycle.Day', 'Endo.Case.Control']].values).astype(np.float32))
	X = torch.tensor(np.vstack(df.iloc[:, len(meta_data_columns):].values).astype(np.float32))


	# create ProbandData objects
	# Create Proband data
	data = []
	for i in range(len(seq_length)):
		data.append(ProbandData(day=df['Cycle.Day'], endo=df['Endo.Case.Control'], meta_data=meta_data[i: i + seq_length[i]], seq_length=seq_length[i], X=X[i: i + seq_length[i]]))


	return ProbandDataset(data)





