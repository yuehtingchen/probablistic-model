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



def proband_data(day, endo, age, birth_control, meta_data, X=None):
	return dict({
		'day': day,
		'endo': endo,
		'age': age,
		'birth_control': birth_control,
		'meta_data': meta_data,
		'X': X
	})


def create_proband_data(df, meta_data_columns):
	# check that the dataframe has patient, cycle, cycle.day, and endo columns
	assert df.index.name == 'Patient'
	assert 'Cycle' in df.columns
	assert 'Cycle.Day' in df.columns
	assert 'Endo.Case.Control' in df.columns
	assert 'Age.Binned' in df.columns
	assert 'Birth.Control' in df.columns

	# check that the meta_data_columns are in the dataframe
	for col in meta_data_columns:
		assert col in df.columns

	# sort the dataframe by patient, cycle, and cycle.day
	df.sort_index(inplace=True)
	df.sort_values(by=['Patient', 'Cycle', 'Cycle.Day'], inplace=True)


	# create patient data as tensors
	day = torch.tensor(np.vstack(df['Cycle.Day'].values).astype(np.float32))
	endo = torch.tensor(np.vstack(df['Endo.Case.Control'].values).astype(np.float32))
	age = torch.tensor(np.vstack(df['Age.Binned'].values).astype(np.float32))
	birth_control = torch.tensor(np.vstack(df['Birth.Control'].values).astype(np.float32))
	meta_data = torch.tensor(np.vstack(df[['Cycle.Day', 'Endo.Case.Control']].values).astype(np.float32))
	X = torch.tensor(np.vstack(df.iloc[:, len(meta_data_columns):].values).astype(np.float32))


	# create ProbandData objects
	# Create Proband data
	data = []
	for i in range(len(day)):
		data.append(proband_data(day=day[i], endo=endo[i], age=age[i], birth_control=birth_control[i], meta_data=meta_data[i], X=X[i]))


	return ProbandDataset(data)





