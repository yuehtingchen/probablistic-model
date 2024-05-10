import Model, LogNorm, Normal

def test_Normal():
	pass


def test_default_daytostate(model):
	assert model.dayToState(1, 0) == 0


def test_LogNorm():
	model = LogNorm.LogNorm(num_states=5, num_cell_types=5, num_genes=5, shift=False)
	test_default_daytostate(model=model)

def test_Model():
	pass

def test_main():
	test_Normal()
	test_LogNorm()
	test_Model()


if __name__ == '__main__':
	test_main()
	print("testModels.py passed")