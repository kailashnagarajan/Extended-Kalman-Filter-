import yaml 
from easydict import EasyDict

def read_params():

	with open('./parameters/parameters.yaml') as file:

		parameters =  yaml.load(file, Loader=yaml.FullLoader)
		
		parameters = EasyDict(parameters)
		
		return parameters
