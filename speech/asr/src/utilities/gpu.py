from tensorflow.python.client import device_lib

# Return the number of GPUs available on this system
def get_variable_gpus():
	local_device_protos = device_lib.list_local_devices()
	
	return [x.name for x in local_device_protos if x.device_type == 'GPU']

# Returns boolean of if a specific gpu_name (string) is available On the system
def check_if_gpu_available(gpu_name):
	list_of_gpus = get_variable_gpus()
	if gpu_name not in list_of_gpus:
		return False
	else:
		reutrn True