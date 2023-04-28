#include "device_perf_model.h"
#include <iostream>
#include "json/json.h"
#include "json/json-forwards.h"

DevPerfModel::DevPerfModel(){
}

DevPerfModel::~DevPerfModel(){
}

int DevPerfModel::setup(std::string latency_info_file, std::string model_const_file,std::string util_file, int dev_mem, int bandwidth, float alpha, float beta){
	_latModel.setupTable(latency_info_file);
	//_intModel.setup(model_const_file,util_file);    
	_devMem=dev_mem;
	//_bandwidth=bandwidth;
	//_SchAlpha=alpha;
	//_SchBeta=beta;

	return EXIT_SUCCESS;

}

int DevPerfModel::setup_new(std::string latency_info_file, std::string latency_info_file_1, int dev_mem, int bandwidth, float alpha, float beta){
	_latModel.setupTable_new(latency_info_file);
	_latModel.setupTable(latency_info_file_1);
	_devMem=dev_mem;
	_bandwidth=bandwidth;
	_SchAlpha=alpha;
	_SchBeta=beta;

	return EXIT_SUCCESS;

}

float DevPerfModel::getLatency(std::string model, int batch, int part){
	return _latModel.getLatency(model, batch, part);
}

float DevPerfModel::getLatency_new(std::string model, int batch, int part){
	float data_trans=(3*224*224+1000)/_bandwidth; //*batch
	//return data_trans+_latModel.getLatency_new(model, batch, part);
	return data_trans+_latModel.getLatency(model, batch, part);
}

float DevPerfModel::getGPURatio(std::string model, int batch, int part){
	return _latModel.getGPURatio(model, batch, part);
}

float DevPerfModel::getInterference(std::string my_model, int my_batch, int my_partition, \
		std::string your_model, int your_batch, int your_partition){
	return _intModel.getInterference(my_model,my_batch,my_partition,your_model,your_batch,your_partition);
}

float DevPerfModel::getInterference_new(std::string my_model, int my_batch, int my_partition, \
		std::string your_model, int your_partition){
	return _latModel.getInterference(my_model,my_batch,my_partition,your_model,your_partition);
}

int DevPerfModel::getDevMem(){
	return  _devMem;
}

float DevPerfModel::getInterferenceIdle(std::string model, int count){
	int kernel=_latModel.getkernels(model);
	return _SchAlpha*count + _SchBeta;
}


