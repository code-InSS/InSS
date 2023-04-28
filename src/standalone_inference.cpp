#include <torch/script.h> // One-stop header.
#include <cuda_runtime.h>
#include <numeric>
#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <sys/time.h>
#include <pthread.h>
#include <assert.h>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <queue>
#include <condition_variable>
#include <cuda_profiler_api.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include "json/json.h"
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include "socket.h"
#include "torch_utils.h"
#include "common_utils.h" //printTimeStamp moved to here
#include <torch/csrc/jit/runtime/graph_executor.h>
#define IMAGENET_ROW 224
#define IMAGENET_COL 224
using namespace std; 
namespace po = boost::program_options; 
using namespace cv;

// mean interval between inference (in seconds)
double g_mean;

// total number of interference executions
int g_numReqs;

// batch size
int g_batchSize;

// name of model
std::string g_task;

// directory to model (.pt) file
std::string g_taskFile;

typedef struct _InputInfo {
	std::vector<std::vector<int>*> InputDims;
	std::vector<std::string> InputTypes;
} InputInfo;

std::map<std::string,InputInfo*> g_nameToInputInfo;

po::variables_map parse_opts(int ac, char** av) {
	po::options_description desc("Allowed options");
	desc.add_options()("help,h", "Produce help message")
		("task,t", po::value<std::string>()->default_value("resnet50"), "name of model")
		("taskfile",po::value<std::string>()->default_value("resnet50.pt"), "dir/to/model.pt")
		("batch,b", po::value<int>()->default_value(1),"size of batch to send") 
		("requests,r",po::value<int>()->default_value(1),"how many requests are going to be issued to the server" ) 
		("mean,m,",po::value<double>()->default_value(0.3),"how long is the average time between each request(in seconds)")
		("input,i",po::value<std::string>()->default_value("input.txt"),"txt file that contains list of inputs")
		("input_config_json", po::value<std::string>()->default_value("input_config.json"), "json file for input dimensions");
	po::variables_map vm;
	po::store(po::parse_command_line(ac, av, desc), vm); 
	po::notify(vm); 
	if (vm.count("help")) {
		std::cout << desc << "\n"; exit(1);   
	} 
	return vm;
}

int readInputJSONFile(const char* input_config_file, std::map<std::string, InputInfo*> &mapping){
#ifdef DEBUG
	printf("Reading App JSON File: %s \n", input_config_file);
#endif
	Json::Value root;   //
	std::ifstream ifs;
	ifs.open(input_config_file);

	Json::CharReaderBuilder builder;
	JSONCPP_STRING errs;
	if (!parseFromStream(builder, ifs, &root, &errs)) {
		std::cout << errs << std::endl;
		ifs.close();
		return EXIT_FAILURE;
	}
	for(unsigned int i=0; i < root["ModelInfoSpecs"].size(); i++){
		std::string model_name = root["ModelInfoSpecs"][i]["ModelName"].asString();
		mapping[model_name]=new InputInfo();
		for(unsigned int j=0; j< root["ModelInfoSpecs"][i]["Inputs"].size(); j++){
			mapping[model_name]->InputDims.push_back(new std::vector<int>());
			for(unsigned int k=0; k<root["ModelInfoSpecs"][i]["Inputs"][j]["InputDim"].size(); k++){
				mapping[model_name]->InputDims[j]->push_back(root["ModelInfoSpecs"][i]["Inputs"][j]["InputDim"][k].asInt());
			}
			mapping[model_name]->InputTypes.push_back(root["ModelInfoSpecs"][i]["Inputs"][j]["InputType"].asString());
		}
	}
	ifs.close();
	return EXIT_SUCCESS;
}

void setupGlobalVars(po::variables_map &vm){
	g_task = vm["task"].as<std::string>();
	g_mean = vm["mean"].as<double>();
	g_numReqs=vm["requests"].as<int>();
	g_batchSize= vm["batch"].as<int>();
	assert(g_batchSize!=0);
	g_taskFile = vm["taskfile"].as<std::string>();
	if(readInputJSONFile(vm["input_config_json"].as<std::string>().c_str(), g_nameToInputInfo))
	{
		printf("Failed reading json file: %s \n", vm["input_config_json"].as<std::string>().c_str());
		exit(1);
	}
	return;
}

void PyTorchInit(){
	uint64_t total_end, total_start;
	std::vector<torch::jit::IValue> inputs;
	std::vector<int64_t> sizes={1};
	torch::TensorOptions options;
	options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA,0).requires_grad(false);
	total_start = getCurNs();
	torch::Tensor dummy1 = at::empty(sizes,options);
	torch::Tensor dummy2 = at::empty(sizes,options);
	torch::Tensor dummy3 = dummy1 + dummy2;
	cudaDeviceSynchronize();
	total_end = getCurNs();
	std::cout << double(total_end - total_start)/1000000 << " PyTorchInit total ms "<<std::endl;
	return;
}

torch::Tensor getRandomNLPInput(std::vector<int> &input_dims, std::string &input_type){
	std::vector<int64_t> dims;
	// read input dimensions
	dims.push_back(g_batchSize);
	int cnt=1;  // skip the first element
	for(auto dim : input_dims){
		if(cnt  !=0){
			cnt = cnt -1;
			continue;
		}
		dims.push_back(dim);
	}
	torch::TensorOptions options;
	if (input_type == "INT64")
		options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA,0).requires_grad(false);
	else if(input_type == "FP32")
		options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA,0).requires_grad(false);
	else{
		printf("unsupported type: %s \n", input_type.c_str());
		exit(1);   
	}
	torch::Tensor input=torch::ones(dims,options);
	return input;
}


torch::Tensor getRandomImgInput(std::vector<int> &input_dims, std::string &input_type, int batch_size){
	std::vector<int64_t> dims;
	// read input dimensions
	dims.push_back(batch_size);
	for(auto dim : input_dims)
		dims.push_back(dim);
	torch::TensorOptions options;
	if (input_type == "INT64")
		options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA,0).requires_grad(false);
	else if(input_type == "FP32")
		options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA,0).requires_grad(false);
	else{
		printf("unsupported type: %s \n", input_type.c_str());
		exit(1);   
	}
	torch::Tensor input=torch::ones(dims,options);
	return input;
}

void getInputs(const char* netname, std::vector<torch::jit::IValue> &inputs, int batch_size){
	torch::Tensor input;
	torch::Device gpu_dev(torch::kCUDA,0);
#ifdef DEBUG
	//printf("get input for %s \n", netname);
#endif 
	std::string c_str_bert = "bert";
	std::string str_name = std::string(netname); 
	std::string c_str_fp32 = "FP32";
	std::vector<int> DIMS = {3,224,224};
	input = getRandomImgInput(DIMS,c_str_fp32,batch_size);       
	// assume this model is for profiling random model
/*
	if(g_nameToInputInfo.find(str_name) == g_nameToInputInfo.end()){
		std::string c_str_fp32 = "FP32";
		std::vector<int> DIMS = {3,224,224};
		input = getRandomImgInput(DIMS,c_str_fp32,batch_size);
	}
	else{

		for(unsigned int i=0; i < g_nameToInputInfo[str_name]->InputDims.size(); i++){
			if(str_name.find(c_str_bert) != std::string::npos)
				input = getRandomNLPInput(*(g_nameToInputInfo[str_name]->InputDims[i]), g_nameToInputInfo[str_name]->InputTypes[i]);
			else
				input = getRandomImgInput(*(g_nameToInputInfo[str_name]->InputDims[i]), g_nameToInputInfo[str_name]->InputTypes[i], batch_size);
		}
	}
*/
	input = input.to(gpu_dev);
	//std::cout << input.sizes() << std::endl;
	inputs.push_back(input);
	return;
}

void computeRequest(){
	#ifdef DEBUG
	//std::cout<<"started copmuting thread"<<std::endl;
#endif
	torch::Tensor input;
	std::vector<torch::jit::IValue> inputs;
	torch::Device gpu_dev(torch::kCUDA,0);
	uint64_t total_end, total_start;
	const char *netname = g_task.c_str();
	int i;
	
	//PyTorchInit();
	//std::cout<< "waiting for 3 seconds after PyTorchInit" << std::endl;
	//usleep(3*1000*1000);
	uint64_t t1,t2,t3,t4;
	t1 = getCurNs();
	std::shared_ptr<torch::jit::script::Module> module = std::make_shared<torch::jit::script::Module>(torch::jit::load(g_taskFile.c_str(),gpu_dev));
	t2 = getCurNs();
	module->to(gpu_dev);
	module->eval();
	cudaDeviceSynchronize();

	
	t3= getCurNs();  
    
	// warmup
	/*
	for(int batch_size = 32; batch_size >=1; batch_size--){
		getInputs(netname, inputs,batch_size);
		//std::cout<< "end input" << std::endl;
		module->forward(inputs);
		//std::cout<< "end forward" << std::endl;
		cudaDeviceSynchronize();
		inputs.clear();
	}*/	
	for (int i=1;i<=1;i++){
		getInputs(netname, inputs,g_batchSize);
		//inputs.to(gpu_dev)
		//std::cout<< "end input" << std::endl;
		torch::IValue output = module->forward(inputs);
		//std::cout<< "end forward" << std::endl;
		cudaDeviceSynchronize();
		inputs.clear();
	}

	t4 = getCurNs();
	//std::cout<< "waiting for 3 seconds after warmup" << std::endl;
	
	//usleep(3*1000*1000);



	uint64_t start,end;
	uint64_t start_t,end_t;
		
	/*
	double *trans_time;
	trans_time = (double*)malloc(g_numReqs*sizeof(double));
	double *gpu_time;
	gpu_time = (double*)malloc(g_numReqs*sizeof(double));
	double *cpu_time;
	cpu_time = (double*)malloc(g_numReqs*sizeof(double));
	double trans_time_sum;
	double gpu_time_sum;
	double cpu_time_sum;

	ofstream trans_file;
	trans_file.open("./trans_data.txt", ios::app);
	ofstream gpu_file;
	gpu_file.open("./gpu_data.txt", ios::app);
	ofstream feed_file;
	feed_file.open("./feed_data.txt", ios::app);
	ofstream thp_file;
	thp_file.open("./thp_data.txt", ios::app);
*/
	getInputs(netname,inputs,g_batchSize);
	//start_t = getCurNs();
	total_start=getCurNs();
	int num=0;
	for (int i =0; i < g_numReqs; i++){
		//uint64_t start_trans = getCurNs();
		//printf("start_trans%llu\n",start_trans);
		
		//inputs.to(gpu_dev);
		//uint64_t start_GPU = getCurNs();
		
		
		//uint64_t start_check= getCurNs();
		cudaProfilerStart();
		//std::cout << inputs.size()<< std::endl;
		torch::IValue output = module->forward(inputs);

		torch::Tensor t;
		if(output.isTuple()){
			t = output.toTuple()->elements()[0].toTensor(); // 1st output;
			//t = t.to(torch::Device(torch::kCPU));
			//std::cout << t.sizes()<< std::endl;
		}
		else{t = output.toTensor();
		//std::cout << t.sizes()<< std::endl;
		}
		//uint64_t end_GPU = getCurNs();

		t=t.to(torch::Device(torch::kCPU));
		//uint64_t end_CPU = getCurNs();

		

		cudaDeviceSynchronize();
		//end=getCurNs(); 
		cudaProfilerStop();
		//printTimeStampWithName(netname, "END EXEC");
		//printf("latency: %lf\n", double(end-start)/1000000);
		//printf("throughput: %lf\n", g_batchSize/(double(end-start)/1000000000) );
		//inputs.clear();  
		/*
		trans_time[i] = double(start_GPU - start_trans)/1000000;
		trans_time_sum += trans_time[i];
		gpu_time[i] = double(end_GPU - start_GPU)/1000000;
		gpu_time_sum += gpu_time[i];
		cpu_time[i] = double(end_CPU - end_GPU)/1000000;
		cpu_time_sum += cpu_time[i];
		*/
	/*
		end_t= getCurNs();
		num++;
		if (double(end_t-start_t)/1000000>500){
			printTimeStamp(" ");
			printf("%lf\n", double(end_t-start_t)/num/1000000);
			printf("%lf\n\n", num * g_batchSize/(double(end_t-start_t)/1000000000) );
			start_t = getCurNs();
			num=0;
		}
		
		if(i%10==9){
			end_t= getCurNs();
			printTimeStamp("data");
			printf("%lf\n", double(end_t-start_t)/10000000);
			printf(" %lf\n\n", g_batchSize/(double(end_t-start_t)/10000000000) );
			start_t = getCurNs();
		}
		*/
	}
//#ifdef DEBUG
	total_end=getCurNs();

	printf("model:%s\n",netname);
	
	//printf("trans latency: %lf \n", trans_time_sum / g_numReqs);
	//printf("gpu latency: %lf \n", gpu_time_sum / g_numReqs);
	//printf("cpu latency: %lf \n", cpu_time_sum / g_numReqs);
	printf("total latency: %lf \n", double(total_end-total_start)/1000000/g_numReqs);
	printf("total throughput: %lf\n", g_numReqs*(g_batchSize)/(double(total_end-total_start)/1000000000) );
//#endif
/*
	trans_file << trans_time_sum / g_numReqs <<",";

	gpu_file << gpu_time_sum / g_numReqs <<"," ;
	feed_file << cpu_time_sum / g_numReqs <<"," ;
	thp_file << g_numReqs*(g_batchSize)/(double(total_end-total_start)/1000000000) <<",";

	trans_file.close();
	gpu_file.close();

	//free(trans_time);
	//free(gpu_time);
	//free(cpu_time);*/
}

int main(int argc, char** argv) {
	//torch::jit::getBailoutDepth() = 0;
	torch::jit::getProfilingMode() = false;
	/*get parameters for this program*/
	po::variables_map vm = parse_opts(argc, argv);
	setupGlobalVars(vm);
	//printTimeStamp("START PROGRAM");
	computeRequest();   
	//printTimeStamp("END PROGRAM");
	return 0;
}
