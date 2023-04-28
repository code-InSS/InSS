#ifndef _LAT_MODEL_H__
#define _LAT_MODEL_H__

#include <string> 
#include <vector>
#include <map>
#include <unordered_map>

typedef struct _Entry
{
  int batch;
  int part;
  float latency;
  float gpu_ratio;
} Entry;

class LatencyModel {
	public:
	void setupTable(std::string TableFile);
        void setupTable_new(std::string TableFile);
	float getLatency(std::string model, int batch, int part);
        float getLatency_new(std::string model, int batch, int part);
        float getActivetime(std::string model, int batch, int part);
        int getkernels(std::string model);
        float getGPURatio(std::string model, int batch, int part);
        float getInterference(std::string my_model,  int my_batch, int my_partition, std::string your_model, int your_partition);
        int makeKey(int batch, int part);
        Entry* parseKey(int key);
	private:
        float getBatchPartInterpolatedLatency(std::string model, int batch, int part);
        float getBatchInterpolatedLatency(std::string model, int batch, int part);
        std::map<std::string,int> _perModelKernels;
        std::map<std::string,float> _perModelBaseidle;
        std::map<std::string,float> _perModelCache;
        std::map<std::string,float> _perModelUtilAlpha;
        std::map<std::string,float> _perModelUtilBeta;
        std::map<std::string,float> _perModelRun1;
        std::map<std::string,float> _perModelRun2;
        std::map<std::string,float> _perModelRun3;
        std::map<std::string,float> _perModelRun4;
        std::map<std::string,float> _perModelRun5;

        std::map<std::string, std::unordered_map<int,float>*> _perModelLatnecyTable;
	std::map<std::string, std::unordered_map<int,float>*> _perModelGPURatioTable;
	std::map<std::string, std::map<int,std::vector<int>>> _perModelBatchVec;
};

#else
#endif
