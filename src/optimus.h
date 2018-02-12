#include "layer.h"
#include "xtensor/xarray.hpp"
#include <iostream>
#include <vector>

namespace optimus{

class Model{

private:
	std::vector<Layer> layers;
	double learning_rate;
	xt::xarray<double> input_data,output_data,in_gradient,out_gradient;

	
public:
	Model() = default;

	void add(Layer layer){
		layers.push_back(layer);
	}
// TO-DO: Create placeholder for batch dispatch
	void forward(){

		std::vector<Layer>::iterator it;

		for(it=layers.begin();it!=layers.end();it++){
			
		}
	}


};

}
