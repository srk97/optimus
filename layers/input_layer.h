#include <iostream>
#include <vector>
#include "xtensor/xarray.hpp"

namespace optimus{

class InputLayer: public Layer{

private:
	xt::xarray<double> input_data; 
	int batch_size;

public:
	InputLayer(xt::xarray<double> input_data){
		this.input_data = input_data;
		this.batch_size = batch_size;
	}

	void forward(xt::xarray<double> &input_data,xt::xarray<double> &output_data){
		output_data = input_data;
	}

};


}