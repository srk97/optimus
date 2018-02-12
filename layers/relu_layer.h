#include "xtensor/xarray.hpp"
#include "../layer.h"
#include "xtensor/xindex_view.hpp"

namespace optimus{

class Relu:public Layer{

public:

	Relu() = default;

	void forward(xt::xarray<double> &input_data,xt::xarray<double> &output_data){
		output_data = input_data;
		xt::filter(output_data,output_data<=0) = 0;
	}

	void backward(xt::xarray<double> &in_gradient,xt::xarray<double> &out_gradient,xt::xarray<double> &output_data,xt::xarray<double> &input_data){
		out_gradient = in_gradient;
		xt::filter(out_gradient,output_data<=0) = 0;
	}

};

}