#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor-blas/xlinalg.hpp" 
#include "../layer.h"

namespace optimus{

class FullyConnectedLayer:public Layer{

private:
	int incoming_dim;
	int output_dim;
	xt::xarray<double> weights;
	xt::xarray<double> d_weights;

public:
	FullyConnectedLayer(int incoming_dim,int output_dim){
		this.incoming_dim = incoming_dim;
		this.output_dim = output_dim;
		weights = xt::random::randn<double>({incoming_dim, output_dim});
		d_weights = xt::zeros<double>(weights.shape());
	}

	void forward(xt::xarray<double> &input_data,xt::xarray<double> &output_data){
		output_data = xt::linalg::dot(input_data,weights);
	}

	void backward(xt::xarray<double> &in_gradient,xt::xarray<double> &out_gradient,xt::xarray<double> &output_data,xt::xarray<double> &input_data){
		out_gradient = xt::linalg::dot(in_gradient,xt::transpose(weights));
		d_weights = xt::linalg::dot(xt::transpose(input_data),in_gradient);
	}


}

}