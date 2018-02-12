#include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor-blas/xlinalg.hpp" 


int main(){
xt::xarray<double> arr1
  {{1.0, 2.0, 3.0},
   {2.0, 5.0, 7.0},
   {-2.0, 5.0, 7.0}};

xt::xarray<double> arr3
  {{1.0, 2.0, 3.0},
   {2.0, 5.0, -7.0},
   {-2.0, 5.0, 7.0}};   

xt::xarray<double> arr2
  {5.0, 6.0, 7.0};


xt::filter(arr1,arr3<=0) = 0;

std::cout<<arr1<<"\n\n"<<arr3<<"\n";

xt::xarray<double> res;
xt::xarray<double> res2;
res = xt::random::randn<double>({2,3});
res2 = xt::random::randn<double>({2,3});

auto c = xt::linalg::dot(res,xt::transpose(res2));

std::cout<<c;
}