#include <Panzer_Traits.hpp>

#include "VertexCFD_Closure_MethodManufacturedSolutionSource.hpp"
#include "VertexCFD_Closure_MethodManufacturedSolutionSource_impl.hpp"

template class VertexCFD::ClosureModel::
    MethodManufacturedSolutionSource<panzer::Traits::Tangent, panzer::Traits, 2>;
template class VertexCFD::ClosureModel::
    MethodManufacturedSolutionSource<panzer::Traits::Tangent, panzer::Traits, 3>;
