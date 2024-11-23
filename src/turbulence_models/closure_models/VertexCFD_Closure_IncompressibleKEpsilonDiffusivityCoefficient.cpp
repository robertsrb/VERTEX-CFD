#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_Closure_IncompressibleKEpsilonDiffusivityCoefficient.hpp"
#include "VertexCFD_Closure_IncompressibleKEpsilonDiffusivityCoefficient_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS(
    VertexCFD::ClosureModel::IncompressibleKEpsilonDiffusivityCoefficient)
