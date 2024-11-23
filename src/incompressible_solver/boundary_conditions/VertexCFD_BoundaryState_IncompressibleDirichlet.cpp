#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleDirichlet.hpp"
#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleDirichlet_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS_NUMSPACEDIM(
    VertexCFD::BoundaryCondition::IncompressibleDirichlet)
