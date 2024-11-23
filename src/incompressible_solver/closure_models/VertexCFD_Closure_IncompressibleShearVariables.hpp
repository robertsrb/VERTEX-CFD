#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLESHEARVARIABLES_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLESHEARVARIABLES_HPP

#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Kokkos_Core.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
// Multi-dimension shear variables evaluation.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleShearVariables
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleShearVariables(
        const panzer::IntegrationRule& ir,
        const FluidProperties::ConstantFluidProperties& fluid_prop);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _tau_w;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _u_tau;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _normals;
    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_velocity;

    double _nu;
    double _rho;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLESHEARVARIABLES_HPP
