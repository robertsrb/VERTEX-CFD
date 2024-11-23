#ifndef VERTEXCFD_BOUNDARYSTATE_TURBULENCEBOUNDARYEDDYVISCOSITY_HPP
#define VERTEXCFD_BOUNDARYSTATE_TURBULENCEBOUNDARYEDDYVISCOSITY_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <string>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
// Populates boundary eddy viscosity fields according to boundary condition
// type
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class TurbulenceBoundaryEddyViscosity
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    TurbulenceBoundaryEddyViscosity(const panzer::IntegrationRule& ir,
                                    const Teuchos::ParameterList& bc_params,
                                    const std::string& flux_prefix);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_nu_t;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _interior_nu_t;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _wall_func_nu_t;

    bool _wall_func;
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_TURBULENCEBOUNDARYEDDYVISCOSITY_HPP
