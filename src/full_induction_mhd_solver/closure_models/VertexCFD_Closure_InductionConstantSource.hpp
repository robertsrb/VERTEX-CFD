#ifndef VERTEXCFD_CLOSURE_INDUCTIONCONSTANTSOURCE_HPP
#define VERTEXCFD_CLOSURE_INDUCTIONCONSTANTSOURCE_HPP

#include "full_induction_mhd_solver/mhd_properties/VertexCFD_FullInductionMHDProperties.hpp"

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
// Linear-in-time external magnetic field source for induction equations
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class InductionConstantSource : public panzer::EvaluatorWithBaseImpl<Traits>,
                                public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    InductionConstantSource(const panzer::IntegrationRule& ir,
                            const Teuchos::ParameterList& model_params);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _induction_source;

  private:
    Kokkos::Array<double, num_space_dim> _ind_input_source;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INDUCTIONCONSTANTSOURCE_HPP
