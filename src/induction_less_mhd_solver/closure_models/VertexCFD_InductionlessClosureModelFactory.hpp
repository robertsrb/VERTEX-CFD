#ifndef VERTEXCFD_INDUCTIONLESSCLOSUREMODELFACTORY_HPP
#define VERTEXCFD_INDUCTIONLESSCLOSUREMODELFACTORY_HPP

#include <Panzer_Traits.hpp>

#include <Phalanx_Evaluator.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
class InductionlessFactory
{
  public:
    void buildClosureModel(
        const std::string& closure_type,
        const Teuchos::RCP<panzer::IntegrationRule>& ir,
        const Teuchos::ParameterList& user_params,
        const Teuchos::ParameterList& closure_params,
        bool& found_model,
        std::string& error_msg,
        Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>>
            evaluators);
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_INDUCTIONLESSCLOSUREMODELFACTORY_HPP
