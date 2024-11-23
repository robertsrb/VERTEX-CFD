#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "turbulence_models/boundary_conditions/VertexCFD_BoundaryState_TurbulenceBoundaryEddyViscosity.hpp"

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_config.hpp>

#include <mpi.h>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
// Test data dependencies.
template<class EvalType>
struct Dependencies : public PHX::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _nu_t_int;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _nu_t_wf;

    Dependencies(const panzer::IntegrationRule& ir)
        : _nu_t_int("turbulent_eddy_viscosity", ir.dl_scalar)
        , _nu_t_wf("wall_func_turbulent_eddy_viscosity", ir.dl_scalar)
    {
        this->addEvaluatedField(_nu_t_int);
        this->addEvaluatedField(_nu_t_wf);
        this->setName(
            "Turbulence Model Boundary Eddy Viscosity Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData /**d**/) override
    {
        _nu_t_int.deep_copy(2.5);
        _nu_t_wf.deep_copy(3.5);
    }
};

//---------------------------------------------------------------------------//
// Inlet/outlet cases
enum class EddyViscosity
{
    wall_func,
    wall_modeled
};

//---------------------------------------------------------------------------//
template<class EvalType>
void testEval(const EddyViscosity bc_type)
{
    // Test fixture
    const int num_space_dim = 2;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Set BC type
    std::string type = "";

    switch (bc_type)
    {
        case (EddyViscosity::wall_func):
            type = "Wall Function Condition";
            break;
        case (EddyViscosity::wall_modeled):
            type = "Wall Modeled Condition";
            break;
    }

    // Create dependencies
    const auto dep_eval
        = Teuchos::rcp(new Dependencies<EvalType>(*test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Create boundary eddy viscosity evaluator.
    Teuchos::ParameterList bc_params;
    bc_params.set("Type", type);
    const auto eddy_visc_eval = Teuchos::rcp(
        new BoundaryCondition::TurbulenceBoundaryEddyViscosity<EvalType,
                                                               panzer::Traits>(
            *test_fixture.ir, bc_params, "BOUNDARY_"));
    test_fixture.registerEvaluator<EvalType>(eddy_visc_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(eddy_visc_eval->_boundary_nu_t);

    // Evaluate values
    test_fixture.evaluate<EvalType>();

    // Check values
    const auto boundary_nu_t_result = test_fixture.getTestFieldData<EvalType>(
        eddy_visc_eval->_boundary_nu_t);

    const int num_point = boundary_nu_t_result.extent(1);

    const double exp_value = type == "Wall Function Condition" ? 3.5 : 2.5;

    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_value, fieldValue(boundary_nu_t_result, 0, qp));
    }
}
//---------------------------------------------------------------------------//
// Value parameterized test fixture
struct EvaluationTest : public testing::TestWithParam<EddyViscosity>
{
    // Case generator for parameterized test
    struct ParamNameGenerator
    {
        std::string
        operator()(const testing::TestParamInfo<EddyViscosity>& info) const
        {
            const auto bc_type = info.param;
            switch (bc_type)
            {
                case (EddyViscosity::wall_func):
                    return "wall_func";
                case (EddyViscosity::wall_modeled):
                    return "wall_modeled";
                default:
                    return "INVALID_NAME";
            }
        }
    };
};

//---------------------------------------------------------------------------//
// Residual evaluation
TEST_P(EvaluationTest, residual)
{
    EddyViscosity bc_type;
    bc_type = GetParam();
    testEval<panzer::Traits::Residual>(bc_type);
}

// Jacobian evaluation
TEST_P(EvaluationTest, jacobian)
{
    EddyViscosity bc_type;
    bc_type = GetParam();
    testEval<panzer::Traits::Jacobian>(bc_type);
}

//---------------------------------------------------------------------------//
// Generate test suite with wall function and wall modeled options
INSTANTIATE_TEST_SUITE_P(EddyViscosityBC,
                         EvaluationTest,
                         testing::Values(EddyViscosity::wall_func,
                                         EddyViscosity::wall_modeled),
                         EvaluationTest::ParamNameGenerator{});

//---------------------------------------------------------------------------//
} // end namespace Test
} // namespace VertexCFD
