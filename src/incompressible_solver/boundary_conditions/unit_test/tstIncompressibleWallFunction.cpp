#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleWallFunction.hpp"
#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
// Continuity Equation cases
enum class ContinuityModel
{
    AC,
    EDAC
};
//---------------------------------------------------------------------------//
// Test data dependencies.
template<class EvalType>
struct Dependencies : public PHX::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    double _phi, _u_0, _u_1, _u_2;
    bool _build_temp_equ;
    ContinuityModel _continuity_model;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _lagrange_pressure;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _temperature;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _velocity_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _velocity_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _velocity_2;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_u_tau;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_y_plus;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_nu_t;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> _normals;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_velocity_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_velocity_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_velocity_2;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_temperature;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_lagrange_pressure;

    Dependencies(const panzer::IntegrationRule& ir,
                 const double phi,
                 const double u_0,
                 const double u_1,
                 const double u_2,
                 const bool build_temp_equ,
                 const ContinuityModel continuity_model)
        : _phi(phi)
        , _u_0(u_0)
        , _u_1(u_1)
        , _u_2(u_2)
        , _build_temp_equ(build_temp_equ)
        , _continuity_model(continuity_model)
        , _lagrange_pressure("lagrange_pressure", ir.dl_scalar)
        , _temperature("temperature", ir.dl_scalar)
        , _velocity_0("velocity_0", ir.dl_scalar)
        , _velocity_1("velocity_1", ir.dl_scalar)
        , _velocity_2("velocity_2", ir.dl_scalar)
        , _boundary_u_tau("BOUNDARY_friction_velocity", ir.dl_scalar)
        , _boundary_y_plus("BOUNDARY_y_plus", ir.dl_scalar)
        , _boundary_nu_t("BOUNDARY_turbulent_eddy_viscosity", ir.dl_scalar)
        , _normals("Side Normal", ir.dl_vector)
        , _grad_velocity_0("GRAD_velocity_0", ir.dl_vector)
        , _grad_velocity_1("GRAD_velocity_1", ir.dl_vector)
        , _grad_velocity_2("GRAD_velocity_2", ir.dl_vector)
        , _grad_temperature("GRAD_temperature", ir.dl_vector)
        , _grad_lagrange_pressure("GRAD_lagrange_pressure", ir.dl_vector)
    {
        this->addEvaluatedField(_lagrange_pressure);
        if (_build_temp_equ)
            this->addEvaluatedField(_temperature);
        this->addEvaluatedField(_velocity_0);
        this->addEvaluatedField(_velocity_1);
        this->addEvaluatedField(_velocity_2);
        this->addEvaluatedField(_boundary_u_tau);
        this->addEvaluatedField(_boundary_y_plus);
        this->addEvaluatedField(_boundary_nu_t);

        this->addEvaluatedField(_normals);

        this->addEvaluatedField(_grad_velocity_0);
        this->addEvaluatedField(_grad_velocity_1);
        this->addEvaluatedField(_grad_velocity_2);
        if (_build_temp_equ)
            this->addEvaluatedField(_grad_temperature);
        if (_continuity_model == ContinuityModel::EDAC)
            this->addEvaluatedField(_grad_lagrange_pressure);

        this->setName("Incompressible Wall Function Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        // Set scalar variables
        _lagrange_pressure.deep_copy(_phi);
        _velocity_0.deep_copy(_u_0);
        _velocity_1.deep_copy(_u_1);
        _velocity_2.deep_copy(_u_2);
        _boundary_u_tau.deep_copy(2.0);
        _boundary_y_plus.deep_copy(13.0);
        _boundary_nu_t.deep_copy(5.0);
        if (_build_temp_equ)
            _temperature.deep_copy(_u_0 + _u_1);

        Kokkos::parallel_for(
            "incompressible wall function test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int c) const
    {
        const int num_point = _lagrange_pressure.extent(1);
        const int num_space_dim = _grad_velocity_0.extent(2);

        using std::pow;

        for (int qp = 0; qp < num_point; ++qp)
        {
            // Set gradient and normal vectors
            for (int d = 0; d < num_space_dim; ++d)
            {
                const int dimqp = (d + 1) * pow(-1, d + 1);

                _normals(c, qp, d) = 0.02 * dimqp;

                _grad_velocity_0(c, qp, d) = 0.250 * dimqp;
                _grad_velocity_1(c, qp, d) = 0.500 * dimqp;
                _grad_velocity_2(c, qp, d) = 0.125 * dimqp;

                if (_build_temp_equ)
                    _grad_temperature(c, qp, d) = (_u_0 + _u_1) * dimqp;
                if (_continuity_model == ContinuityModel::EDAC)
                    _grad_lagrange_pressure(c, qp, d) = (_u_0 - _u_1);
            }
        }
    }
};

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval(const bool build_temp_equ, const ContinuityModel continuity_model)
{
    // Test fixture
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    std::string continuity_model_name = "";
    switch (continuity_model)
    {
        case (ContinuityModel::AC):
            continuity_model_name = "AC";
            break;
        case (ContinuityModel::EDAC):
            continuity_model_name = "EDAC";
            break;
    }

    // Initialize values and create dependencies
    const double _nanval = std::numeric_limits<double>::quiet_NaN();
    const double phi = 1.5;
    const double u_0 = 1.0;
    const double u_1 = -2.0;
    const double u_2 = num_space_dim == 3 ? 3.0 : _nanval;
    const auto dep_eval = Teuchos::rcp(new Dependencies<EvalType>(
        *test_fixture.ir, phi, u_0, u_1, u_2, build_temp_equ, continuity_model));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Equation of state
    Teuchos::ParameterList fluid_prop_list;
    fluid_prop_list.set("Kinematic viscosity", 2.5);
    fluid_prop_list.set("Artificial compressibility", 2.0);
    fluid_prop_list.set("Build Temperature Equation", build_temp_equ);
    if (build_temp_equ)
    {
        fluid_prop_list.set("Thermal conductivity", 0.5);
        fluid_prop_list.set("Specific heat capacity", 0.6);
    }
    const FluidProperties::ConstantFluidProperties fluid_prop(fluid_prop_list);

    // Create wall function evaluator.
    const auto wf_eval = Teuchos::rcp(
        new BoundaryCondition::
            IncompressibleWallFunction<EvalType, panzer::Traits, num_space_dim>(
                *test_fixture.ir, fluid_prop, continuity_model_name));
    test_fixture.registerEvaluator<EvalType>(wf_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(
        wf_eval->_boundary_lagrange_pressure);
    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        test_fixture.registerTestField<EvalType>(
            wf_eval->_boundary_velocity[dim]);
        test_fixture.registerTestField<EvalType>(
            wf_eval->_boundary_grad_velocity[dim]);
    }

    if (build_temp_equ)
    {
        test_fixture.registerTestField<EvalType>(
            wf_eval->_boundary_temperature);
        test_fixture.registerTestField<EvalType>(
            wf_eval->_boundary_grad_temperature);
    }

    if (continuity_model == ContinuityModel::EDAC)
    {
        test_fixture.registerTestField<EvalType>(
            wf_eval->_boundary_grad_lagrange_pressure);
    }

    // Evaluate incompressible wall function
    test_fixture.evaluate<EvalType>();

    // Get wall function field
    auto boundary_lagrange_pressure_result
        = test_fixture.getTestFieldData<EvalType>(
            wf_eval->_boundary_lagrange_pressure);

    // Create expected velocity gradient
    const double grad_u_2D[3] = {-0.25, 0.020512820512820513, _nanval};
    const double grad_u_3D[3] = {-0.25, 0.020512820512820513, -0.75};
    const double* grad_u = (num_space_dim == 3) ? grad_u_3D : grad_u_2D;

    const double grad_v_2D[3] = {-0.5, 1.0, _nanval};
    const double grad_v_3D[3] = {-0.5, 1.0, -1.5};
    const double* grad_v = (num_space_dim == 3) ? grad_v_3D : grad_v_2D;

    const double grad_w_2D[3] = {_nanval, _nanval, _nanval};
    const double grad_w_3D[3] = {-0.125, 0.25, -0.375};
    const double* grad_w = (num_space_dim == 3) ? grad_w_3D : grad_w_2D;

    const double* grad_vel[3] = {grad_u, grad_v, grad_w};

    const double grad_temp_2D[3] = {0.998, -1.996, _nanval};
    const double grad_temp_3D[3] = {0.9944, -1.9888, 2.9832};
    const double* grad_temp = (num_space_dim == 3) ? grad_temp_3D
                                                   : grad_temp_2D;

    const double* vel = grad_temp;

    // Loop over quadrature points and mesh dimension
    const int num_point = boundary_lagrange_pressure_result.extent(1);
    for (int qp = 0; qp < num_point; ++qp)
    {
        // Lagrange pressure
        EXPECT_DOUBLE_EQ(phi,
                         fieldValue(boundary_lagrange_pressure_result, 0, qp));

        // Temperature
        if (build_temp_equ)
        {
            const auto boundary_temperature_result
                = test_fixture.getTestFieldData<EvalType>(
                    wf_eval->_boundary_temperature);
            EXPECT_DOUBLE_EQ(u_0 + u_1,
                             fieldValue(boundary_temperature_result, 0, qp));
        }

        // Loop over mesh dimension to assert gradient vectors
        for (int d = 0; d < num_space_dim; ++d)
        {
            const auto boundary_velocity_d_result
                = test_fixture.getTestFieldData<EvalType>(
                    wf_eval->_boundary_velocity[d]);
            EXPECT_DOUBLE_EQ(vel[d],
                             fieldValue(boundary_velocity_d_result, 0, qp));

            for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
            {
                const auto boundary_grad_velocity_d_result
                    = test_fixture.getTestFieldData<EvalType>(
                        wf_eval->_boundary_grad_velocity[vel_dim]);

                EXPECT_DOUBLE_EQ(
                    grad_vel[vel_dim][d],
                    fieldValue(boundary_grad_velocity_d_result, 0, qp, d));
            }

            if (build_temp_equ)
            {
                const auto boundary_grad_temperature_result
                    = test_fixture.getTestFieldData<EvalType>(
                        wf_eval->_boundary_grad_temperature);
                EXPECT_DOUBLE_EQ(
                    grad_temp[d],
                    fieldValue(boundary_grad_temperature_result, 0, qp, d));
            }

            if (continuity_model == ContinuityModel::EDAC)
            {
                const double exp_val = u_0 - u_1;
                const auto boundary_grad_lagrange_pressure_result
                    = test_fixture.getTestFieldData<EvalType>(
                        wf_eval->_boundary_grad_lagrange_pressure);
                EXPECT_DOUBLE_EQ(
                    exp_val,
                    fieldValue(
                        boundary_grad_lagrange_pressure_result, 0, qp, d));
            }
        }
    }
}

//---------------------------------------------------------------------------//
// Value parameterized test fixture
struct EvaluationTest : public testing::TestWithParam<ContinuityModel>
{
    // Case generator for parameterized test
    struct ParamNameGenerator
    {
        std::string
        operator()(const testing::TestParamInfo<ContinuityModel>& info) const
        {
            const auto continuity_model = info.param;
            switch (continuity_model)
            {
                case (ContinuityModel::AC):
                    return "AC";
                case (ContinuityModel::EDAC):
                    return "EDAC";
                default:
                    return "INVALID_NAME";
            }
        }
    };
};

//---------------------------------------------------------------------------//
// 2-D incompressible isothermal wall function
TEST_P(EvaluationTest, isothermalresidual2D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 2>(false, continuity_model);
}

TEST_P(EvaluationTest, isothermaljacobian2D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 2>(false, continuity_model);
}

//---------------------------------------------------------------------------//
// 3-D incompressible isothermal wall function
TEST_P(EvaluationTest, isothermalresidual3D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 3>(false, continuity_model);
}

TEST_P(EvaluationTest, isothermaljacobian3D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 3>(false, continuity_model);
}

//---------------------------------------------------------------------------//
// 2-D incompressible wall function
TEST_P(EvaluationTest, residual2D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 2>(true, continuity_model);
}

TEST_P(EvaluationTest, jacobian2D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 2>(true, continuity_model);
}

//---------------------------------------------------------------------------//
// 3-D incompressible wall function
TEST_P(EvaluationTest, residual3D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 3>(true, continuity_model);
}

TEST_P(EvaluationTest, jacobian3D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 3>(true, continuity_model);
}

//---------------------------------------------------------------------------//
// Generate test suite with continuity models
INSTANTIATE_TEST_SUITE_P(ContinuityModelType,
                         EvaluationTest,
                         testing::Values(ContinuityModel::AC,
                                         ContinuityModel::EDAC),
                         EvaluationTest::ParamNameGenerator{});

} // end namespace Test
} // end namespace VertexCFD
