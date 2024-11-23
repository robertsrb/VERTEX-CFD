#include <VertexCFD_EvaluatorTestHarness.hpp>
#include <closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp>

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleLiftDrag.hpp"
#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
template<class EvalType>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    double _u;
    double _v;
    double _w;
    bool _use_compressible_formula;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_2;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> normals;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> lagrange_pressure;

    Dependencies(const panzer::IntegrationRule& ir,
                 const double u,
                 const double v,
                 const double w,
                 const bool use_compressible_formula)
        : _u(u)
        , _v(v)
        , _w(w)
        , _use_compressible_formula(use_compressible_formula)
        , grad_vel_0("GRAD_velocity_0", ir.dl_vector)
        , grad_vel_1("GRAD_velocity_1", ir.dl_vector)
        , grad_vel_2("GRAD_velocity_2", ir.dl_vector)
        , normals("Side Normal", ir.dl_vector)
        , lagrange_pressure("lagrange_pressure", ir.dl_scalar)
    {
        this->addEvaluatedField(grad_vel_0);
        this->addEvaluatedField(grad_vel_1);
        this->addEvaluatedField(grad_vel_2);

        this->addEvaluatedField(normals);

        this->addEvaluatedField(lagrange_pressure);

        this->setName("Incompressible Lift Drag Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "lift drag test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = grad_vel_0.extent(1);
        const int num_space_dim = grad_vel_0.extent(2);
        using std::pow;
        for (int qp = 0; qp < num_point; ++qp)
        {
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                const int sign = pow(-1, dim + 1);
                const int dimqp = (dim + 1) * sign;
                grad_vel_0(c, qp, dim) = _u * dimqp;
                grad_vel_1(c, qp, dim) = _v * dimqp;
                grad_vel_2(c, qp, dim) = _w * dimqp;

                normals(c, qp, dim) = (_u + _v) * dimqp;
            }

            lagrange_pressure(c, qp) = (_u + _v);
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval(const bool unscaled_density, const bool use_compressible_formula)
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;
    const double nan_val = std::numeric_limits<double>::quiet_NaN();

    // Initialize velocity components and dependents
    const double u = 0.25;
    const double v = 0.5;
    const double w = num_space_dim == 3
                         ? 0.125
                         : std::numeric_limits<double>::quiet_NaN();

    auto deps = Teuchos::rcp(
        new Dependencies<EvalType>(ir, u, v, w, use_compressible_formula));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize class object to test
    double rho = 1.0;
    const double nu = 0.375;

    Teuchos::ParameterList fluid_prop_list;
    fluid_prop_list.set("Kinematic viscosity", nu);
    fluid_prop_list.set("Artificial compressibility", 2.0);
    fluid_prop_list.set("Build Temperature Equation", false);
    if (unscaled_density)
    {
        rho = 3.0;
        fluid_prop_list.set("Density", rho);
    }

    Teuchos::ParameterList user_params;
    user_params.set("Compressible Formula", use_compressible_formula);
    const FluidProperties::ConstantFluidProperties fluid_prop(fluid_prop_list);
    auto eval = Teuchos::rcp(
        new ClosureModel::
            IncompressibleLiftDrag<EvalType, panzer::Traits, num_space_dim>(
                ir, fluid_prop, user_params));
    test_fixture.registerEvaluator<EvalType>(eval);

    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        test_fixture.registerTestField<EvalType>(eval->_total_force[dim]);
        test_fixture.registerTestField<EvalType>(eval->_viscous_force[dim]);
        test_fixture.registerTestField<EvalType>(eval->_pressure_force[dim]);
    }

    test_fixture.evaluate<EvalType>();

    const auto calc_total_force_0
        = test_fixture.getTestFieldData<EvalType>(eval->_total_force[0]);
    const auto calc_viscous_force_0
        = test_fixture.getTestFieldData<EvalType>(eval->_viscous_force[0]);
    const auto calc_pressure_force_0
        = test_fixture.getTestFieldData<EvalType>(eval->_pressure_force[0]);
    const auto calc_total_force_1
        = test_fixture.getTestFieldData<EvalType>(eval->_total_force[1]);
    const auto calc_viscous_force_1
        = test_fixture.getTestFieldData<EvalType>(eval->_viscous_force[1]);
    const auto calc_pressure_force_1
        = test_fixture.getTestFieldData<EvalType>(eval->_pressure_force[1]);

    const int num_point = ir.num_points;

    // Expected values
    const double exp_pressure_force_3d[3] = {-0.5625, 1.125, -1.6875};
    const double exp_pressure_force_2d[3]
        = {exp_pressure_force_3d[0], exp_pressure_force_3d[1], nan_val};
    const double* exp_pressure_force
        = num_space_dim == 3 ? exp_pressure_force_3d : exp_pressure_force_2d;

    const double exp_viscous_force_3d[3]
        = {(unscaled_density ? -2.63671875 : -0.87890625),
           (unscaled_density ? -6.5390625 : -2.1796875),
           (unscaled_density ? -0.52734375 : -0.17578125)};
    const double exp_viscous_force_2d[3]
        = {(unscaled_density ? -0.421875 : -0.140625),
           (unscaled_density ? -3.375 : -1.125),
           nan_val};
    const double exp_compressible_viscous_force_3d[3]
        = {(unscaled_density ? -2.84765625 : -0.94921875),
           (unscaled_density ? -6.1171875 : -2.0390625),
           (unscaled_density ? -1.16015625 : -0.38671875)};
    const double exp_compressible_viscous_force_2d[3]
        = {(unscaled_density ? -1.0546875 : -0.3515625),
           (unscaled_density ? -2.109375 : -0.703125),
           nan_val};
    const double* exp_viscous_force
        = use_compressible_formula
              ? (num_space_dim == 3 ? exp_compressible_viscous_force_3d
                                    : exp_compressible_viscous_force_2d)
              : (num_space_dim == 3 ? exp_viscous_force_3d
                                    : exp_viscous_force_2d);

    const double exp_total_force_3d[3]
        = {(unscaled_density ? -3.19921875 : -1.44140625),
           (unscaled_density ? -5.4140625 : -1.0546875),
           (unscaled_density ? -2.21484375 : -1.86328125)};
    const double exp_total_force_2d[3]
        = {(unscaled_density ? -0.984375 : -0.703125),
           (unscaled_density ? -2.25 : 0.),
           nan_val};
    const double exp_compressible_total_force_3d[3]
        = {(unscaled_density ? -3.41015625 : -1.51171875),
           (unscaled_density ? -4.9921875 : -0.9140625),
           (unscaled_density ? -2.84765625 : -2.07421875)};
    const double exp_compressible_total_force_2d[3]
        = {(unscaled_density ? -1.6171875 : -0.9140625),
           (unscaled_density ? -0.984375 : 0.421875),
           nan_val};
    const double* exp_total_force
        = use_compressible_formula
              ? (num_space_dim == 3 ? exp_compressible_total_force_3d
                                    : exp_compressible_total_force_2d)
              : (num_space_dim == 3 ? exp_total_force_3d : exp_total_force_2d);

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int dim = 0; dim < num_space_dim; dim++)
        {
            const auto calc_total_force
                = test_fixture.getTestFieldData<EvalType>(
                    eval->_total_force[dim]);
            const auto calc_viscous_force
                = test_fixture.getTestFieldData<EvalType>(
                    eval->_viscous_force[dim]);
            const auto calc_pressure_force
                = test_fixture.getTestFieldData<EvalType>(
                    eval->_pressure_force[dim]);

            EXPECT_EQ(exp_total_force[dim],
                      fieldValue(calc_total_force, 0, qp));
            EXPECT_EQ(exp_viscous_force[dim],
                      fieldValue(calc_viscous_force, 0, qp));
            EXPECT_EQ(exp_pressure_force[dim],
                      fieldValue(calc_pressure_force, 0, qp));
        }
    }
}

//-----------------------------------------------------------------//
struct IncompressibleLiftDragTestParams
{
    std::string test_name;
    bool unscaled_density;
    bool use_compressible_formula;
    int num_space_dim;
};

class IncompressibleLiftDragTest
    : public testing::TestWithParam<IncompressibleLiftDragTestParams>
{
  public:
    struct PrintToStringParamName
    {
        template<class T>
        std::string operator()(const testing::TestParamInfo<T>& info) const
        {
            auto testParam
                = static_cast<IncompressibleLiftDragTestParams>(info.param);
            return testParam.test_name;
        }
    };
};

//-----------------------------------------------------------------//
TEST_P(IncompressibleLiftDragTest, cartesian)
{
    const auto params = GetParam();
    if (std::string::npos != params.test_name.find("residual"))
    {
        if (params.num_space_dim == 2)
        {
            testEval<panzer::Traits::Residual, 2>(
                params.unscaled_density, params.use_compressible_formula);
        }
        else
        {
            testEval<panzer::Traits::Residual, 3>(
                params.unscaled_density, params.use_compressible_formula);
        }
    }
    else if (std::string::npos != params.test_name.find("jacobian"))
    {
        if (params.num_space_dim == 2)
        {
            testEval<panzer::Traits::Jacobian, 2>(
                params.unscaled_density, params.use_compressible_formula);
        }
        else
        {
            testEval<panzer::Traits::Jacobian, 3>(
                params.unscaled_density, params.use_compressible_formula);
        }
    }
}

//-----------------------------------------------------------------//
INSTANTIATE_TEST_SUITE_P(
    Test,
    IncompressibleLiftDragTest,
    testing::Values(
        IncompressibleLiftDragTestParams{
            "ScaledDensity2D_residual", false, false, 2},
        IncompressibleLiftDragTestParams{
            "ScaledDensity2D_jacobian", false, false, 2},
        IncompressibleLiftDragTestParams{
            "UnScaledDensity2D_residual", true, false, 2},
        IncompressibleLiftDragTestParams{
            "UnScaledDensity2D_jacobian", true, false, 2},
        IncompressibleLiftDragTestParams{
            "ScaledDensity3D_residual", false, false, 3},
        IncompressibleLiftDragTestParams{
            "ScaledDensity3D_jacobian", false, false, 3},
        IncompressibleLiftDragTestParams{
            "UnScaledDensity3D_residual", true, false, 3},
        IncompressibleLiftDragTestParams{
            "UnScaledDensity3D_jacobian", true, false, 3},
        IncompressibleLiftDragTestParams{
            "ScaledDensityCompressible2D_residual", false, true, 2},
        IncompressibleLiftDragTestParams{
            "ScaledDensityCompressible2D_jacobian", false, true, 2},
        IncompressibleLiftDragTestParams{
            "UnScaledDensityCompressible2D_residual", true, true, 2},
        IncompressibleLiftDragTestParams{
            "UnScaledDensityCompressible2D_jacobian", true, true, 2},
        IncompressibleLiftDragTestParams{
            "ScaledDensityCompressible3D_residual", false, true, 3},
        IncompressibleLiftDragTestParams{
            "ScaledDensityCompressible3D_jacobian", false, true, 3},
        IncompressibleLiftDragTestParams{
            "UnScaledDensityCompressible3D_residual", true, true, 3},
        IncompressibleLiftDragTestParams{
            "UnScaledDensityCompressible3D_jacobian", true, true, 3}),

    IncompressibleLiftDragTest::PrintToStringParamName());

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.user_params.set("Compressible Formula", true);
    test_fixture.user_params.sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.type_name = "IncompressibleLiftDrag";
    test_fixture.eval_name = "Incompressible Lift/Drag "
                             + std::to_string(num_space_dim) + "D";
    test_fixture.template buildAndTest<
        ClosureModel::IncompressibleLiftDrag<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(IncompressibleLiftDrag_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(IncompressibleLiftDrag_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(IncompressibleLiftDrag_Factory3D, residual_test)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(IncompressibleLiftDrag_Factory3D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
