#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFConvectiveFlux.hpp"

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> pressure;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> vel_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> vel_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> vel_2;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> rho;

    Dependencies(const panzer::IntegrationRule& ir,
                 const double u,
                 const double v,
                 const double w)
        : _u(u)
        , _v(v)
        , _w(w)
        , pressure("lagrange_pressure", ir.dl_scalar)
        , vel_0("velocity_0", ir.dl_scalar)
        , vel_1("velocity_1", ir.dl_scalar)
        , vel_2("velocity_2", ir.dl_scalar)
        , rho("density", ir.dl_scalar)

    {
        this->addEvaluatedField(pressure);
        this->addEvaluatedField(vel_0);
        this->addEvaluatedField(vel_1);
        this->addEvaluatedField(vel_2);
        this->addEvaluatedField(rho);

        this->setName(
            "Incompressible LSVOF Convective Flux Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData) override
    {
        pressure.deep_copy(0.75);
        vel_0.deep_copy(_u);
        vel_1.deep_copy(_v);
        vel_2.deep_copy(_w);
        rho.deep_copy(1.375);
    }
};

template<class EvalType, int NumSpaceDim>
void testEval()
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Initialize velocity components and dependents
    const double _nanval = std::numeric_limits<double>::quiet_NaN();
    const double u = 0.25;
    const double v = 0.5;
    const double w = num_space_dim > 2 ? 0.125 : _nanval;
    const double p = 0.75;
    const double rho = 1.375;

    auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir, u, v, w));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize class object to test
    const auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleLSVOFConvectiveFlux<EvalType,
                                                            panzer::Traits,
                                                            num_space_dim>(ir));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_continuity_flux);
    for (int dim = 0; dim < num_space_dim; ++dim)
        test_fixture.registerTestField<EvalType>(eval->_momentum_flux[dim]);

    test_fixture.evaluate<EvalType>();

    const auto fc_cont
        = test_fixture.getTestFieldData<EvalType>(eval->_continuity_flux);
    const auto fc_mom_0
        = test_fixture.getTestFieldData<EvalType>(eval->_momentum_flux[0]);
    const auto fc_mom_1
        = test_fixture.getTestFieldData<EvalType>(eval->_momentum_flux[1]);

    const int num_point = ir.num_points;

    // Expected values
    const double exp_cont_flux[3] = {rho * u, rho * v, rho * w};
    const double exp_mom_0_flux[3]
        = {rho * u * u + p,
           rho * u * v,
           num_space_dim == 3 ? rho * u * w : _nanval};
    const double exp_mom_1_flux[3]
        = {rho * v * u,
           rho * v * v + p,
           num_space_dim == 3 ? rho * v * w : _nanval};
    const double exp_mom_2_flux[3]
        = {rho * w * u,
           rho * w * v,
           num_space_dim == 3 ? rho * w * w + p : _nanval};

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int dim = 0; dim < num_space_dim; dim++)
        {
            EXPECT_EQ(exp_cont_flux[dim], fieldValue(fc_cont, 0, qp, dim));
            EXPECT_EQ(exp_mom_0_flux[dim], fieldValue(fc_mom_0, 0, qp, dim));
            EXPECT_EQ(exp_mom_1_flux[dim], fieldValue(fc_mom_1, 0, qp, dim));

            if (num_space_dim > 2) // 3D mesh
            {
                const auto fc_mom_2 = test_fixture.getTestFieldData<EvalType>(
                    eval->_momentum_flux[2]);
                EXPECT_EQ(exp_mom_2_flux[dim],
                          fieldValue(fc_mom_2, 0, qp, dim));
            }
        }
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleLSVOFConvectiveFlux2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleLSVOFConvectiveFlux2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleLSVOFConvectiveFlux3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleLSVOFConvectiveFlux3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
