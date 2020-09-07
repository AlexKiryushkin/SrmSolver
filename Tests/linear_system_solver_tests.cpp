
#include <gtest/gtest.h>

#include <SrmSolver/linear_system_solver.h>
#include <SrmSolver/matrix.h>
#include <SrmSolver/matrix_operations.h>

namespace kae_tests {

template <class T>
class linear_system_solver_tests : public testing::Test
{
public:

  using ElemType = T;

  constexpr static auto threshold{ std::is_same<ElemType, float>::value ? static_cast<ElemType>(1e-5) :
                                                                          static_cast<ElemType>(1e-13) };

  template <unsigned size>
  void testCholeskyDecomposition()
  {
    using MatrixType = kae::Matrix<ElemType, size, size>;

    MatrixType matrix = MatrixType::random();
    matrix = transpose(matrix) * matrix;
    const auto l = kae::detail::choleskyDecompositionL(matrix);
    MatrixType calcMatrix = l * transpose(l);
    const auto thresholdMatrix = matrix - calcMatrix;
    const auto maxDiff = maxCoeff(cwiseAbs(thresholdMatrix));
    EXPECT_LE(maxDiff, threshold);
  }

  template <unsigned rows, unsigned cols>
  void testCholeskySolve()
  {
    using LhsMatrixType = kae::Matrix<ElemType, rows, rows>;
    using RhsMatrixType = kae::Matrix<ElemType, rows, cols>;

    LhsMatrixType lhsMatrix = LhsMatrixType::random();
    lhsMatrix = transpose(lhsMatrix) * lhsMatrix;
    RhsMatrixType rhsMatrix = RhsMatrixType::random();
    const auto solution = kae::detail::choleskySolve(lhsMatrix, rhsMatrix);
    const auto calculatedRhsMatrix = lhsMatrix * solution;
    const auto thresholdMatrix = calculatedRhsMatrix - rhsMatrix;
    const auto maxDiff = maxCoeff(cwiseAbs(thresholdMatrix));
    EXPECT_LE(maxDiff, threshold) << lhsMatrix << "\n\n" << rhsMatrix << "\n\n" << solution << "\n\n";
  }

};

using TypeParams = testing::Types<float, double>;
TYPED_TEST_SUITE(linear_system_solver_tests, TypeParams);

TYPED_TEST(linear_system_solver_tests, cholesky_decomposition_1)
{
  this->template testCholeskyDecomposition<2U>();
  this->template testCholeskyDecomposition<3U>();
  this->template testCholeskyDecomposition<4U>();
  this->template testCholeskyDecomposition<5U>();
  this->template testCholeskyDecomposition<6U>();
}

TYPED_TEST(linear_system_solver_tests, cholesky_solve_1)
{
  this->template testCholeskySolve<2U, 4U>();
  this->template testCholeskySolve<3U, 4U>();
  this->template testCholeskySolve<4U, 4U>();
  this->template testCholeskySolve<5U, 4U>();
  this->template testCholeskySolve<6U, 4U>();
}

} // namespace kae_tests