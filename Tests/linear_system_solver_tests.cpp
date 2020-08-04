
#include <gtest/gtest.h>

#include <SrmSolver/linear_system_solver.h>

namespace tests {

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
    using MatrixType = Eigen::Matrix<ElemType, size, size>;

    MatrixType matrix = MatrixType::Random();
    matrix = matrix.transpose() * matrix;
    const auto l = kae::detail::choleskyDecompositionL(matrix);
    MatrixType calcMatrix = l * l.transpose();
    const auto maxDiff = (matrix - calcMatrix).cwiseAbs().maxCoeff();
    EXPECT_LE(maxDiff, threshold);
  }

  template <unsigned rows, unsigned cols>
  void testCholeskySolve()
  {
    using LhsMatrixType = Eigen::Matrix<ElemType, rows, rows>;
    using RhsMatrixType = Eigen::Matrix<ElemType, rows, cols>;

    LhsMatrixType lhsMatrix = LhsMatrixType::Random();
    lhsMatrix = lhsMatrix.transpose() * lhsMatrix;
    RhsMatrixType rhsMatrix = RhsMatrixType::Random();
    const auto solution = kae::detail::choleskySolve(lhsMatrix, rhsMatrix);
    const auto maxDiff = (lhsMatrix * solution - rhsMatrix).cwiseAbs().maxCoeff();
    EXPECT_LE(maxDiff, threshold) << lhsMatrix << "\n\n" << rhsMatrix << "\n\n" << solution << "\n\n" << lhsMatrix * solution << "\n\n";
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

} // namespace tests