
#include <gtest/gtest.h>

#include <SrmSolver/cuda_float_types.h>
#include <SrmSolver/get_stencil_indices.h>
#include <SrmSolver/gpu_grid.h>

namespace tests {

template <class T>
class get_stencil_indices_tests : public ::testing::Test
{
public:

  constexpr static unsigned nPoints{ 3U };
  using ElemType        = T;
  using Real2Type       = kae::CudaFloatT<2U, T>;
  using IndexMatrixType = Eigen::Matrix<unsigned, nPoints, nPoints>;

  constexpr static unsigned nx{ 201U };
  constexpr static unsigned ny{ 101U };
  constexpr static unsigned smExtension{ 3U };
  using LxToType    = std::ratio<2, 1>;
  using LyToType    = std::ratio<1, 1>;
  using GpuGridType = kae::GpuGrid<nx, ny, LxToType, LyToType, smExtension, ElemType>;

};

using TypeParams = ::testing::Types<float, double>;
TYPED_TEST_SUITE(get_stencil_indices_tests, TypeParams);

TYPED_TEST(get_stencil_indices_tests, get_stencil_indices_tests_1)
{
  using kae::detail::getStencilIndices;

  using tf              = TestFixture;
  using ElemType        = typename tf::ElemType;
  using Real2Type       = typename tf::Real2Type;
  using IndexMatrixType = typename tf::IndexMatrixType;
  using GpuGridType     = typename tf::GpuGridType;
  using InitList        = std::initializer_list<unsigned>;

  const auto negativeValue = static_cast<ElemType>(-0.1);
  const auto positiveValue = static_cast<ElemType>(1.0);
  std::vector<ElemType> phiValues(GpuGridType::nx * GpuGridType::ny, negativeValue);

  const auto test = [&](unsigned index = 0U, unsigned nList = 3U, InitList list = {})
  {
    const Real2Type surfacePoint{ static_cast<ElemType>(0.45642), static_cast<ElemType>(0.33522) };
    const Real2Type normal{ static_cast<ElemType>(0.8), static_cast<ElemType>(0.6) };

    phiValues.at(index) = positiveValue;

    const auto indexMatrix = getStencilIndices<GpuGridType, tf::nPoints>(phiValues.data(), surfacePoint, normal);
    const auto list0       = (nList == 0) ? list : InitList{ 6678U, 6879U, 6477U };
    const auto list1       = (nList == 1) ? list : InitList{ 6476U, 6677U, 6275U };
    const auto list2       = (nList == 2) ? list : InitList{ 6475U, 6274U, 6676U };
    const IndexMatrixType goldMatrix{ list0, list1, list2 };
    const IndexMatrixType diffMatrix = indexMatrix - goldMatrix;
    if (diffMatrix.maxCoeff() != 0U)
    {
      std::cout << "\nCalculated matrix:\n" << indexMatrix << "\nGold matrix:\n" << goldMatrix << "\nInitializer lists:\n";
      std::copy(std::begin(list0), std::end(list0), std::ostream_iterator<unsigned>{ std::cout, " " });
      std::cout << '\n';
      std::copy(std::begin(list1), std::end(list1), std::ostream_iterator<unsigned>{ std::cout, " " });
      std::cout << '\n';
      std::copy(std::begin(list2), std::end(list2), std::ostream_iterator<unsigned>{ std::cout, " " });
      std::cout << '\n';
    }
    EXPECT_EQ(diffMatrix.maxCoeff(), 0U);

    phiValues.at(index) = negativeValue;
  };

  test();

  test(6879U, 0U, { 6678U, 6477U, 6276U });
  test(6477U, 0U, { 6678U, 6879U, 7080U });
  test(6678U, 0U, { 6879U, 7080U, 7281U });

  test(6677U, 1U, { 6476U, 6275U, 6074U });
  test(6275U, 1U, { 6476U, 6677U, 6878U });
  test(6476U, 1U, { 6677U, 6878U, 7079U });

  test(6274U, 2U, { 6475U, 6676U, 6877U });
  test(6676U, 2U, { 6475U, 6274U, 6073U });
  test(6475U, 2U, { 6274U, 6073U, 5872U });
}

TYPED_TEST(get_stencil_indices_tests, get_stencil_indices_tests_2)
{
  using kae::detail::getStencilIndices;

  using tf              = TestFixture;
  using ElemType        = typename tf::ElemType;
  using Real2Type       = typename tf::Real2Type;
  using IndexMatrixType = typename tf::IndexMatrixType;
  using GpuGridType     = typename tf::GpuGridType;
  using InitList        = std::initializer_list<unsigned>;

  const auto negativeValue = static_cast<ElemType>(-0.1);
  const auto positiveValue = static_cast<ElemType>(1.0);
  std::vector<ElemType> phiValues(GpuGridType::nx * GpuGridType::ny, negativeValue);

  const auto test = [&](unsigned index = 0U, unsigned nList = 3U, InitList list = {})
  {
    const Real2Type surfacePoint{ static_cast<ElemType>(0.45642), static_cast<ElemType>(0.33522) };
    const Real2Type normal{ static_cast<ElemType>(0.8), static_cast<ElemType>(-0.6) };

    phiValues.at(index) = positiveValue;

    const auto indexMatrix = getStencilIndices<GpuGridType, tf::nPoints>(phiValues.data(), surfacePoint, normal);
    const auto list0       = (nList == 0) ? list : InitList{ 6879U, 7080U, 6678U };
    const auto list1       = (nList == 1) ? list : InitList{ 7079U, 6878U, 7280U };
    const auto list2       = (nList == 2) ? list : InitList{ 7279U, 7078U, 7480U };
    const IndexMatrixType goldMatrix{ list0, list1, list2 };
    const IndexMatrixType diffMatrix = indexMatrix - goldMatrix;
    if (diffMatrix.maxCoeff() != 0U)
    {
      std::cout << "\nCalculated matrix:\n" << indexMatrix << "\nGold matrix:\n" << goldMatrix << "\nInitializer lists:\n";
      std::copy(std::begin(list0), std::end(list0), std::ostream_iterator<unsigned>{ std::cout, " " });
      std::cout << '\n';
      std::copy(std::begin(list1), std::end(list1), std::ostream_iterator<unsigned>{ std::cout, " " });
      std::cout << '\n';
      std::copy(std::begin(list2), std::end(list2), std::ostream_iterator<unsigned>{ std::cout, " " });
      std::cout << '\n';
    }
    EXPECT_EQ(diffMatrix.maxCoeff(), 0U);

    phiValues.at(index) = negativeValue;
  };

  test();

  test(7080U, 0U, { 6879U, 6678U, 6477U });
  test(6678U, 0U, { 6879U, 7080U, 7281U });
  test(6879U, 0U, { 7080U, 7281U, 7482U });

  test(6878U, 1U, { 7079U, 7280U, 7481U });
  test(7280U, 1U, { 7079U, 6878U, 6677U });
  test(7079U, 1U, { 6878U, 6677U, 6476U });

  test(7078U, 2U, { 7279U, 7480U, 7681U });
  test(7480U, 2U, { 7279U, 7078U, 6877U });
  test(7279U, 2U, { 7078U, 6877U, 6676U });
}

TYPED_TEST(get_stencil_indices_tests, get_stencil_indices_tests_3)
{
  using kae::detail::getStencilIndices;

  using tf              = TestFixture;
  using ElemType        = typename tf::ElemType;
  using Real2Type       = typename tf::Real2Type;
  using IndexMatrixType = typename tf::IndexMatrixType;
  using GpuGridType     = typename tf::GpuGridType;
  using InitList        = std::initializer_list<unsigned>;

  const auto negativeValue = static_cast<ElemType>(-0.1);
  const auto positiveValue = static_cast<ElemType>(1.0);
  std::vector<ElemType> phiValues(GpuGridType::nx * GpuGridType::ny, negativeValue);

  const auto test = [&](unsigned index = 0U, unsigned nList = 3U, InitList list = {})
  {
    const Real2Type surfacePoint{ static_cast<ElemType>(0.45642), static_cast<ElemType>(0.33522) };
    const Real2Type normal{ static_cast<ElemType>(-0.8), static_cast<ElemType>(0.6) };

    phiValues.at(index) = positiveValue;

    const auto indexMatrix = getStencilIndices<GpuGridType, tf::nPoints>(phiValues.data(), surfacePoint, normal);
    const auto list0       = (nList == 0) ? list : InitList{ 6679U, 6880U, 6478U };
    const auto list1       = (nList == 1) ? list : InitList{ 6680U, 6479U, 6881U };
    const auto list2       = (nList == 2) ? list : InitList{ 6480U, 6279U, 6681U };
    const IndexMatrixType goldMatrix{ list0, list1, list2 };
    const IndexMatrixType diffMatrix = indexMatrix - goldMatrix;
    if (diffMatrix.maxCoeff() != 0U)
    {
      std::cout << "\nCalculated matrix:\n" << indexMatrix << "\nGold matrix:\n" << goldMatrix << "\nInitializer lists:\n";
      std::copy(std::begin(list0), std::end(list0), std::ostream_iterator<unsigned>{ std::cout, " " });
      std::cout << '\n';
      std::copy(std::begin(list1), std::end(list1), std::ostream_iterator<unsigned>{ std::cout, " " });
      std::cout << '\n';
      std::copy(std::begin(list2), std::end(list2), std::ostream_iterator<unsigned>{ std::cout, " " });
      std::cout << '\n';
    }
    EXPECT_EQ(diffMatrix.maxCoeff(), 0U);

    phiValues.at(index) = negativeValue;
  };

  test();

  test(6880U, 0U, { 6679U, 6478U, 6277U });
  test(6478U, 0U, { 6679U, 6880U, 7081U });
  test(6679U, 0U, { 6880U, 7081U, 7282U });

  test(6479U, 1U, { 6680U, 6881U, 7082U });
  test(6881U, 1U, { 6680U, 6479U, 6278U });
  test(6680U, 1U, { 6479U, 6278U, 6077U });

  test(6279U, 2U, { 6480U, 6681U, 6882U });
  test(6681U, 2U, { 6480U, 6279U, 6078U });
  test(6480U, 2U, { 6279U, 6078U, 5877U });
}

TYPED_TEST(get_stencil_indices_tests, get_stencil_indices_tests_4)
{
  using kae::detail::getStencilIndices;

  using tf = TestFixture;
  using ElemType = typename tf::ElemType;
  using Real2Type = typename tf::Real2Type;
  using IndexMatrixType = typename tf::IndexMatrixType;
  using GpuGridType = typename tf::GpuGridType;
  using InitList = std::initializer_list<unsigned>;

  const auto negativeValue = static_cast<ElemType>(-0.1);
  const auto positiveValue = static_cast<ElemType>(1.0);
  std::vector<ElemType> phiValues(GpuGridType::nx * GpuGridType::ny, negativeValue);

  const auto test = [&](unsigned index = 0U, unsigned nList = 3U, InitList list = {})
  {
    const Real2Type surfacePoint{ static_cast<ElemType>(0.45642), static_cast<ElemType>(0.33522) };
    const Real2Type normal{ static_cast<ElemType>(-0.8), static_cast<ElemType>(-0.6) };

    phiValues.at(index) = positiveValue;

    const auto indexMatrix = getStencilIndices<GpuGridType, tf::nPoints>(phiValues.data(), surfacePoint, normal);
    const auto list0 = (nList == 0) ? list : InitList{ 6880U, 6679U, 7081U };
    const auto list1 = (nList == 1) ? list : InitList{ 7082U, 6881U, 7283U };
    const auto list2 = (nList == 2) ? list : InitList{ 7083U, 7284U, 6882U };
    const IndexMatrixType goldMatrix{ list0, list1, list2 };
    const IndexMatrixType diffMatrix = indexMatrix - goldMatrix;
    if (diffMatrix.maxCoeff() != 0U)
    {
      std::cout << "\nCalculated matrix:\n" << indexMatrix << "\nGold matrix:\n" << goldMatrix << "\nInitializer lists:\n";
      std::copy(std::begin(list0), std::end(list0), std::ostream_iterator<unsigned>{ std::cout, " " });
      std::cout << '\n';
      std::copy(std::begin(list1), std::end(list1), std::ostream_iterator<unsigned>{ std::cout, " " });
      std::cout << '\n';
      std::copy(std::begin(list2), std::end(list2), std::ostream_iterator<unsigned>{ std::cout, " " });
      std::cout << '\n';
    }
    EXPECT_EQ(diffMatrix.maxCoeff(), 0U);

    phiValues.at(index) = negativeValue;
  };

  test();

  test(6679U, 0U, { 6880U, 7081U, 7282U });
  test(7081U, 0U, { 6880U, 6679U, 6478U });
  test(6880U, 0U, { 6679U, 6478U, 6277U });

  test(6881U, 1U, { 7082U, 7283U, 7484U });
  test(7283U, 1U, { 7082U, 6881U, 6680U });
  test(7082U, 1U, { 6881U, 6680U, 6479U });

  test(7284U, 2U, { 7083U, 6882U, 6681U });
  test(6882U, 2U, { 7083U, 7284U, 7485U });
  test(7083U, 2U, { 7284U, 7485U, 7686U });
}

TYPED_TEST(get_stencil_indices_tests, get_stencil_indices_tests_5)
{
  using kae::detail::getStencilIndices;

  using tf              = TestFixture;
  using ElemType        = typename tf::ElemType;
  using Real2Type       = typename tf::Real2Type;
  using IndexMatrixType = typename tf::IndexMatrixType;
  using GpuGridType     = typename tf::GpuGridType;
  using InitList        = std::initializer_list<unsigned>;

  const auto negativeValue = static_cast<ElemType>(-0.1);
  const auto positiveValue = static_cast<ElemType>(1.0);
  std::vector<ElemType> phiValues(GpuGridType::nx * GpuGridType::ny, negativeValue);

  const auto test = [&](unsigned index = 0U, unsigned nList = 3U, InitList list = {})
  {
    const Real2Type surfacePoint{ static_cast<ElemType>(0.45642), static_cast<ElemType>(0.33522) };
    const Real2Type normal{ static_cast<ElemType>(0.6), static_cast<ElemType>(0.8) };

    phiValues.at(index) = positiveValue;

    const auto indexMatrix = getStencilIndices<GpuGridType, tf::nPoints>(phiValues.data(), surfacePoint, normal);
    const auto list0 = (nList == 0) ? list : InitList{ 6678U, 6679U, 6677U };
    const auto list1 = (nList == 1) ? list : InitList{ 6477U, 6476U, 6478U };
    const auto list2 = (nList == 2) ? list : InitList{ 6275U, 6274U, 6276U };
    const IndexMatrixType goldMatrix{ list0, list1, list2 };
    const IndexMatrixType diffMatrix = indexMatrix - goldMatrix;
    if (diffMatrix.maxCoeff() != 0U)
    {
      std::cout << "\nCalculated matrix:\n" << indexMatrix << "\nGold matrix:\n" << goldMatrix << "\nInitializer lists:\n";
      std::copy(std::begin(list0), std::end(list0), std::ostream_iterator<unsigned>{ std::cout, " " });
      std::cout << '\n';
      std::copy(std::begin(list1), std::end(list1), std::ostream_iterator<unsigned>{ std::cout, " " });
      std::cout << '\n';
      std::copy(std::begin(list2), std::end(list2), std::ostream_iterator<unsigned>{ std::cout, " " });
      std::cout << '\n';
    }
    EXPECT_EQ(diffMatrix.maxCoeff(), 0U);

    phiValues.at(index) = negativeValue;
  };

  test();

  test(6679U, 0U, { 6678U, 6677U, 6676U });
  test(6677U, 0U, { 6678U, 6679U, 6680U });
  test(6678U, 0U, { 6679U, 6680U, 6681U });

  test(6476U, 1U, { 6477U, 6478U, 6479U });
  test(6478U, 1U, { 6477U, 6476U, 6475U });
  test(6477U, 1U, { 6476U, 6475U, 6474U });

  test(6274U, 2U, { 6275U, 6276U, 6277U });
  test(6276U, 2U, { 6275U, 6274U, 6273U });
  test(6275U, 2U, { 6274U, 6273U, 6272U });
}

TYPED_TEST(get_stencil_indices_tests, get_stencil_indices_tests_6)
{
  using kae::detail::getStencilIndices;

  using tf              = TestFixture;
  using ElemType        = typename tf::ElemType;
  using Real2Type       = typename tf::Real2Type;
  using IndexMatrixType = typename tf::IndexMatrixType;
  using GpuGridType     = typename tf::GpuGridType;
  using InitList        = std::initializer_list<unsigned>;

  const auto negativeValue = static_cast<ElemType>(-0.1);
  const auto positiveValue = static_cast<ElemType>(1.0);
  std::vector<ElemType> phiValues(GpuGridType::nx * GpuGridType::ny, negativeValue);

  const auto test = [&](unsigned index = 0U, unsigned nList = 3U, InitList list = {})
  {
    const Real2Type surfacePoint{ static_cast<ElemType>(0.45642), static_cast<ElemType>(0.33522) };
    const Real2Type normal{ static_cast<ElemType>(-0.6), static_cast<ElemType>(0.8) };

    phiValues.at(index) = positiveValue;

    const auto indexMatrix = getStencilIndices<GpuGridType, tf::nPoints>(phiValues.data(), surfacePoint, normal);
    const auto list0 = (nList == 0) ? list : InitList{ 6679U, 6680U, 6678U };
    const auto list1 = (nList == 1) ? list : InitList{ 6479U, 6478U, 6480U };
    const auto list2 = (nList == 2) ? list : InitList{ 6279U, 6278U, 6280U };
    const IndexMatrixType goldMatrix{ list0, list1, list2 };
    const IndexMatrixType diffMatrix = indexMatrix - goldMatrix;
    if (diffMatrix.maxCoeff() != 0U)
    {
      std::cout << "\nCalculated matrix:\n" << indexMatrix << "\nGold matrix:\n" << goldMatrix << "\nInitializer lists:\n";
      std::copy(std::begin(list0), std::end(list0), std::ostream_iterator<unsigned>{ std::cout, " " });
      std::cout << '\n';
      std::copy(std::begin(list1), std::end(list1), std::ostream_iterator<unsigned>{ std::cout, " " });
      std::cout << '\n';
      std::copy(std::begin(list2), std::end(list2), std::ostream_iterator<unsigned>{ std::cout, " " });
      std::cout << '\n';
    }
    EXPECT_EQ(diffMatrix.maxCoeff(), 0U);

    phiValues.at(index) = negativeValue;
  };

  test();

  test(6680U, 0U, { 6679U, 6678U, 6677U });
  test(6678U, 0U, { 6679U, 6680U, 6681U });
  test(6679U, 0U, { 6680U, 6681U, 6682U });

  test(6478U, 1U, { 6479U, 6480U, 6481U });
  test(6480U, 1U, { 6479U, 6478U, 6477U });
  test(6479U, 1U, { 6478U, 6477U, 6476U });

  test(6278U, 2U, { 6279U, 6280U, 6281U });
  test(6280U, 2U, { 6279U, 6278U, 6277U });
  test(6279U, 2U, { 6278U, 6277U, 6276U });
}

TYPED_TEST(get_stencil_indices_tests, get_stencil_indices_tests_7)
{
  using kae::detail::getStencilIndices;

  using tf = TestFixture;
  using ElemType = typename tf::ElemType;
  using Real2Type = typename tf::Real2Type;
  using IndexMatrixType = typename tf::IndexMatrixType;
  using GpuGridType = typename tf::GpuGridType;
  using InitList = std::initializer_list<unsigned>;

  const auto negativeValue = static_cast<ElemType>(-0.1);
  const auto positiveValue = static_cast<ElemType>(1.0);
  std::vector<ElemType> phiValues(GpuGridType::nx * GpuGridType::ny, negativeValue);

  const auto test = [&](unsigned index = 0U, unsigned nList = 3U, InitList list = {})
  {
    const Real2Type surfacePoint{ static_cast<ElemType>(0.45642), static_cast<ElemType>(0.33522) };
    const Real2Type normal{ static_cast<ElemType>(0.6), static_cast<ElemType>(-0.8) };

    phiValues.at(index) = positiveValue;

    const auto indexMatrix = getStencilIndices<GpuGridType, tf::nPoints>(phiValues.data(), surfacePoint, normal);
    const auto list0 = (nList == 0) ? list : InitList{ 6879U, 6880U, 6878U };
    const auto list1 = (nList == 1) ? list : InitList{ 7080U, 7079U, 7081U };
    const auto list2 = (nList == 2) ? list : InitList{ 7280U, 7279U, 7281U };
    const IndexMatrixType goldMatrix{ list0, list1, list2 };
    const IndexMatrixType diffMatrix = indexMatrix - goldMatrix;
    if (diffMatrix.maxCoeff() != 0U)
    {
      std::cout << "\nCalculated matrix:\n" << indexMatrix << "\nGold matrix:\n" << goldMatrix << "\nInitializer lists:\n";
      std::copy(std::begin(list0), std::end(list0), std::ostream_iterator<unsigned>{ std::cout, " " });
      std::cout << '\n';
      std::copy(std::begin(list1), std::end(list1), std::ostream_iterator<unsigned>{ std::cout, " " });
      std::cout << '\n';
      std::copy(std::begin(list2), std::end(list2), std::ostream_iterator<unsigned>{ std::cout, " " });
      std::cout << '\n';
    }
    EXPECT_EQ(diffMatrix.maxCoeff(), 0U);
    phiValues.at(index) = negativeValue;
  };

  test();

  test(6880U, 0U, { 6879U, 6878U, 6877U });
  test(6878U, 0U, { 6879U, 6880U, 6881U });
  test(6879U, 0U, { 6880U, 6881U, 6882U });

  test(7079U, 1U, { 7080U, 7081U, 7082U });
  test(7081U, 1U, { 7080U, 7079U, 7078U });
  test(7080U, 1U, { 7079U, 7078U, 7077U });

  test(7279U, 2U, { 7280U, 7281U, 7282U });
  test(7281U, 2U, { 7280U, 7279U, 7278U });
  test(7280U, 2U, { 7279U, 7278U, 7277U });
}

TYPED_TEST(get_stencil_indices_tests, get_stencil_indices_tests_8)
{
  using kae::detail::getStencilIndices;

  using tf              = TestFixture;
  using ElemType        = typename tf::ElemType;
  using Real2Type       = typename tf::Real2Type;
  using IndexMatrixType = typename tf::IndexMatrixType;
  using GpuGridType     = typename tf::GpuGridType;
  using InitList        = std::initializer_list<unsigned>;

  const auto negativeValue = static_cast<ElemType>(-0.1);
  const auto positiveValue = static_cast<ElemType>(1.0);
  std::vector<ElemType> phiValues(GpuGridType::nx * GpuGridType::ny, negativeValue);

  const auto test = [&](unsigned index = 0U, unsigned nList = 3U, InitList list = {})
  {
    const Real2Type surfacePoint{ static_cast<ElemType>(0.45642), static_cast<ElemType>(0.33522) };
    const Real2Type normal{ static_cast<ElemType>(-0.6), static_cast<ElemType>(-0.8) };

    phiValues.at(index) = positiveValue;

    const auto indexMatrix = getStencilIndices<GpuGridType, tf::nPoints>(phiValues.data(), surfacePoint, normal);
    const auto list0 = (nList == 0) ? list : InitList{ 6880U, 6881U, 6879U };
    const auto list1 = (nList == 1) ? list : InitList{ 7082U, 7081U, 7083U };
    const auto list2 = (nList == 2) ? list : InitList{ 7284U, 7283U, 7285U };
    const IndexMatrixType goldMatrix{ list0, list1, list2 };
    const IndexMatrixType diffMatrix = indexMatrix - goldMatrix;
    if (diffMatrix.maxCoeff() != 0U)
    {
      std::cout << "\nCalculated matrix:\n" << indexMatrix << "\nGold matrix:\n" << goldMatrix << "\nInitializer lists:\n";
      std::copy(std::begin(list0), std::end(list0), std::ostream_iterator<unsigned>{ std::cout, " " });
      std::cout << '\n';
      std::copy(std::begin(list1), std::end(list1), std::ostream_iterator<unsigned>{ std::cout, " " });
      std::cout << '\n';
      std::copy(std::begin(list2), std::end(list2), std::ostream_iterator<unsigned>{ std::cout, " " });
      std::cout << '\n';
    }
    EXPECT_EQ(diffMatrix.maxCoeff(), 0U);

    phiValues.at(index) = negativeValue;
  };

  test();

  test(6881U, 0U, { 6880U, 6879U, 6878U });
  test(6879U, 0U, { 6880U, 6881U, 6882U });
  test(6880U, 0U, { 6881U, 6882U, 6883U });

  test(7081U, 1U, { 7082U, 7083U, 7084U });
  test(7083U, 1U, { 7082U, 7081U, 7080U });
  test(7082U, 1U, { 7081U, 7080U, 7079U });

  test(7283U, 2U, { 7284U, 7285U, 7286U });
  test(7285U, 2U, { 7284U, 7283U, 7282U });
  test(7284U, 2U, { 7283U, 7282U, 7281U });
}

} // namespace tests
