#define TRIQS_ARRAYS_ENFORCE_BOUNDCHECK
#include <triqs/gfs.hpp>
using namespace triqs::gfs;
using namespace triqs::arrays;
#define TEST(X) std::cout << BOOST_PP_STRINGIZE((X)) << " ---> " << (X) << std::endl << std::endl;
#include <triqs/gfs/local/functions.hpp>

int main() {
 try {
  double beta = 1, a = 1;

  // construct
  auto gw_n = gf<imfreq, matrix_valued, no_tail>{{beta, Fermion}, {2, 2}};
  auto gt_n = gf<imtime, matrix_valued, no_tail>{{beta, Fermion, 100}, {2, 2}};

  // make a view from a g with a tail
  auto gw = gf<imfreq, matrix_valued>{{beta, Fermion}, {2, 2}};
  auto gt = gf<imtime, matrix_valued>{gf_mesh<imtime>{beta, Fermion, 100}, {2, 2}};

  auto vw = make_gf_view_without_tail(gw);
  auto vt = make_gf_view_without_tail(gt);

  triqs::clef::placeholder<0> tau_;
  // should not and does not compile : improve error message ???
  // gt(tau_) << exp(-a * tau_) / (1 + exp(-beta * a));
  vt(tau_) << exp(-a * tau_) / (1 + exp(-beta * a));

  // test hdf5
  H5::H5File file("ess_g_notail.h5", H5F_ACC_TRUNC);
  h5_write(file, "g", vt);

  // rebuilding a new gf...
  auto g3 = make_gf_from_g_and_tail(vw, gw.singularity());
  // need to test all this....
 
  //test antiperiodicity
  auto Gt = gf<imtime, scalar_valued, no_tail>{ { beta, Fermion, 1000 }, {  } };
  Gt(tau_) << exp(-tau_);

  TEST(Gt(0.01));
  TEST(Gt(.5));
  TEST(Gt(.9));
  TEST(Gt(-.1));//should be equal to line above

  //fourier
   triqs::gfs::local::tail t(1,1);
  gw_n (tau_) << 1/(tau_-1.);
  auto gt_with_full_tail = make_gf_from_inverse_fourier(make_gf_from_g_and_tail(gw_n, gw.singularity()));
  auto gt_tail_with_one_term = make_gf_from_inverse_fourier(make_gf_from_g_and_tail(gw_n, t));
  TEST(gt_with_full_tail(.5));
  TEST(gt_tail_with_one_term(.5));
 }
 TRIQS_CATCH_AND_ABORT;
}
