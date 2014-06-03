/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2014 by O. Parcollet
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#pragma once
#include <triqs/utility/c14.hpp>
#include <triqs/utility/tuple_tools.hpp>
#include <boost/mpi.hpp>
#include <mpi.h>

namespace triqs {
namespace mpi {

 struct environment {
  environment(int argc, char *argv[]) { MPI_Init(&argc, &argv); }
  ~environment() { MPI_Finalize(); }
 };

 class  communicator {
  MPI_Comm _com = MPI_COMM_WORLD;
  
  public : 
  communicator() = default;

  MPI_Comm get() const { return _com;}

  int rank() const {
   int num;
   MPI_Comm_rank(_com, &num);
   return num;
  }

  int size() const {
   int num;
   MPI_Comm_size(_com, &num);
   return num;
  }

  void barrier() const { MPI_Barrier(_com); }
 };

 namespace tag {
  struct reduce {};
  struct allreduce {};
  struct scatter {};
  struct gather {};
  struct allgather {};
 }

 template <typename T, typename Enable = void> struct mpi_impl;

 template<typename T> void do_nothing(T...) {}

 // If type T has a mpi_implementation nested struct, then it is mpi_impl<T>.
 template <typename T> struct mpi_impl<T, decltype(do_nothing(T::mpi_implementation()))> : T::mpi_implementation {};

 // ----------------------------------------
 // ------- top level functions -------
 // ----------------------------------------

 // ----- functions that can be lazy -------

 template <typename T>
 auto reduce(T const &x, communicator c = {}, int root = 0) DECL_AND_RETURN(mpi_impl<T>::invoke(tag::reduce(), c, x, root));
 template <typename T>
 auto scatter(T const &x, communicator c = {}, int root = 0) DECL_AND_RETURN(mpi_impl<T>::invoke(tag::scatter(), c, x, root));
 template <typename T>
 auto gather(T const &x, communicator c = {}, int root = 0) DECL_AND_RETURN(mpi_impl<T>::invoke(tag::gather(), c, x, root));
 template <typename T>
 auto allreduce(T const &x, communicator c = {}, int root = 0) DECL_AND_RETURN(mpi_impl<T>::invoke(tag::allreduce(), c, x, root));
 template <typename T>
 auto allgather(T const &x, communicator c = {}, int root = 0) DECL_AND_RETURN(mpi_impl<T>::invoke(tag::allgather(), c, x, root));

 // ----- functions that cannot be lazy -------

 template <typename T> void reduce_in_place(T &x, communicator c = {}, int root = 0) { mpi_impl<T>::reduce_in_place(c, x, root); }
 template <typename T> void broadcast(T &x, communicator c = {}, int root = 0) { mpi_impl<T>::broadcast(c, x, root); }

 // transformation type -> mpi types
 template <class T> struct mpi_datatype;
#define D(T, MPI_TY)                                                                                                             \
 template <> struct mpi_datatype<T> {                                                                                            \
  static MPI_Datatype invoke() { return MPI_TY; }                                                                                \
 };
 D(int, MPI_INT) D(long, MPI_LONG) D(double, MPI_DOUBLE) D(float, MPI_FLOAT) D(std::complex<double>, MPI_DOUBLE_COMPLEX);
#undef D

 /** ------------------------------------------------------------
   *  basic types
   *  ----------------------------------------------------------  **/

 template <typename T> struct mpi_impl_basic {

  static MPI_Datatype D() { return mpi_datatype<T>::invoke(); }

  static T invoke(tag::reduce, communicator c, T a, int root) {
   T b;
   MPI_Reduce(&a, &b, 1, D(), MPI_SUM, root, c.get());
   return b;
  }

  static T invoke(tag::allreduce, communicator c, T a, int root) {
   T b;
   MPI_Allreduce(&a, &b, 1, D(), MPI_SUM, root, c.get());
   return b;
  }

  static void reduce_in_place(communicator c, T &a, int root) {
   MPI_Reduce((c.rank() == root ? MPI_IN_PLACE : &a), &a, 1, D(), MPI_SUM, root, c.get());
  }

  static void allreduce_in_place(communicator c, T &a, int root) { MPI_Allreduce(MPI_IN_PLACE, &a, 1, D(), MPI_SUM, root, c.get()); }

  static void broadcast(communicator c, T &a, int root) { MPI_Bcast(&a, 1, D(), root, c.get()); }

 };

 template <typename T>
 struct mpi_impl<T, std14::enable_if_t<std::is_arithmetic<T>::value || triqs::is_complex<T>::value>> : mpi_impl_basic<T> {};

 /** ------------------------------------------------------------
   *  basic C arrays of basic types
   *  ----------------------------------------------------------  **/

 template <typename T> struct mpi_impl_basic_arrays {

  static MPI_Datatype D() { return mpi_datatype<T>::invoke(); }

  static void invoke(tag::reduce, communicator c, const T *a, T *b, long slow_size, long slow_stride, int root) {
   MPI_Reduce(a, b, slow_size * slow_stride, D(), MPI_SUM, root, c.get());
  }

  static void invoke(tag::allreduce, communicator c, const T *a, T *b, long slow_size, long slow_stride,  int root) {
   MPI_Allreduce(a, b, slow_size * slow_stride, D(), MPI_SUM, root, c.get());
  }

  static void reduce_in_place(communicator c, T *a, long slow_size, long slow_stride, int root) {
   MPI_Reduce((c.rank() == root ? MPI_IN_PLACE : a), a, slow_size * slow_stride, D(), MPI_SUM, root, c.get());
  }

  static void allreduce_in_place(communicator c, T *a, long slow_size, long slow_stride, int root) {
   MPI_Allreduce(MPI_IN_PLACE, a, slow_size * slow_stride, D(), MPI_SUM, root, c.get());
  }

  static void broadcast(communicator c, T &a, size_t size, int root) { MPI_Bcast(&a, size, D(), root, c.get()); }

  struct _displ {
   int recvcount;
   std::vector<int> sendcounts, displs;
 
   static long slice_length (size_t imax, communicator c, int r) {
     auto imin = 0;
     long j = (imax - imin + 1) / c.size();
     long i = imax - imin + 1 - c.size() * j;
     auto r_min = (r <= i - 1 ? imin + r * (j + 1) : imin + r * j + i);
     auto r_max = (r <= i - 1 ? imin + (r + 1) * (j + 1) - 1 : imin + (r + 1) * j + i - 1);
     return r_max - r_min + 1;
    };


   _displ(communicator c, long slow_size, long slow_stride) : sendcounts(c.size()), displs(c.size() + 1, 0) {

    recvcount = slice_length(slow_size - 1, c, c.rank()) * slow_stride;
    for (int r = 0; r < c.size(); ++r) {
     sendcounts[r] = slice_length(slow_size * slow_stride - 1, c, r) * slow_stride;
     displs[r + 1] = sendcounts[r] + displs[r];
    }
   }
  };

  static void invoke(tag::scatter, communicator c, T const *a, T *b, long slow_size, long slow_stride, int root) {
   auto d = _displ(c,slow_size, slow_stride);
    MPI_Scatterv(a, &d.sendcounts[0], &d.displs[0], D(), b, d.recvcount, D(), root, c.get());
  }

  static void invoke(tag::gather, communicator c, T const *a, T *b, long slow_size, long slow_stride, int root) {
   auto d = _displ(c,slow_size, slow_stride);
   MPI_Gatherv(a, &d.sendcounts[0], D(), b, d.recvcount, &d.displs[0], D(), root, c.get());
  }
 };

 // When value_type is a basic type, we can directly call the C API
 template <typename T> struct mpi_impl_std_vector_basic {

  using impl = mpi_impl_basic_arrays<T>;

  static void reduce_in_place(communicator c, std::vector<T> &a, int root) {
   impl::_reduce_in_place(c, a.data(), a.size(), 1, root);
  }
  static void allreduce_in_place(communicator c, std::vector<T> &a, int root) {
   impl::_allreduce_in_place(c, a.data(), a.size(), 1, root);
  }
  static void broadcast(communicator c, std::vector<T> &a, int root) { impl::broadcast(c, a.data(), a.size(), 1, root); }

  static std::vector<T> invoke(tag::reduce, communicator c, T const &a, int root) {
   std::vector<T> b(a.size());
   impl::invoke(tag::reduce(), c, a.data(), b.data(), a.size(), 1, root);
   return b;
  }

  static std::vector<T> invoke(tag::allreduce, communicator c, std::vector<T> const &a, int root) {
   std::vector<T> b(a.size());
   impl::invoke(tag::allreduce(), c, a.data(), b.data(), a.size(), 1, root);
   return b;
  }

  static std::vector<T> invoke(tag::scatter, communicator c, std::vector<T> const &a, int root) {
   std::vector<T> b(impl::slice_length(a.size() - 1, c, c.rank()));
   impl::invoke(tag::scatter(), c, a.data(), b.data(), a.size(), 1, root);
   return b;
  }

  static std::vector<T> invoke(tag::gather, communicator c, std::vector<T> const &a, int root) {
   long size = reduce(a.size(), c,root);
   std::vector<T> b(size);
   impl::invoke(tag::gather(), c, a.data(), b.data(), size, 1, root);
   return b;
  }

 static std::vector<T> invoke(tag::allgather, communicator c, std::vector<T> const &a, int root) {
   long size = reduce(a.size(), c,root);
   std::vector<T> b(size);
   impl::invoke(tag::allgather(), c, a.data(), b.data(), size, 1, root);
   return b;
  }

 };


 template <typename T>
 struct mpi_impl<std::vector<T>,
                 std14::enable_if_t<std::is_arithmetic<T>::value || triqs::is_complex<T>::value>> : mpi_impl_std_vector_basic<T> {};

 // vector <T> for T non basic


 template <typename Tag, typename T> struct mpi_lazy {
  T const &ref;
  int root;
  communicator c;
 };

 // When value_type is a basic type, we can directly call the C API
 template <typename A> struct mpi_impl_triqs_arrays {

  using a_t = typename A::value_type;
  using impl = mpi_impl_basic_arrays<a_t>;

  static void check(A const &a) {
   if (!has_contiguous_data(a)) TRIQS_RUNTIME_ERROR << "Non contiguous view in mpi_reduce_in_place";
  }

  static void reduce_in_place(communicator c, A &a, int root) {
   check(a);
   impl::reduce_in_place(c, a.data_start(), a.domain().length[0](), a.domain().stride[0], root);
  }

  static void allreduce_in_place(communicator c, A &a, int root) {
   check(a);
   impl::allreduce_in_place(c, a.data_start(), a.domain().length[0](), a.domain().stride[0], root);
  }

  static void broadcast(communicator c, A &a, int root) {
   check(a);
   impl::broadcast(c, a.data_start(), a.domain().length[0](), a.domain().stride[0], root);
  }

  template <typename Tag> static mpi_lazy<Tag, A> invoke(Tag, communicator c, A const &a, int root) {
   check(a);
   return {a, root, c};
  }

  template <typename Tag> static void complete_operation(A &target, mpi_lazy<Tag, A> laz) {
   impl::invoke(Tag(), laz.c, laz.ref, target, laz.ref.domain().length[0](), laz.ref.domain().stride[0], laz.root);
  }
 };

 // constructor and = for the arrays.

 /** ------------------------------------------------------------
  *  Type which are recursively treated by reducing them to a tuple
  *  of smaller objects.
  *  ----------------------------------------------------------  **/
 template <typename T> struct mpi_impl_tuple {

  template <typename Tag> static mpi_lazy<Tag, T> invoke(Tag, communicator c, T const &a, int root) {
   check(a);
   return {a, root, c};
  }

#ifdef __cpp_generic_lambdas
  static void reduce_in_place(communicator c, T &a, int root) {
   tuple::for_each([c, root](auto &x) { triqs::mpi::reduce_in_place(c, x, root); }, view_as_tuple(a));
  }

  static void broadcast(communicator c, T &a, int root) {
   tuple::for_each([c, root](auto &x) { triqs::mpi::broadcast(c, x, root); }, view_as_tuple(a));
  }

  template <typename Tag> static void complete_operation(T &target, mpi_lazy<Tag, T> laz) {
   auto l = [laz](auto & t, auto & s) {
    t = triqs::mpi::mpi_impl<T>::invoke(Tag(), laz.c, s, laz.root);
   };
   //tuple::for_each_on_zip2(l, view_as_tuple(target), view_as_tuple(laz.ref));
  }
#else

  struct aux1{
    communicator c;
    int root;

    template <typename T1>
    void operator()(T1 &x) const { triqs::mpi::reduce_in_place(c, x, root); }
  };

  static void reduce_in_place(communicator c, T &a, int root) {
   tuple::for_each(aux1{c,root}, view_as_tuple(a));
  }

  struct aux2{
    communicator c;
    int root;

    template <typename T2>
    void operator()(T2 &x) const { triqs::mpi::broadcast(c, x, root); }
  };

  static void broadcast(communicator c, T &a, int root) {
   tuple::for_each(aux2{c,root}, view_as_tuple(a));
  }

  template <typename Tag>
  struct aux3{
    mpi_lazy<Tag, T> laz;

    template <typename T1, typename T2>
    void operator()(T1 &t, T2 &s) const { t = triqs::mpi::mpi_impl<T>::invoke(Tag(), laz.c, laz.s); }
  };

  template <typename Tag> static void complete_operation(T &target, mpi_lazy<Tag, T> laz) {
   auto l = aux3<Tag>{laz};
   //tuple::for_each_on_zip2(l, view_as_tuple(target), view_as_tuple(laz.ref));
  }
#endif

 };
 /** ------------------------------------------------------------
   *  Type which we use boost::mpi
   *  ----------------------------------------------------------  **/

 template <typename T> struct mpi_impl_boost_mpi {

  static T invoke(tag::reduce, communicator c, T const &a, int root) {
   T b;
   boost::mpi::reduce(c, a, b, std::c14::plus<>(), root);
   return b;
  }

  static T invoke(tag::allreduce, communicator c, T const &a, int root) {
   T b;
   boost::mpi::all_reduce(c, a, b, std::c14::plus<>(), root);
   return b;
  }

  static void reduce_in_place(communicator c, T &a, int root) { boost::mpi::reduce(c, a, a, std::c14::plus<>(), root); }
  static void broadcast(communicator c, T &a, int root) { boost::mpi::broadcast(c, a, root); }

  static void scatter(communicator c, T const &, int root) = delete;
  static void gather(communicator c, T const &, int root) = delete;
  static void allgather(communicator c, T const &, int root) = delete;
 };

 // default
 template <typename T> struct mpi_impl<T> : mpi_impl_boost_mpi<T> {};

}}//namespace




