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
#include <triqs/arrays.hpp>
#include "./_basic_array.hpp"

namespace triqs {
namespace mpi {
 
 //--------------------------------------------------------------------------------------------------------
 template <typename Tag, typename A> struct mpi_lazy_array {
  A const &ref;
  int root;
  communicator c;

  using domain_type = typename A::domain_type;

  long slow_size() const { return first_dim(ref); }
  long slow_size_target() const { return _slow_size_target(Tag()); }
  long slow_stride() const { return ref.indexmap().strides()[0]; }

  /// compute the array domain of the target array
  domain_type domain() const {
   auto dims = ref.shape();
   dims[0] = slow_size_target();
   return domain_type{dims};
  }

  private:

  long _slow_size_target(tag::scatter) const {
   long slow_size = first_dim(ref);
   return mpi_impl_basic_arrays<A>::slice_length(slow_size - 1, c, c.rank());
  }

  /// TODO !!!
  /// Only true if gather results from a previously scattered array : otherwise need gather the dims...
  long _slow_size_target(tag::gather) const {
   long slow_size_total = 0, slow_size = first_dim(ref);
   MPI_Reduce(&slow_size, &slow_size_total, 1, mpi_datatype<long>::invoke(), MPI_SUM, root, c.get());
   return slow_size_total;
  }

  long _slow_size_target(tag::allgather) const {
   long slow_size_total = 0, slow_size = first_dim(ref);
   MPI_Allreduce(&slow_size, &slow_size_total, 1, mpi_datatype<long>::invoke(), MPI_SUM, c.get());
   return slow_size_total;
  }
 };

 //--------------------------------------------------------------------------------------------------------

 // When value_type is a basic type, we can directly call the C API
 template <typename A> class mpi_impl_triqs_arrays {

  using impl = mpi_impl_basic_arrays<typename A::value_type>;

  static void check(A const &a) {
   if (!has_contiguous_data(a)) TRIQS_RUNTIME_ERROR << "Non contiguous view in mpi_reduce_in_place";
  }

  public:

  static void reduce_in_place(communicator c, A &a, int root) {
   check(a);
   impl::reduce_in_place(c, a.data_start(), a.indexmap().lengths()[0], a.indexmap().strides()[0], root);
  }

  static void allreduce_in_place(communicator c, A &a, int root) {
   check(a);
   impl::allreduce_in_place(c, a.data_start(), a.indexmap().lengths()[0], a.indexmap().strides()[0], root);
  }

  static void broadcast(communicator c, A &a, int root) {
   check(a);
   auto sh = a.shape();
   MPI_Bcast(&sh[0], sh.size(), mpi_datatype<typename decltype(sh)::value_type>::invoke(), root, c.get());
   if (c.rank() != root) a.resize(sh);
   impl::broadcast(c, a.data_start(), a.indexmap().lengths()[0], a.indexmap().strides()[0], root);
  }

  template <typename Tag> static mpi_lazy_array<Tag, A> invoke(Tag, communicator c, A const &a, int root) {
   check(a);
   return {a, root, c};
  }

 };

 template <typename A>
 struct mpi_impl<A, std14::enable_if_t<triqs::arrays::is_amv_value_or_view_class<A>::value>> : mpi_impl_triqs_arrays<A> {};

}

//--------------------------------------------------------------------------------------------------------

namespace arrays {

 template <typename Tag, typename A> struct ImmutableCuboidArray<triqs::mpi::mpi_lazy_array<Tag, A>> : ImmutableCuboidArray<A> {};

 namespace assignment {

  template <typename LHS, typename Tag, typename A>
  struct is_special<LHS, triqs::mpi::mpi_lazy_array<Tag, A>> : std::true_type {};

  template <typename LHS, typename A, typename Tag> struct impl<LHS, triqs::mpi::mpi_lazy_array<Tag, A>, 'E', void> {

   using mpi_impl_basic_array = triqs::mpi::mpi_impl_basic_arrays<typename A::value_type>;
   using laz_t = triqs::mpi::mpi_lazy_array<Tag, A>;
   LHS &lhs;
   laz_t laz;

   impl(LHS &lhs_, laz_t laz_) : lhs(lhs_), laz(laz_) {}

   void invoke() { _invoke(Tag()); }

   private:

   void _invoke(triqs::mpi::tag::scatter) {
    lhs.resize(laz.domain());
    mpi_impl_basic_array::scatter(laz.c, laz.ref.data_start(), lhs.data_start(), laz.slow_size(), laz.slow_stride(),
                                  laz.root);
   }
   void _invoke(triqs::mpi::tag::gather) {
    lhs.resize(laz.domain());
    mpi_impl_basic_array::gather(laz.c, laz.ref.data_start(), lhs.data_start(), laz.slow_size(), laz.slow_size_target(),
                                 laz.slow_stride(), laz.root);
   }
   void _invoke(triqs::mpi::tag::allgather) {
    lhs.resize(laz.domain());
    mpi_impl_basic_array::allgather(laz.c, laz.ref.data_start(), lhs.data_start(), laz.slow_size(), laz.slow_size_target(),
                                 laz.slow_stride());
   }
  };
 }
}

}
