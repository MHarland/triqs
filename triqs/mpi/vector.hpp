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
#include "./_basic_array.hpp"

namespace triqs {
namespace mpi {

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

 }}//namespace




