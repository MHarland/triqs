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

 template <typename Tag, typename T> struct mpi_lazy {
  T const &ref;
  int root;
  communicator c;
 };

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
 
}}//namespace




