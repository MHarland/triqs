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
#include "./base.hpp"

namespace triqs {
namespace mpi {
 /** ------------------------------------------------------------
   *  basic C arrays of basic types
   *  Implementation class only
   *  Handles arrays of dim 1, as C pointers. Used later by std::vector, triqs::arrays
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

  static void broadcast(communicator c, T *a, long slow_size, long slow_stride, int root) {
   MPI_Bcast(a, slow_size * slow_stride, D(), root, c.get());
  }

   static long slice_length (size_t imax, communicator c, int r) {
     auto imin = 0;
     long j = (imax - imin + 1) / c.size();
     long i = imax - imin + 1 - c.size() * j;
     auto r_min = (r <= i - 1 ? imin + r * (j + 1) : imin + r * j + i);
     auto r_max = (r <= i - 1 ? imin + (r + 1) * (j + 1) - 1 : imin + (r + 1) * j + i - 1);
     return r_max - r_min + 1;
    };

  struct _displ {
   int recvcount;
   std::vector<int> sendcounts, displs;
 
   _displ(communicator c, long slow_size, long slow_stride) : sendcounts(c.size()), displs(c.size() + 1, 0) {

    recvcount = slice_length(slow_size - 1, c, c.rank()) * slow_stride;
    for (int r = 0; r < c.size(); ++r) {
     sendcounts[r] = slice_length(slow_size - 1, c, r) * slow_stride;
     displs[r + 1] = sendcounts[r] + displs[r];
    }
   }
  };

  static void scatter(communicator c, T const *a, T *b, long slow_size, long slow_stride, int root) {
   auto d = _displ(c,slow_size, slow_stride);
   MPI_Scatterv((void*)a, &d.sendcounts[0], &d.displs[0], D(), (void*)b, d.recvcount, D(), root, c.get());
  }

  static void gather(communicator c, T const *a, T *b, long slow_size_on_node, long slow_size_total, long slow_stride, int root) {
   auto d = _displ(c, slow_size_total, slow_stride);
   int sendcount = slow_size_on_node * slow_stride;
   MPI_Gatherv((void *)a, sendcount, D(), (void *)b, &d.sendcounts[0], &d.displs[0], D(), root, c.get());
  }
 
  static void allgather(communicator c, T const *a, T *b, long slow_size_on_node, long slow_size_total, long slow_stride) {
   auto d = _displ(c, slow_size_total, slow_stride);
   int sendcount = slow_size_on_node * slow_stride;
   MPI_Allgatherv((void *)a, sendcount, D(), (void *)b, &d.sendcounts[0], &d.displs[0], D(), c.get());
  }
 };
}}//namespace

