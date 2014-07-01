/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2013 by O. Parcollet
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
#include <iostream>
#include <type_traits>
#include <triqs/arrays.hpp>
#include <triqs/mpi.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace triqs;
using namespace triqs::arrays;
using namespace triqs::mpi;

 struct my_object {

  array<double, 1> a, b;


  struct mpi_implementation : mpi_impl_tuple<my_object>{};

  my_object() = default;
  template <typename Tag> my_object(mpi_lazy<Tag, my_object> x) : my_object() { operator=(x); }

  template <typename Tag> my_object &operator=(mpi_lazy<Tag, my_object> x) {
   mpi_impl_tuple<my_object>::complete_operation(*this, x);
   return *this;
  }
 };

  auto view_as_tuple(my_object const &x) DECL_AND_RETURN(std::tie(x.a, x.b));
  auto view_as_tuple(my_object &x) DECL_AND_RETURN(std::tie(x.a, x.b));


int main(int argc, char* argv[]) {

 mpi::environment env(argc, argv);
 mpi::communicator world;

 auto obj = my_object();

  //using ARR = array<double,2>;
  using ARR = array<std::complex<double>,2>;

  ARR A(7,3), B, AA;

  triqs::clef::placeholder<0> i_;
  triqs::clef::placeholder<1> j_;

  A(i_,j_) << i_ + 10*j_;

  B = mpi::scatter(A, world);

  ARR C = mpi::scatter(A, world);

  std::ofstream out("node" + std::to_string(world.rank()));
  out << "  A = " << A << std::endl;
  out << "  B = " << B << std::endl;
  out << "  C = " << C << std::endl;

  B *=-1;
  //AA = A;
  AA() =0;

  AA = mpi::gather(B, world);
  out << " AA = " << AA << std::endl;

  mpi::broadcast(AA, world);
  out << " cast AA = " << AA << std::endl;

  AA() =0;
 

  AA = mpi::allgather(B, world);
  out << " AA = " << AA << std::endl;

/*
 mpi::environment env(argc, argv);
 mpi::communicator world;

 array<long,2> A {{1,2}, {3,4}}, C(2,2);

 // boost mpi
 boost::mpi::reduce (world, A,C, std::c14::plus<>(),0);
 int s= world.size();
 if (world.rank() ==0) std::cout<<" C = "<<C<< "  should be "<< std::endl << array<long,2>(s*A) <<std::endl;

 // triqs mpi
 C = A;
 mpi::reduce_in_place (world, C);
 if (world.rank() ==0) std::cout<<" C = "<<C<< "  should be "<< std::endl << array<long,2>(s*A) <<std::endl;

 // test rvalue views
 C = A;
 mpi::reduce_in_place (world, C());
 if (world.rank() ==0) std::cout<<" C = "<<C<< "  should be "<< std::endl << array<long,2>(s*A) <<std::endl;

 // more complex class
 auto x = S { { {1,2},{3,4}}, { 1,2,3,4}};
 mpi::reduce_in_place (world, x);
 if (world.rank() ==0) std::cout<<" S.x = "<<x.x<<" S.y = "<<x.y<<std::endl;

 // a simple number
 double y = 1+world.rank(), z=0;
 mpi::reduce(world,y,z);
 if (world.rank() ==0) std::cout<<" y = "<<y<< "  should be "<< 1+world.rank()<<std::endl;
 if (world.rank() ==0) std::cout<<" z = "<<z<< "  should be "<< s*(s+1)/2 <<std::endl;
 mpi::reduce_in_place(world,y);
 if (world.rank() ==0) std::cout<<" y = "<<y<< "  should be "<< s*(s+1)/2 <<std::endl;

 mpi::broadcast(world,C);

 // reduced x,y,C, .... a variadic form
 mpi::reduce_in_place_v (world, x,y,C);

 // more complex object
 auto ca = array< array<int,1>, 1 > { array<int,1>{1,2}, array<int,1>{3,4}};
 auto cC = ca;
 mpi::reduce_in_place (world, cC);
 if (world.rank() ==0) std::cout<<" cC = "<<cC<< std::endl;
*/
 return 0;
}

