Green functions
==========================

SHALL I explain the data structure and the HDF5 after ???


Data structure
---------------------

The Green functions in TRIQS are quite general.
They can store any functions stored on a discretized mesh.
Hence they are more general than "Green" functions, but we keep the name
since there are already so many objects called "functions" around...


A Green function is simply a function, which has : 

* a `domain` for its variable(s) (e.g. Matsubara/real time/frequencies, Legendre coefficients).

* a `target` space, i.e. the value of the Green function which can be : 
   
   * a scalar (double, complex)
   * a matrix, 
   * another Green function (e.g. currying multi-variable Green functions)
 

A discretized Green function has : 

* a `mesh`, which is the representation of the domain in the computer.
  
  * Typically it is a real mesh on a simple domain (e.g. a mesh on a Brillouin zone), 
    but it should be understood here in a more general and abstract way,
    as the representation of the domain in the computer (e.g. a set of Legendre coefficients).

  * a mesh has an associated domain.

* a `target` space.

SHALL WE DESCRIBE IT HERE ?


HDF5 format
-----------------------

Generalities
^^^^^^^^^^^^^

In TRIQS, the main data storage format is `HDF5 <http://www.hdfgroup.org/HDF5/>`_ (Hierarchical Data Format v5).

PUT HERE " WHAT IS HDF5" from doc... for newcomers.

In this document, we explain how each object is stored.

**Guiding principles**

* HDF5 files must be readable by Python, C++ TRIQS containers and foreign projects as well.
  As such, their structure have to be as simple as possible.
  
  In particular, no project-specific serialization algorithm can be used 
  (e.g. serializing an object into a string, and writing the string),
  unless it is *really* necessary and the algorithm is simple, stable  and *fully documented*. 
  
* Any object more complex than a single array is stored in a specific hdf5 subgroup.

* This subgroup is tagged with an attribute : 

   * named `TRIQS_HDF5_data_scheme`
   * of type "fixed size string", C-style (i.e. H5T_C_S1).
   * which is unique for each type.

  This allows an easy retrieval of the object by the Python interface (Python associates a type to each TRIQS_HDF5_data_scheme).

WE NEED TO AGREE ON THE TYPE in HDF5 FILES (Native,  ???).

Notation : we use Python notation to indicate the meaning of arrays.

Arrays
^^^^^^^^^

All arrays are stored in the C ordering conventions (first index is the slowest),
which is the default for C++ and Python numpy.

Complex numbers
^^^^^^^^^^^^^^^^^

* There is no standard representation of the complex numbers in hdf5.

* We choose to write d-dimensionnal array of complex numbers as
  d+1 dimensional array of real (double) numbers.

  More precisely, given an array `A` of dimension `d` of complex,  we stored it 
  as a `d+1` dimensions array of real numbers `B` such that ::

    A[i1, ... , id] = B[i1, ... , id, 0] + i * B[i1, ... , id, 1]

  This is compatible with a simple reinterpration of data in memory in C++ (reinterpret_cast).  

* An attribute :

   * named "__complex__"
   * of type "fixed size string", C-style (i.e. H5T_C_S1).
   * of value "1"

  is added to the array. This allows Python to make the difference between a `d` dimensional array of complex
  and a `d+1` dimensional array of real numbers when reading the array.

* This is compatible with the ALPS 2.x storage.

* This is **not** compatible with the regular h5py storage for complex. 
  
  In TRIQS, in Python the recommendation is to use our little HDFArchive interface, which makes the convertion automatically.


Green functions
^^^^^^^^^^^^^^^^

Domains...

Meshes...


Now the GF.


Imaginary time Green function 
.................................

**TRIQS_HDF5_data_scheme**  = "GfImTime"

**Objects contents** : 

+--------------------+--------------------+---------------------------------------------------------------------------+
| Name               | Type               | Meaning                                                                   |
+====================+====================+===========================================================================+
| data               | 3d array of double | data[n,:,:] is the value of the Green function at the mesh point number n |
+--------------------+--------------------+---------------------------------------------------------------------------+
| mesh               | ImaginaryTimeMesh  | mesh                                                                      |
+--------------------+--------------------+---------------------------------------------------------------------------+
| singularity        | HighFrequencyTail  | The high frequency tail of the Green function.                            |
+--------------------+--------------------+---------------------------------------------------------------------------+
| Optional           |                    |                                                                           |
+--------------------+--------------------+---------------------------------------------------------------------------+
| indices/left,right | 1d array of string | The indices ?DOC?                                                         |
+--------------------+--------------------+---------------------------------------------------------------------------+

  
  
ALL OTHER GF here after.




