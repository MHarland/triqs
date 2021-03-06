SET(PythonBuildExecutable ${CMAKE_BINARY_DIR}/build_pytriqs)

# runs a c++ test
# if there is a .ref file a comparison test is done
# Example: add_cpp_test(my_code)
#   where my_code is the cpp executable my_code.ref is the expected output
macro(add_cpp_test testname)
 enable_testing()

 if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${testname}.ref)

  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/${testname}.ref DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

  add_test(${testname}
   ${CMAKE_COMMAND}
   -Dname=${testname}${ARGN}
   -Dcmd=${CMAKE_CURRENT_BINARY_DIR}/${testname}${ARGN}
   -Dreference=${CMAKE_CURRENT_SOURCE_DIR}/${testname}.ref
   -P ${TRIQS_SOURCE_DIR}/cmake/run_test.cmake
  )

 else (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${testname}.ref)

  add_test(${testname}${ARGN} ${testname}${ARGN})

 endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${testname}.ref)

endmacro(add_cpp_test)
 
# runs a python test
# if there is a .ref file a comparison test is done
# Example: add_python_test(my_script)
#   where my_script.py is the script and my_script.ref is the expected output
macro(add_python_test testname)
 enable_testing()

 if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${testname}.ref)

  file( COPY ${CMAKE_CURRENT_SOURCE_DIR}/${testname}.ref DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

  add_test(${testname}
   ${CMAKE_COMMAND}
   -Dname=${testname}
   -Dcmd=${PythonBuildExecutable}
   -Dinput=${CMAKE_CURRENT_SOURCE_DIR}/${testname}.py
   -Dreference=${CMAKE_CURRENT_SOURCE_DIR}/${testname}.ref
   -P ${TRIQS_SOURCE_DIR}/cmake/run_test.cmake
  )

 else (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${testname}.ref)

  add_test(${testname}
   ${CMAKE_COMMAND}
   -Dname=${testname}
   -Dcmd=${PythonBuildExecutable}
   -Dinput=${CMAKE_CURRENT_SOURCE_DIR}/${testname}.py
   -P ${TRIQS_SOURCE_DIR}/cmake/run_test.cmake
  )

 endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${testname}.ref)

endmacro(add_python_test)
