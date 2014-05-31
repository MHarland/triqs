from wrap_generator import *

# The operator class
op = class_(
        py_type = "Operator",
        c_type = "many_body_operator<double>",
        is_printable= True,
        arithmetic = ("algebra","with_unit","with_unary_minus","double")
        )

op.add_constructor(signature="()", doc="create zero operator")

# The many_body_operators module
module = module_(full_name = "operators2", doc = "Doc to be written")
module.add_include("<triqs/operators/many_body_operator.hpp>")
module.add_include("<triqs/arrays.hpp>")
module.add_using("namespace triqs::utility")
module.add_class(op)

# Add various overload of c, c_dag to the module Annihilation & Creation operators
for name, doc in [ ("c","annihilation operator"), ("c_dag","creation operator")] :
    for sign in [
            "",
            "std::string ind1",
            "std::string ind1, std::string ind2",
            "int i, std::string ind1",
            "int i, int j"
            ]:
        module.add_function(name = name, signature="many_body_operator<double>(%s)"%sign, doc=doc)

module.generate_code()
