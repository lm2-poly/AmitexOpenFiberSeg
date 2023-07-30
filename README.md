# AmitexProcessOnly_ComputeCanada

Use the data produced by OpenFiberSeg as input to AMITEX_FFTP. This code is released alongside publication in https://doi.org/10.1016/j.ijsolstr.2023.112421.

In file "materialProperties.py", provide the path to the "libUmatAmitex.so" library in your Amitex installation. 
The UMAT files provided need to be added to the <AMITEX PATH>/libAmitex/src/materiaux/, and compiled (using the provided Makefile).

