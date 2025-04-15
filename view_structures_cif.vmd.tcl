
# Load all-atom structure and trajectory
mol new "1CRN.cif" type cif waitfor all


# Load coarse-grained structure and trajectory
mol new "1CRN_cg.cif" type cif waitfor all


# All-atom representation
mol modselect 0 0 all
mol modstyle 0 0 Lines
mol modcolor 0 0 ColorID 1

# CG representation
mol modselect 0 1 all
mol modstyle 0 1 VDW
mol modcolor 0 1 ColorID 2

scale to 1.0
