include("dqmc.jl")

println("Loaded dqmc.jl")

using ProfileView

println("Loaded ProfileView")

Profile.clear()

println("Go!")

l = Lattice()
Lpos = maximum(search(p.lattice_file,"L_"))+1
l.L = parse(Int, p.lattice_file[Lpos:Lpos+minimum(search(p.lattice_file[Lpos:end],"_"))-2])
l.t = reshape([parse(Float64, f) for f in split(params["HOPPINGS"], ',')],(2,2))
init_lattice_from_filename(params["LATTICE_FILE"], l)
println("Initializing neighbor-tables")
init_neighbors_table(p,l)
init_time_neighbors_table(p,l)
println("Initializing hopping exponentials")
if !p.Bfield
  init_hopping_matrix_exp_Bfield(p,l)
else
  init_hopping_matrix_exp(p,l)
end
if p.chkr
  if p.Bfield
    init_checkerboard_matrices_Bfield(p,l)
  else
    init_checkerboard_matrices(p,l)
  end
end

s = Stack()
a = Analysis()

@profile MC_thermalize(s, p, l, a)

println("Done. Openening view.")

ProfileView.view()