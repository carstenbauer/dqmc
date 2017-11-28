if !isdefined(:HoppingType)
  global const HoppingType = Complex128; # assume worst case
  warn("HoppingType wasn't set on loading hoppings.jl")
  println("HoppingType = ", HoppingType)
end

function init_hopping_matrices(p::Parameters, l::Lattice)
  if p.Bfield
    init_peirls_phases(p,l)
    init_hopping_matrix_exp_Bfield(p,l)
    init_checkerboard_matrices_Bfield(p,l)
  else
    init_hopping_matrix_exp(p,l)
    init_checkerboard_matrices(p,l)
  end
  nothing
end

function init_hopping_matrix_exp(p::Parameters,l::Lattice)::Void
  println("Initializing hopping exponentials")
  !p.Bfield || warn("You should be using `init_hopping_matrix_exp_Bfield()` or set p.Bfield = false!")

  Tx = diagm(fill(-p.mu,l.sites))
  Ty = diagm(fill(-p.mu,l.sites))
  hor_nb = [2,4]
  ver_nb = [1,3]

  # Nearest neighbor hoppings
  @inbounds @views begin
    for src in 1:l.sites
      for nb in hor_nb # horizontal neighbors
        trg = l.neighbors[nb,src]
        Tx[trg,src] += -l.t[1,1] # t_x_hor
        Ty[trg,src] += -l.t[1,2] # t_y_hor
      end

      for nb in ver_nb # vertical neighbors
        trg = l.neighbors[nb,src]
        Tx[trg,src] += -l.t[2,1] # t_x_ver
        Ty[trg,src] += -l.t[2,2] # t_y_ver
      end
    end
  end

  eTx_minus = expm(-0.5 * p.delta_tau * Tx)
  eTy_minus = expm(-0.5 * p.delta_tau * Ty)
  eTx_plus = expm(0.5 * p.delta_tau * Tx)
  eTy_plus = expm(0.5 * p.delta_tau * Ty)

  if p.opdim == 3
    l.hopping_matrix_exp = cat([1,2],eTx_minus,eTy_minus,eTx_minus,eTy_minus)
    l.hopping_matrix_exp_inv = cat([1,2],eTx_plus,eTy_plus,eTx_plus,eTy_plus)
  else # O(1) and O(2)
    l.hopping_matrix_exp = cat([1,2],eTx_minus,eTy_minus)
    l.hopping_matrix_exp_inv = cat([1,2],eTx_plus,eTy_plus)
  end

  return nothing
end

function init_peirls_phases(p::Parameters, l::Lattice)
  println("Initializing Peirls phases (Bfield)")
  const L = l.L

  B = zeros(2,2) # colidx = flavor, rowidx = spin up,down
  if p.Bfield
    B[1,1] = B[2,2] = 2 * pi / l.sites
    B[1,2] = B[2,1] = - 2 * pi / l.sites
  end

  l.peirls = Matrix{Matrix{Float64}}(2,2) # colidx = flavor, rowidx = spin up,down
  for f in 1:2 # flv
    for s in 1:2 # spin
      phis = fill(NaN, L, L, L, L)

      for x in 1:L
        for y in 1:L
          xp = mod1(x + 1, L)
          yp = mod1(y + 1, L)
          
          #nn
          phis[x,y,x,yp] = 0
          phis[x,yp,x,y] = 0
          
          phis[x,y,xp,y] = - B[s,f] * (y - 1)
          phis[xp,y,x,y] = - phis[x,y,xp,y]
          if y == L
              phis[x,y,x,yp] = B[s,f] * L * (x -1)
              phis[x,yp,x,y] = - phis[x,y,x,yp]
          end
        end
      end
      l.peirls[s,f] = reshape(permutedims(phis, [2,1,4,3]), (l.sites,l.sites))

    end
  end
end

function init_hopping_matrix_exp_Bfield(p::Parameters,l::Lattice)::Void  
  println("Initializing hopping exponentials (Bfield)")
  p.Bfield || warn("You should be using `init_hopping_matrix_exp()` or set p.Bfield = true!")

  T = Matrix{Matrix{HoppingType}}(2,2) # colidx = flavor, rowidx = spin up,down
  for i in 1:4
    T[i] = convert(Matrix{HoppingType}, diagm(fill(-p.mu,l.sites)))
  end

  hor_nb = [2,4]
  ver_nb = [1,3]

  # Nearest neighbor hoppings
  @inbounds @views begin
    for f in 1:2
      for s in 1:2

        for src in 1:l.sites
          for nb in hor_nb # horizontal neighbors
            trg = l.neighbors[nb,src]
            T[s,f][trg,src] += - exp(im * l.peirls[s,f][trg,src]) * l.t[1,f] # horizontal
          end

          for nb in ver_nb # vertical neighbors
            trg = l.neighbors[nb,src]
            T[s,f][trg,src] += - exp(im * l.peirls[s,f][trg,src]) * l.t[2,f] # vertical
          end
        end

      end
    end
  end

  eT_minus = map(Ti -> expm(-0.5 * p.delta_tau * Ti), T)
  eT_plus = map(Ti -> expm(0.5 * p.delta_tau * Ti), T)

  if p.opdim == 3
    l.hopping_matrix_exp = cat([1,2], eT_minus[1,1], eT_minus[2,2], eT_minus[2,1], eT_minus[1,2])
    l.hopping_matrix_exp_inv = cat([1,2], eT_plus[1,1], eT_plus[2,2], eT_plus[2,1], eT_plus[1,2])
  else # O(1) and O(2)
    l.hopping_matrix_exp = cat([1,2], eT_minus[1,1], eT_minus[2,2])
    l.hopping_matrix_exp_inv = cat([1,2], eT_plus[1,1], eT_plus[2,2])
  end

  return nothing
end