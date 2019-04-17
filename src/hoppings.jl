function init_hopping_matrices(mc::AbstractDQMC)
  p = mc.p
  l = mc.l

  if p.Bfield
    init_peirls_phases(mc)
    init_hopping_matrix_exp_Bfield(mc)
    p.chkr && init_checkerboard_matrices_Bfield(mc)
  else
    init_hopping_matrix_exp(mc)
    p.chkr && init_checkerboard_matrices(mc)
  end
  nothing
end

function init_hopping_matrix_exp(mc::AbstractDQMC)
  p = mc.p
  l = mc.l

  println("Initializing hopping exponentials")
  !p.Bfield || @warn("You should be using `init_hopping_matrix_exp_Bfield()` or set p.Bfield = false!")

  Tx = diagm(0 => fill(-p.mu1,l.sites))
  Ty = diagm(0 => fill(-p.mu2,l.sites))
  hor_nb = [2,4]
  ver_nb = [1,3]

  # Nearest neighbor hoppings
  if p.hoppings != "none"
    @inbounds @views begin
      for src in 1:l.sites
        for nb in hor_nb # horizontal neighbors
          trg = l.neighbors[nb,src]
          Tx[trg,src] += -l.t[1,1] # t_x_hor = tx1
          Ty[trg,src] += -l.t[1,2] # t_y_hor = tx2
        end

        for nb in ver_nb # vertical neighbors
          trg = l.neighbors[nb,src]
          Tx[trg,src] += -l.t[2,1] # t_x_ver = ty1
          Ty[trg,src] += -l.t[2,2] # t_y_ver = ty2
        end
      end
    end
  end

  # Next nearest neighbor hoppings (diagonal)
  if p.Nhoppings != "none"
    @inbounds @views begin
      for src in 1:l.sites
        for nb in 1:4
          trg = l.Nneighbors[nb,src]
          Tx[trg,src] += -l.tN[1,1] # t_x_diag = txy1
          Ty[trg,src] += -l.tN[1,2] # t_y_diag = txy2
        end
      end
    end
  end

  # Next-next nearest neighbor hoppings
  if p.NNhoppings != "none"
    @inbounds @views begin
      for src in 1:l.sites
        for nb in hor_nb # horizontal neighbors
          trg = l.NNneighbors[nb,src]
          Tx[trg,src] += -l.tNN[1,1] # t_x_hor = txx1
          Ty[trg,src] += -l.tNN[1,2] # t_y_hor = txx2
        end

        for nb in ver_nb # vertical neighbors
          trg = l.NNneighbors[nb,src]
          Tx[trg,src] += -l.tNN[2,1] # t_x_ver = tyy1
          Ty[trg,src] += -l.tNN[2,2] # t_y_ver = tyy2
        end
      end
    end
  end

  eTx_minus = exp(-0.5 * p.delta_tau * Tx)
  eTy_minus = exp(-0.5 * p.delta_tau * Ty)
  eTx_plus = exp(0.5 * p.delta_tau * Tx)
  eTy_plus = exp(0.5 * p.delta_tau * Ty)

  if p.opdim == 3
    l.hopping_matrix_exp = cat(eTx_minus,eTy_minus,eTx_minus,eTy_minus, dims=(1,2))
    l.hopping_matrix_exp_inv = cat(eTx_plus,eTy_plus,eTx_plus,eTy_plus, dims=(1,2))
  else # O(1) and O(2)
    l.hopping_matrix_exp = cat(eTx_minus,eTy_minus, dims=(1,2))
    l.hopping_matrix_exp_inv = cat(eTx_plus,eTy_plus, dims=(1,2))
  end

  return nothing
end

function init_peirls_phases(mc::AbstractDQMC)
  println("Initializing Peirls phases (Bfield)")
  l = mc.l
  L = mc.l.L
  p = mc.p

  B = zeros(2,2) # colidx = flavor, rowidx = spin up,down
  if p.Bfield
    B[1,1] = B[2,2] = 2 * pi / l.sites
    B[1,2] = B[2,1] = - 2 * pi / l.sites
  end

  l.peirls = Matrix{Matrix{Float64}}(undef, 2,2) # colidx = flavor, rowidx = spin up,down
  for f in 1:2 # flv
    for s in 1:2 # spin
      phis = fill(NaN, L, L, L, L)

      for x in 1:L
        for y in 1:L
          xp = mod1(x + 1, L)
          yp = mod1(y + 1, L)
          xm = mod1(x - 1, L)
          ym = mod1(y - 1, L)
          xp2 = mod1(x + 2, L)
          yp2 = mod1(y + 2, L)

          #nn
          if p.hoppings != "none"
            phis[x,y,x,yp] = 0
            phis[x,yp,x,y] = 0
            
            phis[x,y,xp,y] = - B[s,f] * (y - 1)
            phis[xp,y,x,y] = - phis[x,y,xp,y]
            if y == L
                phis[x,y,x,yp] = B[s,f] * L * (x -1)
                phis[x,yp,x,y] = - phis[x,y,x,yp]
            end
          end

          #nnn
          if p.Nhoppings != "none"
            phis[x,y,xp,yp] = -B[s,f] * (y -1 + 0.5)
            phis[xp,yp,x,y] = -phis[x,y,xp,yp]
            
            phis[x,y,xm,yp] = B[s,f] * (y -1 + 0.5)
            phis[xm,yp,x,y] = -phis[x,y,xm,yp]
            
            if y == L
                phis[x,y,xp,yp] = B[s,f] * (L * (x -1) + 0.5)
                phis[xp,yp,x,y] = - phis[x,y,xp,yp]
                
                phis[x,y,xm,yp] = B[s,f] * (L * (x-1) - 0.5)
                phis[xm,yp,x,y] = - phis[x,y,xm,yp]
            end
          end
              
          #nnnn
          if p.NNhoppings != "none"
            phis[x,y,x,yp2] = 0
            phis[x,yp2,x,y] = 0
            
            phis[x,y,xp2,y] = -2 * B[s,f] * (y-1)
            phis[xp2,y,x,y] = -phis[x,y,xp2,y]
            if (y == L) || (y == L - 1)
                phis[x,y,x,yp2] = B[s,f] * L * (x-1)
                phis[x,yp2,x,y] = - phis[x,y,x,yp2]
            end
          end

        end
      end
      l.peirls[s,f] = reshape(permutedims(phis, [2,1,4,3]), (l.sites,l.sites))

    end
  end
end

function init_hopping_matrix_exp_Bfield(mc::AbstractDQMC)
  p = mc.p
  l = mc.l
  H = heltype(mc)

  println("Initializing hopping exponentials (Bfield)")
  p.Bfield || @warn("You should be using `init_hopping_matrix_exp()` or set p.Bfield = true!")

  T = Matrix{Matrix{H}}(undef, 2,2) # colidx = flavor, rowidx = spin up,down

  hor_nb = [2,4]
  ver_nb = [1,3]

  for f in 1:2
    for s in 1:2
      T[s,f] = convert(Matrix{H}, diagm(0 => fill(f == 1 ? -p.mu1 : -p.mu2,l.sites)))

      # Nearest neighbor hoppings
      if p.hoppings != "none"
        @inbounds @views begin
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

      # Next nearest neighbor hoppings (diagonal)
      if p.Nhoppings != "none"
        @inbounds @views begin
          for src in 1:l.sites
            for nb in 1:4
              trg = l.Nneighbors[nb,src]
              T[s,f][trg,src] += - exp(im * l.peirls[s,f][trg,src]) * l.tN[1,f]
            end
          end
        end
      end

      # Next-next nearest neighbor hoppings
      if p.NNhoppings != "none"
        @inbounds @views begin
          for src in 1:l.sites
            for nb in hor_nb # horizontal neighbors
              trg = l.NNneighbors[nb,src]
              T[s,f][trg,src] += - exp(im * l.peirls[s,f][trg,src]) * l.tNN[1,f] # horizontal
            end

            for nb in ver_nb # vertical neighbors
              trg = l.NNneighbors[nb,src]
              T[s,f][trg,src] += - exp(im * l.peirls[s,f][trg,src]) * l.tNN[2,f] # vertical
            end
          end
        end
      end

    end
  end

  eT_minus = map(Ti -> exp(-0.5 * p.delta_tau * Ti), T)
  eT_plus = map(Ti -> exp(0.5 * p.delta_tau * Ti), T)

  if p.opdim == 3
    l.hopping_matrix_exp = cat(eT_minus[1,1], eT_minus[2,2], eT_minus[2,1], eT_minus[1,2], dims=(1,2))
    l.hopping_matrix_exp_inv = cat(eT_plus[1,1], eT_plus[2,2], eT_plus[2,1], eT_plus[1,2], dims=(1,2))
  else # O(1) and O(2)
    l.hopping_matrix_exp = cat(eT_minus[1,1], eT_minus[2,2], dims=(1,2))
    l.hopping_matrix_exp_inv = cat(eT_plus[1,1], eT_plus[2,2], dims=(1,2))
  end

  return nothing
end