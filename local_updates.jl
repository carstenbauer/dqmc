function local_updates(mc::AbstractDQMC)
  p = mc.p
  s = mc.s
  l = mc.l
  a = mc.a

  acc_rat = 0.0
  @inbounds for i in 1:l.sites
    @views new_op = p.hsfield[:,i,s.current_slice] + rand(p.box, p.opdim)
    exp_delta_S_boson = exp(- calc_boson_action_diff(mc, i, s.current_slice, new_op) )
    detratio = calc_detratio(mc,i,new_op)

    if p.opdim == 3
      p_acc_fermion = real(detratio)
    elseif p.opdim < 3
      p_acc_fermion = real(detratio * conj(detratio))
    end

    if p.opdim == 3 && abs(imag(detratio)/real(detratio)) > 1e-4
      @printf("%d, %d \t Determinant ratio isn't real. \t abs imag: %.3e \t relative: %.1f%%\n", s.current_slice, i, abs.(imag(detratio)), abs.(imag(detratio))/abs.(real(detratio))*100)
    # elseif real(detratio) < 0
    #   println("Negative fermion weight.")
    # elseif detratio == 0
    #   println("Encountered non-invertible M with det = 0.")
    elseif p.opdim < 3 && p_acc_fermion < 0
      println("Negative fermion weight!")
    end

    p_acc = exp_delta_S_boson * p_acc_fermion

    if p_acc > 1.0 || rand() < p_acc
      acc_rat += 1
      @views p.hsfield[:,i,s.current_slice] = new_op
      p.boson_action += -log(exp_delta_S_boson)
      update_greens!(mc,i)
    end
  end
  return acc_rat / l.sites
end


@inline function calc_detratio(mc::AbstractDQMC, i::Int, new_op::Vector{Float64})
  p = mc.p
  s = mc.s
  l = mc.l

  @mytimeit mc.a.to "calc_detratio" begin

  interaction_matrix_exp_op!(mc,p.hsfield[:,i,s.current_slice],-1.,s.eVop1) #V1i
  interaction_matrix_exp_op!(mc,new_op,1.,s.eVop2) #V2i
  mul!(s.eVop1eVop2, s.eVop1, s.eVop2)
  s.delta_i .= s.eVop1eVop2 .- s.eye_flv
  s.Mtmp .= s.eye_flv .- s.greens[i:l.sites:end,i:l.sites:end]
  mul!(s.Mtmp2, s.delta_i, s.Mtmp)
  s.M .= s.eye_flv .+ s.Mtmp2

  end #timeit
  return det(s.M)
end

@inline function update_greens!(mc::AbstractDQMC, i::Int)
  p = mc.p
  s = mc.s
  l = mc.l
  g = mc.s.greens
  ab = mc.s.AB

  @mytimeit mc.a.to "update_greens!" begin

  s.A = s.greens[:,i:l.sites:end]
  
  if p.opdim == 3
    @simd for k in 0:3
        s.A[i+k*l.sites,k+1] -= 1.
    end
  else
    @simd for k in 0:1
        s.A[i+k*l.sites,k+1] -= 1.
    end
  end

  s.A *= inv(s.M)
  mul!(s.B, s.delta_i, s.greens[i:l.sites:end,:])

  # benchmark: most of time (>90% is spend below)
  mul!(s.AB, s.A, s.B)
  
  # more explicit way of doing s.greens .+= s.AB
  @inbounds @simd for i in eachindex(g)
    g[i] += ab[i]
  end

  end #timeit
  nothing
end



# function get_full_delta(mc,i)
#   delta_i = mc.s.delta_i
#   N = mc.l.sites
#   Delta = zeros(mc.s.greens)
#   Delta[i:N:end, i:N:end] = delta_i
#   return Delta
# end

# function cols2mat(mc,cols,i)
#   N = mc.l.sites
#   mat = zeros(mc.s.greens)
#   mat[:,i:N:end] .= cols
#   return mat
# end

# function shouldbe(mc,i)
#   Delta = get_full_delta(mc,i)
#   D = sparse(Delta)
#   A = (mc.s.eye_full - mc.s.greens)
#   return A*D #[:,i:mc.l.sites:end]
# end

# function shouldbe2(mc,i)
#   S = shouldbe(mc,i)
#   return inv(one(S) + S) #[:,i:mc.l.sites:end]
# end


# function update_greens_max!(mc::AbstractDQMC, i::Int, cols, colsTimesG)
#   G = geltype(mc)
#   N = mc.l.sites
#   flv = mc.p.flv
#   g = mc.s.greens
#   delta_i = mc.s.delta_i

#   site = i

#   # Calculate (1-G)*Delta
#   # The result is a (sparse) matrix with just a bunch of non-zero columns.
#   # Hence, we only calculate those columns.
#   # [O(N)]
#   # cols = zeros(G, flv*N, flv)
#   @inbounds for c in 1:flv
#     for r in 1:flv*N
#       cols[r,c] = - g[r,site] * delta_i[1, c]
#     end
#     cols[site, c] += delta_i[1,c]
#     for dr in 2:flv
#       for r in 1:flv*N # max had 4 instead of flv. mistake!?
#         cols[r,c] += - g[r, site + (dr-1)*N] * delta_i[dr,c]
#       end
#       cols[site + (dr-1)*N, c] += delta_i[dr,c]
#     end
#   end

#   # return cols2mat(mc,cols,site)
#   # return cols


#   # Now, calculate [1 + (1-G)*Delta]^(-1) based on (1-G)*Delta result above.
#   # The result is, again, a sparse matrix with only few (p.flv) non-zero columns.
#   # However, there are ones on the diagonal as well.
#   determinant = 1.0 + 0.0im
#   @inbounds for j in 1:flv
#     col = cols[:,j] #view?
#     for k in (j-1):-1:1
#       col[site + (k-1)*N] .= 0
#     end
#     for k in (j-1):-1:1
#       col .+= cols[site + (k-1)*N, j] * cols[:,k]
#     end

#     # @views begin
#       divisor = 1 + col[site + (j-1)*N]
#       cols[:,j] .= -1/divisor * col
#       cols[site + (j-1)*N, j] += 1
#       for k in (j-1):-1:1
#         cols[:,k] .-= (cols[site + (j-1)*N, k] / divisor) * col;
#       end
#       determinant *= divisor
#     # end
#   end
#   # @show determinant
#   # return cols

#   # Compensate for already included diagonal entries of I in cols
#   cols[site,1] -= 1
#   cols[site+N,2] -= 1
#   if p.opdim == 3
#     cols[site+2*N,3] -= 1
#     cols[site+3*N,4] -= 1
#   end

#   perform_update_greens!(mc, i, cols, colsTimesG)
#   nothing



#   # Based on the following by Max:
#   # //****
#   # //Compute the determinant and inverse of I + Delta*(I - G)
#   # //based on Sherman-Morrison formula / Matrix-Determinant lemma
#   # //****

#   # //Delta*(I - G) is a sparse matrix containing just 4 rows:
#   # //site, site+N, site+2N, site+3N
#   # //Compute the values of these rows [O(N)]:
#   # for (uint32_t r = 0; r < MatrixSizeFactor; ++r) {
#   #     //TODO: Here are some unnecessary operations:
#   #     //delta_forsite contains many repeated elements, and even
#   #     //some zeros
#   #     rows[r] = VecData(MatrixSizeFactor*pars.N);
#   #     for (uint32_t col = 0; col < MatrixSizeFactor*pars.N; ++col) {
#   #         rows[r][col] = -delta_forsite(r,0) * g.col(col)[site];
#   #     }
#   #     rows[r][site] += delta_forsite(r,0);
#   #     for (uint32_t dc = 1; dc < MatrixSizeFactor; ++dc) {
#   #         for (uint32_t col = 0; col < 4*pars.N; ++col) {
#   #             rows[r][col] += -delta_forsite(r,dc) * g.col(col)[site + dc*pars.N];
#   #         }
#   #         rows[r][site + dc*pars.N] += delta_forsite(r,dc);
#   #     }
#   # }

#   # // [I + Delta*(I - G)]^(-1) again is a sparse matrix
#   # // with two (four) rows site, site+N(, site+2N, site+3N).
#   # // compute them iteratively, together with the determinant of
#   # // I + Delta*(I - G)
#   # // Apart from these rows, the remaining diagonal entries of
#   # // [I + Delta*(I - G)]^(-1) are 1
#   # //
#   # // before this loop rows[] holds the entries of Delta*(I - G),
#   # // after the loop rows[] holds the corresponding rows of [I + Delta*(I - G)]^(-1)
#   # DataType det = 1;
#   # for (uint32_t l = 0; l < MatrixSizeFactor; ++l) {
#   #     VecData row = rows[l];
#   #     for (int k = int(l)-1; k >= 0; --k) {
#   #         row[site + unsigned(k)*pars.N] = 0;
#   #     }
#   #     for (int k = int(l)-1; k >= 0; --k) {
#   #         row += rows[l][site + unsigned(k)*pars.N] * rows[(unsigned)k];
#   #     }
#   #     DataType divisor = DataType(1) + row[site + l*pars.N];
#   #     rows[l] = (-1.0/divisor) * row;
#   #     rows[l][site + l*pars.N] += 1;
#   #     for (int k = int(l) - 1; k >= 0; --k) {
#   #         rows[(unsigned)k] -= (rows[unsigned(k)][site + l*pars.N] / divisor) * row;
#   #     }
#   #     det *= divisor;
#   # }

#   # //compensate for already included diagonal entries of I in invRows
#   # rows[0][site] -= 1;
#   # rows[1][site + pars.N] -= 1;
#   # if (OPDIM == 3) {
#   #     rows[2][site + 2*pars.N] -= 1;
#   #     rows[3][site + 3*pars.N] -= 1;
#   # }
#   # //compute G' = G * [I + Delta*(I - G)]^(-1) = G * [I + invRows]
#   # // [O(N^2)]
#   # MatData gTimesInvRows(MatrixSizeFactor*pars.N,
#   #                       MatrixSizeFactor*pars.N);
#   # auto& G = g;
#   # for (uint32_t col = 0; col < MatrixSizeFactor*pars.N; ++col) {
#   #     for (uint32_t row = 0; row < MatrixSizeFactor*pars.N; ++row) {
#   #         gTimesInvRows(row, col) =
#   #             G(row, site)            * rows[0][col]
#   #             + G(row, site + pars.N)   * rows[1][col];
#   #         if (OPDIM == 3) {
#   #             gTimesInvRows(row, col) +=
#   #                 G(row, site + 2*pars.N) * rows[2][col]
#   #                 + G(row, site + 3*pars.N) * rows[3][col];
#   #         }
#   #     }
#   # }
#   # g += gTimesInvRows

# end

# function perform_update_greens!(mc,i,cols,colsTimesG)
#   N = mc.l.sites
#   flv = mc.p.flv
#   g = mc.s.greens
#   site = i

#   # Compute G' = [I + (I - G)*Delta]^(-1) * G = [I + cols] * G
#   # [O(N^2)]
#   # colsTimesG = zeros(g)
#   @inbounds for c in 1:flv*N
#     @simd for r in 1:flv*N
#       colsTimesG[r,c] = cols[r,1] * g[site, c] + cols[r,2] * g[site + N, c]
#       if mc.p.opdim == 3
#         colsTimesG[r,c] += cols[r,3] * g[site + 2*N, c] + cols[r,4] * g[site + 3*N, c]
#       end
#     end
#   end
#   # TODO how to make this fast???


#   # g.+=colsTimesG;
#   @inbounds @simd for i in eachindex(g)
#     g[i] += colsTimesG[i]
#   end

#   # @show maximum(abs.(colsTimesG))
#   # return colsTimesG
#   nothing
# end