function calculate_boson_action(p::Parameters, l::Lattice, hsfield::Array{Float64,3}=p.hsfield)

  S = 0.0

  h = reshape(hsfield, (3,l.L,l.L,p.slices))

  # temporal gradient
  t = h - circshift(h, (0,0,0,-1))
  S += 0.5/p.delta_tau * 1./p.c^2 * dot(t,t);

  # spatial gradient
  # Count only top and right neighbor (avoid overcounting)
  @inbounds @simd for n in 2:3
    x = zeros(Int, 4)
    x[n] = -1
    t = h - circshift(h, x)
    S += p.delta_tau * 0.5 * dot(t,t);
  end

  # mass term & quartic interaction
  @inbounds @views begin
    @simd for s in 1:p.slices
      @simd for i in 1:l.sites

        squared = dot(hsfield[:,i,s],hsfield[:,i,s])
        S += p.delta_tau * p.r/2.0 * squared;
        S += p.delta_tau * p.u/4.0 * squared * squared;

      end
    end
  end

  return S;
end

"""
Calculate Delta_S_boson = S_boson' - S_boson
"""
function calculate_boson_action_diff(p::Parameters, l::Lattice, site::Int, slice::Int, new_op::Vector{Float64})
  
  dS = 0.0

  @inbounds @views begin
    old_op = p.hsfield[:,site,slice]
    diff = new_op - old_op

    old_op_sq = dot(old_op, old_op)
    new_op_sq = dot(new_op, new_op)
    sq_diff = new_op_sq - old_op_sq

    old_op_pow4 = old_op_sq * old_op_sq
    new_op_pow4 = new_op_sq * new_op_sq
    pow4_diff = new_op_pow4 - old_op_pow4

    op_earlier = p.hsfield[:,site,l.time_neighbors[2,slice]]
    op_later = p.hsfield[:,site,l.time_neighbors[1,slice]]
    op_time_neighbors = op_later + op_earlier

    op_space_neighbors = zeros(3)
    @simd for n in 1:4
      op_space_neighbors += p.hsfield[:,l.neighbors[n,site],slice]
    end

    dS += 1.0/(p.delta_tau * p.c^2)  * (sq_diff - dot(op_time_neighbors, diff));

    dS += 0.5 * p.delta_tau * (4 * sq_diff - 2.0 * dot(op_space_neighbors, diff));

    dS += p.delta_tau * (0.5 * p.r * sq_diff + 0.25 * p.u * pow4_diff);
    # dS += p.delta_tau * (0.5 * p.r * sq_diff);
  end

  return dS
end