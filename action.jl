function boson_action(p::parameters, l::lattice,
                        hsfield::Array{Float64,3}=p.hsfield)
  S = 0.0
  for s in 1:p.slices
    for i in 1:l.sites

      # temporal gradient
      diff = hsfield[:,i,s] - hsfield[:,i,time_neighbor(s,-1)]
      S += 0.5/p.delta_tau * dot(diff,diff);

      # spatial gradient
      # Count only top and right neighbor (avoid overcounting)
      for n in 1:2
        diff = hsfield[:,i,s] - hsfield[:,l.neighbors[n,i],s]
        S += p.delta_tau * 0.5 * dot(diff,diff);
      end

      # mass term & quartic interaction
      squared = dot(hsfield[:,i,s],hsfield[:,i,s])
      S += p.delta_tau * p.r/2.0 * squared;
      S += p.delta_tau * p.u/4.0 * squared * squared;

    end
  end
  return S;
end

"""
Calculate Delta_S_boson = S_boson' - S_boson
"""
function boson_action_diff(p::parameters, l::lattice, site::Int, new_op::Vector{Float64},
                            hsfield::Array{Float64, 3}=p.hsfield, slice::Int=s.current_slice)
  old_op = view(hsfield,:,site,slice)
  diff = new_op - old_op

  old_op_sq = dot(old_op, old_op)
  new_op_sq = dot(new_op, new_op)
  sq_diff = new_op_sq - old_op_sq

  old_op_pow4 = old_op_sq * old_op_sq
  new_op_pow4 = new_op_sq * new_op_sq
  pow4_diff = new_op_pow4 - old_op_pow4

  op_earlier = hsfield[:,site,time_neighbor(slice,-1)]
  op_later = hsfield[:,site,time_neighbor(slice,1)]
  op_time_neighbors = op_later + op_earlier

  op_space_neighbors = zeros(3)
  for n in 1:4
    op_space_neighbors += hsfield[:,l.neighbors[n,site],slice]
  end

  dS = 0.0

  dS += 1.0/p.delta_tau  * (sq_diff - dot(op_time_neighbors, diff));

  dS += 0.5 * p.delta_tau * (4 * sq_diff - 2.0 * dot(op_space_neighbors, diff));

  dS += p.delta_tau * (0.5 * p.r * sq_diff + 0.25 * p.u * pow4_diff);

  return dS
end
