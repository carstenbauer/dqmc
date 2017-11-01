## Add this to lattice.jl

  # Explicit multiplication of chkr_hop_4site (Max's version)
  chkr_hop_4site::Array{SparseMatrixCSC,3}
  chkr_hop_4site_half::Array{SparseMatrixCSC,3}
  chkr_hop_4site_inv::Array{SparseMatrixCSC,3}
  chkr_hop_4site_half_inv::Array{SparseMatrixCSC,3}


######

# and add  global const chkr_corners = [A_corners, B_corners] somewhere

# g: group (A, B), tflv: hopping flavor (tx, ty)
function multiply_chkr_hop_flv_left!(p::Parameters, l::Lattice, g::Int, tflv::Int, M::Matrix{Complex128})
  # For each corner
  for (cc, i) in enumerate(chkr_corners[g])
    j = l.neighbors[1,i]
    p = l.neighbors[2,i]
    q = l.neighbors[2,j]

    # Make copy of old rows
    ri = M[i, :]
    rj = M[j, :]
    rp = M[p, :]
    rq = M[q, :]

    @views c = l.chkr_hop_4site[cc, g, tflv] # OPT: view or not?

    # Calculate new rows
    @inbounds @views begin
      M[i,:] = c[i,i]*ri + c[i,j]*rj + c[i,p]*rp + c[i,q]*rq
      M[j,:] = c[j,i]*ri + c[j,j]*rj + c[j,p]*rp + c[j,q]*rq
      M[p,:] = c[p,i]*ri + c[p,j]*rj + c[p,p]*rp + c[p,q]*rq
      M[q,:] = c[q,i]*ri + c[q,j]*rj + c[q,p]*rp + c[q,q]*rq
    end
  end
end

# Above function replaces foldl. But how to apply it to full matrix i.e. with flavor and spin. (see cat below)? Max does it by explicitly performing the block matrix product