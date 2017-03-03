include("tests_gf_functions.jl")

"""
Boson action diff
"""
function test_calculate_boson_action_diff(p,l)
  old_hsfield = copy(p.hsfield)
  new_hsfield = copy(old_hsfield)

  site = rand(1:l.sites)
  slice = rand(1:p.slices)
  new_op = rand(3)
  # new_op = [100.,0.2,1234.56]
  new_hsfield[:,site,slice] = new_op[:]

  Sbef = calculate_boson_action(p,l,old_hsfield)
  Saft = calculate_boson_action(p,l,new_hsfield)
  dS_direct = Saft - Sbef
  dS = calculate_boson_action_diff(p,l,site,new_op,old_hsfield,slice)
  if dS==dS_direct
    error("Inconsistency between boson_action and boson_action_diff!")
  end
  return true
end
# test_calculate_boson_action_diff(p,l)
# Worked


"""
Green's function update after spatial lattice sweep
"""
function delta_naive(s::Stack, p::Parameters, l::Lattice, i::Int, new_op::Vector{Float64})
  V1 = interaction_matrix_exp(p,l,s.current_slice,-1.0)
  bkp = copy(p.hsfield[:,i,s.current_slice])
  p.hsfield[:,i,s.current_slice] = new_op[:]
  V2 = interaction_matrix_exp(p,l,s.current_slice)
  p.hsfield[:,i,s.current_slice] = bkp[:]
  return (V1 * V2 - eye(p.flv*l.sites,p.flv*l.sites))
end

function delta_i_naive(s::Stack, p::Parameters, l::Lattice, i::Int, new_op::Vector{Float64})
  return delta_naive(s,p,l,i,new_op)[i:l.sites:end,i:l.sites:end]
end

function test_delta_i_consistency(s,p,l)
  old_hsfield = copy(p.hsfield)

  site = rand(1:l.sites)
  slice = s.current_slice
  # new_op = [1.,2.,3.]
  new_op = rand(3)

  detratio = calculate_detratio(s,p,l,site,new_op)
  delta_n = delta_naive(s,p,l,site,new_op)
  delta_i_n = delta_i_naive(s,p,l,site,new_op)

  if !isapprox(delta_i_n,s.delta_i)
    error("Efficient calculation of delta_i inconsistent with naive calculation!")
  elseif delta_n[site:l.sites:end,site:l.sites:end] != s.delta_i
    error("Delta_i inconsistent with full Delta!")
  end
  return true
end
# test_delta_i_consistency(s,p,l)
# Worked

function test_local_update_gf(s::Stack, p::Parameters, l::Lattice)
  p.hsfield = rand(3,l.sites,p.slices)
  s.current_slice = rand(1:p.slices)

  gfbefore = calculate_greens_udv_chkr(p,l,s.current_slice)
  s.greens = copy(gfbefore)

  for site in 1:l.sites
    new_op = rand(p.box, 3)
    detratio = calculate_detratio(s,p,l,site,new_op)
    update_greens!(s,p,l,site)
    p.hsfield[:,site,s.current_slice] = new_op[:]
  end

  gfafter = calculate_greens_udv_chkr(p,l,s.current_slice)

  println("Num exact vs updated")
  compare(s.greens, gfafter)
  # println("")
  # println("Before vs After (num exact)")
  # compare_greens(gfbefore, gfafter)
  nothing
end
# What about max rel dev (order 1)? Max abs dev around 1e-3 to 1e-2



"""
Time slice wrapping
"""
function wrap_greens(gf::Array{Complex{Float64},2},slice::Int,direction::Int)
  if direction == -1
    A = slice_matrix_no_chkr(p, l, slice - 1, -1.)
    temp = A * gf
    A = slice_matrix_no_chkr(p, l, slice - 1, 1.)
    return temp * A
  else
    A = slice_matrix_no_chkr(p, l, slice, 1.)
    temp = A * gf
    A = slice_matrix_no_chkr(p, l, slice, -1.)
    return temp * A
  end
end

function wrap_greens2(gf::Array{Complex{Float64},2},slice::Int,direction::Int)
  if direction == -1
    B = slice_matrix_no_chkr(p, l, slice - 1)
    temp = inv(B) * gf
    return temp * B
  else
    B = slice_matrix_no_chkr(p, l, slice)
    temp = B * gf
    return temp * inv(B)
  end
end

function test_wrap_greens_vs_wrap_greens2(p,l)
  # slice = rand(1:p.slices)
  slice = 10
  gf = calculate_greens_udv(p,l,slice)

  # wrapping down
  gfwrapped1 = wrap_greens(gf,slice,-1)
  gfwrapped2 = wrap_greens2(gf,slice,-1)

  println("Comparing wrapped down vs num exact")
  compare_greens(gfwrapped1,gfwrapped2)
end
# Worked! Found mistake in slice_matrix_no_chkr in stack.jl


function test_gf_wrapping(p::Parameters, l::Lattice)
  # slice = rand(1:p.slices)
  slice = 10
  gf = calculate_greens_udv(p,l,slice)

  # wrapping down
  gfwrapped = wrap_greens(gf,slice,-1)
  gfwrapped2 = wrap_greens(gfwrapped,slice-1,-1)
  gfexact = calculate_greens_udv(p,l,slice - 1)
  gfexact2 = calculate_greens_udv(p,l,slice - 2)

  println("Comparing wrapped down vs num exact")
  compare_greens(gfwrapped,gfexact)
  println("")
  println("Comparing twice wrapped down vs num exact")
  compare_greens(gfwrapped2,gfexact2)

  # wrapping up
  gfwrapped = wrap_greens(gf,slice,1)
  gfwrapped2 = wrap_greens(gfwrapped,slice+1,1)
  gfexact = calculate_greens_udv(p,l,slice + 1)
  gfexact2 = calculate_greens_udv(p,l,slice + 2)

  println("")
  println("")
  println("Comparing wrapped up vs num exact")
  compare_greens(gfwrapped,gfexact)
  println("")
  println("Comparing twice wrapped up vs num exact")
  compare_greens(gfwrapped2,gfexact2)

  nothing
end