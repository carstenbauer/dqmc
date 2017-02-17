include("tests_gf_functions.jl")


"""
Green's function update after spatial lattice sweep
"""
function test_local_update_gf(s::Stack, p::Parameters, l::Lattice)
  p.hsfield = rand(3,l.sites,p.slices)
  s.current_slice = rand(1:p.slices)

  gfbefore = calculate_greens_udv(p,l,s.current_slice)
  s.greens = copy(gfbefore)

  for site in 1:l.sites
    new_op = rand(p.box, 3)
    detratio = calculate_detratio(s,p,l,site,new_op)
    update_greens!(s,p,l,site)
    p.hsfield[:,site,s.current_slice] = new_op[:]
  end

  gfafter = calculate_greens_udv(p,l,s.current_slice)

  println("Num exact vs updated")
  compare_greens(s.greens, gfafter)
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
  gfwrapped2 = wrap_greens(gfwrapped,slice,-1)
  gfexact = calculate_greens_udv(p,l,slice - 1)
  gfexact2 = calculate_greens_udv(p,l,slice - 2)

  println("Comparing wrapped down vs num exact")
  compare_greens(gfwrapped,gfexact)
  println("")
  println("Comparing twice wrapped down vs num exact")
  compare_greens(gfwrapped2,gfexact2)

  # wrapping up
  gfwrapped = wrap_greens(gf,slice,1)
  gfwrapped2 = wrap_greens(gfwrapped,slice,1)
  gfexact = calculate_greens_udv(p,l,slice + 1)
  gfexact2 = calculate_greens_udv(p,l,slice + 2)

  println("")
  println("")
  println("Comparing wrapped down vs num exact")
  compare_greens(gfwrapped,gfexact)
  println("")
  println("Comparing twice wrapped down vs num exact")
  compare_greens(gfwrapped2,gfexact2)

  nothing
end
# Single wrapping: Fine. Twice wrapping: deviation to large?
