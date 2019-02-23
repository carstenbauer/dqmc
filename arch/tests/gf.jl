# -------------------------------------------------------
#                     GF wrapping
# -------------------------------------------------------
function test_gf_wrapping(mc::AbstractDQMC, safe_mult::Int=mc.p.safe_mult)
  const p = mc.p
  const s = mc.s

  # slice = rand(3:p.slices-2)
  slice = 3
  println("Slice: ", slice)
  println("safe_mult: ", safe_mult)
  gf, = calc_greens_and_logdet(mc,slice,safe_mult)

  # wrapping down
  gfwrapped = wrap_greens(mc,gf,slice,-1)
  gfwrapped2 = wrap_greens(mc,gfwrapped,slice-1,-1)
  gfexact, = calc_greens_and_logdet(mc,slice - 1,safe_mult)
  gfexact2, = calc_greens_and_logdet(mc,slice - 2,safe_mult)

  println("Comparing wrapped down vs num exact")
  compare(gfwrapped,gfexact)
  println("")
  println("Comparing twice wrapped down vs num exact")
  compare(gfwrapped2,gfexact2)

  # wrapping up
  gfwrapped = wrap_greens(mc,gf,slice,1)
  gfwrapped2 = wrap_greens(mc,gfwrapped,slice+1,1)
  gfexact, = calc_greens_and_logdet(mc,slice + 1, safe_mult)
  gfexact2, = calc_greens_and_logdet(mc,slice + 2, safe_mult)

  println("")
  println("")
  println("Comparing wrapped up vs num exact")
  compare(gfwrapped,gfexact)
  println("")
  println("Comparing twice wrapped up vs num exact")
  compare(gfwrapped2,gfexact2)

  nothing
end # ~1e-14



# -------------------------------------------------------
#         Compare up stack with down stack
# -------------------------------------------------------
function loggreens(mc)
    gs = Vector{Matrix{geltype(mc)}}()
    initialize_stack(mc); build_stack(mc);
    for i in 1:p.slices
        propagate(mc)
        push!(gs, copy(mc.s.greens));
        # @show mc.s.current_slice
    end
    return gs
end

function loggreens_up(mc)
    gs = Vector{Matrix{geltype(mc)}}()
    initialize_stack(mc); build_stack(mc);
    while mc.s.current_slice!=1 || mc.s.direction == -1
        propagate(mc)
    end
    for i in 1:p.slices
        push!(gs, copy(mc.s.greens));
        # @show mc.s.current_slice
        propagate(mc)
    end
    return gs
end

function test_compare_up_down_stack(mc)
  gs = loggreens(mc);
  gsup = loggreens_up(mc);

  println("Comparing Green's functions in up stack and down stack:")
  maximum.(absdiff.(reverse(gsup),gs))
end # 1e-14



# -------------------------------------------------------
#                   safe_mult check
# -------------------------------------------------------
function Bchain(mc, safe_mult::Int)
  F = calc_Bchain(mc, 1, mc.p.slices, safe_mult);
  return F[1]*spdiagm(F[2])*F[3]
end
function Bchain_dagger(mc, safe_mult::Int)
  F = calc_Bchain_dagger(mc, 1, mc.p.slices, safe_mult);
  return F[1]*spdiagm(F[2])*F[3]
end

function test_Bchain_safe_mult(mc::AbstractDQMC)
  g = Bchain(mc, 1)
  gsm = Bchain(mc, 10)

  println("Chain (safe_mult 1 vs 10):")
  compare(g, gsm)

  g = Bchain_dagger(mc, 1)
  gsm = Bchain_dagger(mc, 10)

  println()
  println("Chain dagger (safe_mult 1 vs 10):")
  compare(g, gsm)
end # 1e-14

function test_greens_and_logdet_safe_mult(mc::AbstractDQMC, slice::Int=1)
  g, = calc_greens_and_logdet(mc, slice, 1)
  gsm, = calc_greens_and_logdet(mc, slice, 10)

  println("Greens (safe_mult 1 vs 10):")
  compare(g, gsm)
end # 1e-15