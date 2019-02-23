include("tests_gf_functions.jl")

"""
Free fermion system tests
"""
function test_interactions_are_zero(s::Stack,p::Params,l::Lattice)
    assert(p.lambda == 0)
    eV = interaction_matrix_exp(p,l,1)
    if eV != eye(eltype(eV),size(eV)...)
        error("Interaction matrix exponential not identity for free fermions!")
    end

    new_op = rand(3)
    V1i = interaction_matrix_exp_op(p,l,p.hsfield[:,rand(1:l.sites),rand(1:p.slices)],-1.)
    V2i = interaction_matrix_exp_op(p,l,new_op)
    delta_i = V1i * V2i  - eye_flv
    if delta_i != zeros(eltype(delta_i),size(delta_i)...)
        error("Delta_i not zero for free fermions!")
    end
    return true
end


function test_local_updates_dont_change_gf(s::Stack,p::Params,l::Lattice)
    assert(p.lambda == 0)
    g = copy(s.greens)
    local_updates(s,p,l)
    if !compare(g,s.greens)
        error("Local updates did change Green's function even though no coupling between systems!")
    end
    return true
end


function test_gf_is_correct(s::Stack,p::Params,l::Lattice)
    assert(p.lambda == 0)

    initialize_stack(s,p,l)
    build_stack(s,p,l)

    # reference greens function
    gchkr = calculate_greens_udv_chkr(p,l,1)
    g = calculate_greens_udv(p,l,1)

    # better not use down-sweep greens function because of inherent wrapping errors
    for k in 1:p.slices+1
        propagate(s,p,l)
    end

    assert(s.current_slice == 1)
    assert(s.direction == 1)

    if !compare(gchkr,s.greens)
        error("Green's function not correct!")
    end
    return true
end