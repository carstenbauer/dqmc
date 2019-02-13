using SparseArrays, LinearAlgebra

##### Fermion creation & annihilation operators

"""
Syntax: cup,cdn = fermiops(ns). Form spin 1/2 fermi operators of ns sites.
Details:
Define d_j: locally anticommuting fermi operators which commute between
different sites.
To enforce anticommutation between different sites, attach a 'string' by
defining c_k=Pi_{j<k} (-1)^(n_j) d_k.
"""
function fermiops(ns::Int)
    cups = sparse([0 0 0 0; 1 0 0 0; 0 0 0 0; 0 0 1 0]')
    cdns = sparse([0 0 0 0; 0 0 0 0; 1 0 0 0; 0 -1 0 0]')
    N = sparsevec([0, 1, 1, 2])
    signN = sparse(Diagonal([1, -1, -1, 1]))
    
    cup = SparseMatrixCSC{Float64,Int64}[]
    cdn = SparseMatrixCSC{Float64,Int64}[]
    id = sparse(1I, 4, 4)
    
    for k in 1:ns
        matup = spzeros(1,1)
        matup .= 1.
        matdn = spzeros(1,1)
        matdn .= 1.
        for j in 1:ns
            if j == k
                matup = kron(matup, cups)
                matdn = kron(matdn, cdns)
            elseif j < k
                matup = kron(matup, signN)
                matdn = kron(matdn, signN)
            elseif j > k
                matup = kron(matup, id)
                matdn = kron(matdn, id)
            end
        end
        push!(cup, matup)
        push!(cdn, matdn)
    end
    return cup, cdn
end

"""
Syntax: c = fermiops(ns). Form fermi operators of ns sites.
Details:
Define d_j: locally anticommuting fermi operators which commute between
different sites.
To enforce anticommutation between different sites, attach a 'string' by
defining c_k=Pi_{j<k} (-1)^(n_j) d_k.
"""
function fermiops_spinless(ns::Int)
    cs = sparse([0 0; 1 0]')
    N = sparsevec([0, 1])
    signN = sparse(Diagonal([1, -1])) # even / uneven # of fermion operators
    
    c = SparseMatrixCSC{Int64,Int64}[]
    id = sparse(1I, 2, 2)
    
    for k in 1:ns
        mat = 1
        for j in 1:ns
            if j == k
                mat = kron(mat, cs)
            elseif j < k
                mat = kron(mat, signN)
            elseif j > k
                mat = kron(mat, id)
            end
        end
        push!(c, mat)
    end
    return c
end


# Note: the resulting cs are not really fermion operators
# i.e. they do not fullfill the usual anticommutation relations!
function fermiops_spinless_spbasis(ns::Int)
    n = ns + 1 # the one is the all zeros state
    c = Array{SparseMatrixCSC{Int64,Int64}, 1}(undef, ns)
    
    for k in 1:ns
        c[k] = spzeros(Int, n,n)
        c[k][1,k+1] = 1
    end
    
    return c
end


anticomm(a,b) = Matrix(a*b + b*a)
comm(a,b) = Matrix(a*b - b*a)


function check_fermiops(c::AbstractVector{T}) where T<:AbstractArray
    n = size(c[1], 1)
    id = sparse(1I, n, n)
    for (j, c1) in enumerate(c)
        for (k, c2) in enumerate(c)
             # cc anticommutator should vanish
            cc = c1 * c2 + c2 * c1
            @assert sum(abs.(cc)) == 0
            
            # c^dagger c  anticommutator
            # should be identity if j==k and zero if j!=k
            cdc = c1' * c2 + c2 * c1'
            if j == k
                @assert sum(abs.(cdc .- id)) == 0
            else
                @assert sum(abs.(cdc)) == 0
            end
        end
    end
    return true
end

check_fermiops(c::AbstractArray{T}) where T = check_fermiops([c])



#### Hamiltonians

function generate_H_tb_spinless(t, mu, ns, c)
    L = Int(sqrt(ns))
    sqlattice = reshape(collect(1:ns), (L,L))
    # up and right neighbors
    unn = circshift(sqlattice, (1,0))[:]
    rnn = circshift(sqlattice, (0,-1))[:]
    dim = Int(2^ns)

    # chemical potential

    # initialize N:
    N = spzeros(Float64, dim, dim)

    for i in 1:ns
        N = N + c[i]' * c[i]
    end
    
    H = - mu * N
    
    @assert ishermitian(H)
    
    # hoppings

    for i in 1:ns
        H += - t * (c[i]' * c[unn[i]]) - t * (c[unn[i]]' * c[i])
        H += - t * (c[i]' * c[rnn[i]]) - t * (c[rnn[i]]' * c[i])
    end

    @assert ishermitian(H)
    return H
end


function generate_H_tb_spinless_spbasis(t, mu, ns, c)
    L = Int(sqrt(ns))
    sqlattice = reshape(collect(1:ns), (L,L))
    # up and right neighbors
    unn = circshift(sqlattice, (1,0))[:]
    rnn = circshift(sqlattice, (0,-1))[:]
    dim = Int(ns+1)

    # chemical potential

    # initialize N:
    N = spzeros(Float64, dim, dim)

    for i in 1:ns
        N = N + c[i]' * c[i]
    end
    
    H = - mu * N
    
    @assert ishermitian(H)
    
    # hoppings

    for i in 1:ns
        H += - t * (c[i]' * c[unn[i]]) - t * (c[unn[i]]' * c[i])
        H += - t * (c[i]' * c[rnn[i]]) - t * (c[rnn[i]]' * c[i])
    end

    @assert ishermitian(H)
    return H
end


function peirls_phase(i,j,L,B)
    @assert i in 1:L^2
    @assert j in 1:L^2
    
    sqlattice = reshape(collect(1:L^2), (L,L))
    i2, i1 = ind2sub(sqlattice, i)
    j2, j1 = ind2sub(sqlattice, j)
    
    if (i1 in 1:L-1 && j1 == i1+1) || (i1 == L && j1 == 1)
        return - 2*pi * B * (i2 - 1)
        
    elseif (i1 in 2:L && j1 == i1-1) || (i1 == 1 && j1 == L)
        return 2*pi * B * (i2 - 1)
        
    elseif i2 == L && j2 == 1
        return 2*pi * B * L * (i1 - 1)
        
    elseif i2 == 1 && j2 == L
        return - 2*pi * B * L * (i1 - 1)
        
    else
        return 0.
    end
end

function generate_H_tb_spinless_bfield_spbasis(t, mu, B, ns, c)
    L = Int(sqrt(ns))
    sqlattice = reshape(collect(1:ns), (L,L))
    # up and right neighbors
    unn = circshift(sqlattice, (1,0))[:]
    rnn = circshift(sqlattice, (0,-1))[:]
    dim = Int(ns+1)

    # chemical potential

    # initialize N:
    N = spzeros(Float64, dim, dim)

    for i in 1:ns
        N = N + c[i]' * c[i]
    end
    
    H = - mu * N
    
    @assert ishermitian(H)
    
    # hoppings

    for i in 1:ns
        Au = exp(im * peirls_phase(i, unn[i], L, B))
        Aut = exp(im * peirls_phase(unn[i], i, L, B))
        Ar = exp(im * peirls_phase(i, rnn[i], L, B))
        Art = exp(im * peirls_phase(rnn[i], i, L, B))
        
        H += - t * Au * (c[i]' * c[unn[i]]) - t * Aut * (c[unn[i]]' * c[i])
        H += - t * Ar * (c[i]' * c[rnn[i]]) - t * Art * (c[rnn[i]]' * c[i])
    end

    @assert ishermitian(H)
    return H
end

function spinspin(u, j, k, cup, cdn)
    """
    Spin-spin and density-density interaction. Returns
    sum_i u(i) (cdag(j) sigma(i) c(k)+ H.C)^2.
    Sigma are Pauli matrices (NOT spin matrices, i.e no factor of 1/2).
    u is a list of coefficients for (0, x, y, z) components of spin, where 0 is
    just the identity. j, k are site indices.
    """
    v = (cup[j]' * cup[k] + cdn[j]' * cdn[k] +
         cup[k]' * cup[j] + cdn[k]' * cdn[j])
    V = u[1] * v * v

    v = (cup[j]' * cdn[k] + cdn[j]' * cup[k] +
         cup[k]' * cdn[j] + cdn[k]' * cup[j])
    V = V + u[2] * v * v

    v = (cup[j]' * cdn[k] - cdn[j]' * cup[k] +
         cup[k]' * cdn[j] - cdn[k]' * cup[j])
    V = V - u[3] * v * v

    v = (cup[j]' * cup[k] - cdn[j]' * cdn[k] +
         cup[k]' * cup[j] - cdn[k]' * cdn[j])
    V = V + u[4] * v * v

    return V
end

function thop(t, j, k, cup, cdn)
    T = t * (cup[j]' * cup[k]) + t * (cup[k]' * cup[j])
    T = T + t * (cdn[j]' * cdn[k]) + t * (cdn[k]' * cdn[j])
    return T
end

function generate_H_SDW(params, ns, cup, cdn)
    txh = params["txh"]
    txv = params["txv"]
    tyh = params["tyh"]
    tyv = params["tyv"]
    mu = params["mu"]
    r = params["r"]
    lambdax = params["lambdax"]
    lambday = params["lambday"]
    lambdaz = params["lambdaz"]
    lambda0 = params["lambda0"]

    lambdas = Float64[lambda0, lambdax, lambday, lambdaz]
    u = Float64[-lamb^2 / (2 * r) for lamb in lambdas]

    if ns != 8
        throw("Number of sites != 8")
    end

    # chemical potential

    # initialize N:
    dim = Int(4^ns)
    N1 = spzeros(Float64, dim, dim)
    N2 = spzeros(Float64, dim, dim)

    for i in 1:4
        N1 = N1 + cup[i]' * cup[i]
        N1 = N1 + cdn[i]' * cdn[i]
    end
    for i in 5:8
        N2 = N2 + cup[i]' * cup[i]
        N2 = N2 + cdn[i]' * cdn[i]
    end
    
    H = - mu * N1 - mu * N2

    # Hopping part:
    H = H - thop(2 * txv, 1, 2, cup, cdn)
    H = H - thop(2 * txv, 3, 4, cup, cdn)
    H = H - thop(2 * txh, 1, 3, cup, cdn)
    H = H - thop(2 * txh, 2, 4, cup, cdn)
    H = H - thop(2 * tyv, 5, 6, cup, cdn)
    H = H - thop(2 * tyv, 7, 8, cup, cdn)
    H = H - thop(2 * tyh, 5, 7, cup, cdn)
    H = H - thop(2 * tyh, 6, 8, cup, cdn)

    # Interaction
    for j in 1:4
        H = H + spinspin(u, j, j + 4, cup, cdn)
    end

    @assert ishermitian(H)
    return H
end





function generate_H_SDW_spbasis(params, ns, c)
    th = params["th"]
    tv = params["tv"]
    mu = params["mu"]

    L = Int(sqrt(ns))
    sql = reshape(1:ns, (L,L))
    # up and right neighbors
    unn = circshift(sql, (-1,0))[:]
    rnn = circshift(sql, (0,-1))[:]
    dim = Int(ns+1)

    # initialize N:
    N = spzeros(Float64, dim, dim)

    for i in 1:ns
        N = N + c[i]' * c[i]
    end
    
    H = - mu * N
    
    @assert ishermitian(H)
    
    # hoppings

    for i in 1:ns
        H += - tv * (c[i]' * c[unn[i]]) - tv * (c[unn[i]]' * c[i])
        H += - th * (c[i]' * c[rnn[i]]) - th * (c[rnn[i]]' * c[i])
    end

    @assert ishermitian(H)
    return H
end



# This probably doesn't work that way.
# We probably have to adjust the phases because e.g 1->2 once exists on the lattice and once via PBC.
function peirls_SDW(i::Int , j::Int, B::Float64, sql::Matrix{Int}, pbc::Bool)::ComplexF64
    # peirls_phase_factors e^{im*Aij}

    i2, i1 = ind2sub(sql, i)
    j2, j1 = ind2sub(sql, j)
    L = size(sql, 1)

    if pbc
        if (i1 == L && j1 == 1)
            return exp(im*(- 2*pi * B * (i2 - 1)))
            
        elseif (i1 == 1 && j1 == L)
            return exp(im*(2*pi * B * (i2 - 1)))
            
        elseif i2 == L && j2 == 1
            return exp(im*(2*pi * B * L * (i1 - 1)))
            
        elseif i2 == 1 && j2 == L
            return exp(im*(- 2*pi * B * L * (i1 - 1)))
            
        else
            return exp(im*0.)
        end
    else
        if (i1 in 1:L-1 && j1 == i1+1)
            return exp(im*(- 2*pi * B * (i2 - 1)))
            
        elseif (i1 in 2:L && j1 == i1-1)
            return exp(im*(2*pi * B * (i2 - 1)))
            
        else
            return exp(im*0.)
        end
    end
end

function generate_H_SDW_Bfield(params, ns, cup, cdn)
    B = zeros(2,2) # colidx = flavor, rowidx = spin up,down
    B[1,1] = B[2,2] = 1 / (ns/2) # ns == number of sites *2 because of 2 flavors
    B[1,2] = B[2,1] = - 1 / (ns/2)

    txh = params["txh"]
    txv = params["txv"]
    tyh = params["tyh"]
    tyv = params["tyv"]
    mu = params["mu"]
    r = params["r"]
    lambdax = params["lambdax"]
    lambday = params["lambday"]
    lambdaz = params["lambdaz"]
    lambda0 = params["lambda0"]

    lambdas = Float64[lambda0, lambdax, lambday, lambdaz]
    u = Float64[-lamb^2 / (2 * r) for lamb in lambdas]

    if ns != 8
        throw("Number of sites != 8")
    end

    # chemical potential

    # initialize N:
    dim = Int(4^ns)
    N1 = spzeros(Float64, dim, dim)
    N2 = spzeros(Float64, dim, dim)

    for i in 1:4
        N1 = N1 + cup[i]' * cup[i]
        N1 = N1 + cdn[i]' * cdn[i]
    end
    for i in 5:8
        N2 = N2 + cup[i]' * cup[i]
        N2 = N2 + cdn[i]' * cdn[i]
    end
    
    H = - mu * N1 - mu * N2

    # for linidx to cartesianidx   
    sql = reshape(collect(1:Int(ns/2)), (2,2))


    cs = vcat(cup, cdn);
    t = [txh tyh; txv tyv]

    for f in 1:2 # flavor
        for s in 1:2 # spin
            c = cs[(s-1)*8+(f-1)*4+1:(s-1)*8+(f-1)*4+4]
            A = 
            # hopping within the lattice
            H = H - peirls_SDW(1, 2, B[s,f], sql, false) * t[2,f] * (c[1]' * c[2]) - peirls_SDW(2, 1, B[s,f], sql, false) * t[2,f] * (c[2]' * c[1])
            H = H - peirls_SDW(3, 4, B[s,f], sql, false) * t[2,f] * (c[3]' * c[4]) - peirls_SDW(4, 3, B[s,f], sql, false) * t[2,f] * (c[4]' * c[3])
            H = H - peirls_SDW(1, 3, B[s,f], sql, false) * t[1,f] * (c[1]' * c[3]) - peirls_SDW(3, 1, B[s,f], sql, false) * t[1,f] * (c[3]' * c[1])
            H = H - peirls_SDW(2, 4, B[s,f], sql, false) * t[1,f] * (c[2]' * c[4]) - peirls_SDW(4, 2, B[s,f], sql, false) * t[1,f] * (c[4]' * c[2])

            # hopping using PBC
            H = H - peirls_SDW(1, 2, B[s,f], sql, true) * t[2,f] * (c[1]' * c[2]) - peirls_SDW(2, 1, B[s,f], sql, true) * t[2,f] * (c[2]' * c[1])
            H = H - peirls_SDW(3, 4, B[s,f], sql, true) * t[2,f] * (c[3]' * c[4]) - peirls_SDW(4, 3, B[s,f], sql, true) * t[2,f] * (c[4]' * c[3])
            H = H - peirls_SDW(1, 3, B[s,f], sql, true) * t[1,f] * (c[1]' * c[3]) - peirls_SDW(3, 1, B[s,f], sql, true) * t[1,f] * (c[3]' * c[1])
            H = H - peirls_SDW(2, 4, B[s,f], sql, true) * t[1,f] * (c[2]' * c[4]) - peirls_SDW(4, 2, B[s,f], sql, true) * t[1,f] * (c[4]' * c[2])
        end
    end

    # Interaction
    for j in 1:4
        H = H + spinspin(u, j, j + 4, cup, cdn)
    end

    @assert ishermitian(H)
    return H
end


#### Quantities

# Density of states
function DOS(evals; δ=0.01, ωmin=minimum(evals), ωmax=maximum(evals), N=1000)
    ω = LinRange(ωmin, ωmax, N)
    kernel = 1 ./ (ω .- evals' .+ im*δ);
    ν = -1 ./ pi * imag(vec(sum(kernel, 2)))
    return ν, ω
end


function EV(O::SparseMatrixCSC, evecs, evals, beta::Float64)
    V = sparse(evecs)
    O_trans = V' * (O * V)
    E = evals .- evals[1]
    Z = sum(exp.(- beta * E))
    return real(1.0 / Z * sum(exp.(- beta * E) .* diag(O_trans)))
end

function GF(cs, evecs, evals, beta)
    # single-particle Green's function
    g = zeros(ComplexF64, length(cs), length(cs))
    for (j1, c1) in enumerate(cs)
        for (j2, c2) in enumerate(cs)
            # println(j1, " ", j2)
            g[j1,j2] = EV(c1 * c2', evecs, evals, beta)
        end
    end
    return g
end

"""
Calculates G(τ, 0)
"""
function TDGF(cs, tau, evecs, evals, beta)
    tdgf = zeros(length(cs), length(cs))
    for (j1, c1) in enumerate(cs)
        for (j2, c2) in enumerate(cs)
            tdgf[j1,j2] = corr(c1, c2', tau, evecs, evals, beta)
        end
    end
    return tdgf
end

# using PyCall
# @pyimport numpy as np
"""
    compute correlation function of operators
    chi = < A(tau) B(0) > = 1/Z sum_n_m exp(-(beta -tau) En) Anm exp(-tau Em) Bmn
"""
function corr(A, B, tau, evecs, evals, beta)
    @assert 0. <= tau <= beta
    # tau <= 1e-5 && warn("for short times corr is way off")

    chi = zero(eltype(A))
    V = sparse(evecs)

    A_trans = V' * A * V
    B_trans = V' * B * V

    E = evals .- evals[1]

    Z = sum(exp.(- beta * E))

    u1 = exp.(-(beta - tau) * E)
    u2 = exp.(- tau * E)

    # chi = np.einsum("n,nm,m,mn", u1, A_trans, u2, B_trans)/Z
    chi = einsum_explicit(u1, A_trans, u2, B_trans)/Z
    return chi
end

# explicit and faster version of np.einsum("n,nm,m,mn", u1, A_trans, u2, B_trans)
function einsum_explicit(u1,A_trans,u2,B_trans)
    s = 0.
    @inbounds for n in 1:length(u1)
        for m in 1:length(u2)
            s += u1[n] * A_trans[n,m] * u2[m] * B_trans[m,n]
        end
    end
    s
end



# ARCHIV

# Guess this is wrong because of o = c1_trans * c2dagger_trans
function TDGFelement(c1, c2, tau, evecs, evals, beta)
    E = evals - evals[1]
    V = sparse(evecs)
    c1_trans = V' * (c1 * V)
    c2dagger_trans = V' * (c2' * V)
    o = c1_trans * c2dagger_trans

    le = length(E)
    g = 0.0
    @inbounds for n in 1:le
        for m in 1:le
            g += exp(-beta*E[n] + tau*(E[n] - E[m])) * o[n,m]
        end
    end
    Z = sum(exp.(- beta * E))
    return g/Z
end
nothing