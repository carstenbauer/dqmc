@def stack_shortcuts begin
  N = l.sites
  flv = p.flv
  safe_mult = p.safe_mult
end


mutable struct Stack{G<:Number, CB<:Checkerboard} # G = GreensEltype

    eye_flv::Matrix{Float64} # UPGRADE: Replace by dynamically sized I everywhere
    eye_full::Matrix{Float64} # UPGRADE: Replace by dynamically sized I everywhere
    ones_vec::Vector{Float64}

    # interaction matrix / slice matrix
    C::Vector{G}
    S::Vector{G}
    R::Vector{G}
    eV::SparseMatrixCSC{G,Int64}
    Bl::Matrix{G}

    # calc detratio
    delta_i::Matrix{G}
    M::Matrix{G}
    eVop1::Matrix{G}
    eVop2::Matrix{G}
    eVop1eVop2::Matrix{G}
    Mtmp::Matrix{G}
    Mtmp2::Matrix{G}

    # global update
    gb_u_stack::Array{G, 3}
    gb_d_stack::Matrix{Float64}
    gb_t_stack::Array{G, 3}
    gb_greens::Matrix{G}
    gb_log_det::Float64
    gb_hsfield::Array{Float64, 3}

    # update greens
    A::Matrix{G}
    B::Matrix{G}
    AB::Matrix{G}

    Stack{G, CB}() where {G<:Number, CB<:Checkerboard} = new{G, CB}()
end


function Stack{G, CB}(p::Params, l::Lattice, g::GreensStack{G}) where {G<:Number, CB<:Checkerboard}
    s = new{G,CB}()

    @stack_shortcuts
    s.eye_flv = Matrix{Float64}(I, flv, flv)
    s.eye_full = Matrix{Float64}(I, flv*N,flv*N)
    s.ones_vec = ones(flv*N)

    allocate_interaction_matrix!(s,p,l,g)
    allocate_global_update!(s,p,l,g)
    allocate_calc_detratio!(s,p,l,g)
    allocate_update_greens!(s,p,l,g)

    # slice matrix
    CB === CBFalse && (s.Bl = zeros(G, flv*N, flv*N))
    s
end



function allocate_interaction_matrix!(s::Stack{G,CB}, p::Params, l::Lattice, g::GreensStack{G}) where {G,CB}
    @stack_shortcuts
    # interaction matrix
    s.C = zeros(G, N)
    s.S = zeros(G, N)
    s.R = zeros(G, N)
    s.eV = spzeros(G, flv*N, flv*N)
end

function allocate_global_update!(s::Stack{G,CB}, p::Params, l::Lattice, g::GreensStack{G}) where {G,CB}
    @stack_shortcuts
    # Global update backup
    s.gb_u_stack = zero(g.u_stack)
    s.gb_d_stack = zero(g.d_stack)
    s.gb_t_stack = zero(g.t_stack)
    s.gb_greens = zero(g.greens)
    s.gb_log_det = 0.
    s.gb_hsfield = zero(p.hsfield)
end


function allocate_calc_detratio!(s::Stack{G,CB}, p::Params, l::Lattice, g::GreensStack{G}) where {G,CB}
    @stack_shortcuts
    # calc_detratio
    s.M = zeros(G, flv, flv)
    s.Mtmp = s.eye_flv - g.greens[1:N:end,1:N:end]
    s.delta_i = zeros(G, size(s.eye_flv))
    s.Mtmp2 = zeros(G, size(s.eye_flv))
    s.eVop1eVop2 = zeros(G, size(s.eye_flv))
    s.eVop1 = zeros(G, flv, flv)
    s.eVop2 = zeros(G, flv, flv)
end



function allocate_update_greens!(s::Stack{G,CB}, p::Params, l::Lattice, g::GreensStack{G}) where {G,CB}
    @stack_shortcuts
    # update_greens
    g.A = g.greens[:,1:N:end]
    g.B = g.greens[1:N:end,:]
    g.AB = g.A * g.B
end
