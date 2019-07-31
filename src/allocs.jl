mutable struct Allocs{G<:Number} # G = GreensEltype

    eye_flv::Matrix{Float64} # UPGRADE: Replace by dynamically sized I everywhere
    eye_full::Matrix{Float64} # UPGRADE: Replace by dynamically sized I everywhere
    ones_vec::Vector{Float64}

    function Allocs{G}() where G
        allocs = new{G}()
        allocs.eye_flv = Matrix{Float64}(I, flv, flv)
        allocs.eye_full = Matrix{Float64}(I, flv*N,flv*N)
        allocs.ones_vec = ones(flv*N)
        allocs
    end
end

