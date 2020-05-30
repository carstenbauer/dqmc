"""
    sparsity(A)

Calculates the sparsity of the given array.
The sparsity is the number of zero-valued elements over the
total number of elements.
"""
sparsity(A::AbstractArray{<:Number}) = count(iszero, A)/length(A)

"""
    reldiff(A, B)

Relative difference of absolute values of `A` and `B` defined as

``
\\operatorname{reldiff} = 2 \\dfrac{\\operatorname{abs}(A - B)}{\\operatorname{abs}(A+B)}.
``
"""
function reldiff(A::AbstractArray{T}, B::AbstractArray{S}) where T<:Number where S<:Number
  return 2*abs.(A-B)./abs.(A+B)
end


"""
    effreldiff(A, B, threshold=1e-14)

Same as `reldiff(A,B)` but with all elements set to zero where corresponding element of
`absdiff(A,B)` is smaller than `threshold`. This is useful in avoiding artificially large
relative errors.
"""
function effreldiff(A::AbstractArray{T}, B::AbstractArray{S}, threshold::Float64=1e-14) where T<:Number where S<:Number
  r = reldiff(A,B)
  r[findall(x->abs.(x)<threshold,absdiff(A,B))] .= 0.
  return r
end


"""
    absdiff(A, B)

Difference of absolute values of `A` and `B`.
"""
function absdiff(A::AbstractArray{T}, B::AbstractArray{S}) where T<:Number where S<:Number
  return abs.(A-B)
end


"""
    compare(A, B)

Compares two matrices `A` and `B`, prints out the maximal absolute and relative differences
and returns a boolean indicating wether `isapprox(A,B)`.
"""
function compare(A::AbstractArray{T}, B::AbstractArray{S}) where T<:Number where S<:Number
  @printf("max absdiff: %.1e\n", maximum(absdiff(A,B)))
  @printf("mean absdiff: %.1e\n", mean(absdiff(A,B)))
  @printf("max reldiff: %.1e\n", maximum(reldiff(A,B)))
  @printf("mean reldiff: %.1e\n", mean(reldiff(A,B)))

  r = effreldiff(A,B)
  @printf("effective max reldiff: %.1e\n", maximum(r))
  @printf("effective mean reldiff: %.1e\n", mean(r))

  return isapprox(A,B)
end


"""
    compare_full(A, B)

Compares two matrices `A` and `B`, prints out all absolute and relative differences
and returns a boolean indicating wether `isapprox(A,B)`.
"""
function compare_full(A::AbstractArray{T}, B::AbstractArray{S}) where T<:Number where S<:Number
  compare(A,B)
  println("")
  println("absdiff: ")
  display(absdiff(A,B))
  println("")
  println("reldiff: ")
  display(reldiff(A,B))
  println("")
  return isapprox(A,B)
end


"""
    setrng(rng)
Replaces current `Random.default_rng()` with `rng`.
"""
function setrng(rng::MersenneTwister)
  Random.default_rng().idxI = rng.idxI
  Random.default_rng().idxF = rng.idxF
  Random.default_rng().state = rng.state
  Random.default_rng().vals = rng.vals
  Random.default_rng().seed = rng.seed
  Random.default_rng().ints = rng.ints
  nothing
end

"""
  saverng(filename [, rng::MersenneTwister; group="GLOBAL_RNG"])
  saverng(HDF5.HDF5File [, rng::MersenneTwister; group="GLOBAL_RNG"])
Saves the current state of Julia's random generator (`Random.default_rng()`) to HDF5.
"""
function saverng(f::HDF5.HDF5File, rng::MersenneTwister=Random.default_rng(); group::String="GLOBAL_RNG")
  g = endswith(group, "/") ? group : group * "/"
  try
    if HDF5.exists(f, g)
      HDF5.o_delete(f, g)
    end

    f[g*"idxF"] = rng.idxF
    f[g*"idxI"] = rng.idxI
    f[g*"state_val"] = rng.state.val
    f[g*"vals"] = rng.vals
    f[g*"seed"] = rng.seed
    f[g*"ints"] = Int.(rng.ints)
  catch e
    error("Error while saving RNG state: ", e)
  end
  nothing
end
function saverng(filename::String, rng::MersenneTwister=Random.default_rng(); group::String="GLOBAL_RNG")
  mode = isfile(filename) ? "r+" : "w"
  HDF5.h5open(filename, mode) do f
    saverng(f, rng; group=group)
  end
end

"""
  loadrng(filename [; group="GLOBAL_RNG"]) -> MersenneTwister
  loadrng(f::HDF5.HDF5File [; group="GLOBAL_RNG"]) -> MersenneTwister
Loads a random generator from HDF5.
"""
function loadrng(f::HDF5.HDF5File; group::String="GLOBAL_RNG")::MersenneTwister
  rng = MersenneTwister(0)
  g = endswith(group, "/") ? group : group * "/"
  try
    rng.idxI = read(f[g*"idxI"])
    rng.idxF = read(f[g*"idxF"])
    rng.state = Random.DSFMT.DSFMT_state(read(f[g*"state_val"]))
    rng.vals = read(f[g*"vals"])
    rng.seed = read(f[g*"seed"])
    rng.ints = UInt128.(read(f[g*"ints"]))
  catch e
    error("Error while restoring RNG state: ", e)
  end
  return rng
end
function loadrng(filename::String; group::String="GLOBAL_RNG")
  HDF5.h5open(filename, "r") do f
    loadrng(f; group=group)
  end
end


"""
  restorerng(filename [; group="GLOBAL_RNG"]) -> Void
  restorerng(f::HDF5.HDF5File [; group="GLOBAL_RNG"]) -> Void
Restores a state of Julia's random generator (`Random.GLOBAL_RNG`) from HDF5.
"""
function restorerng(filename::String; group::String="GLOBAL_RNG")
  HDF5.h5open(filename, "r") do f
    restorerng(f; group=group)
  end
  nothing
end
restorerng(f::HDF5.HDF5File; group::String="GLOBAL_RNG") = setrng(loadrng(f; group=group))


"""
    h5repack(src, trg)
Repacks a HDF5 file e.g. to free unused space. If `src == trg`Wrapper to external h5repack
application.
"""
function h5repack(src::String, trg::String)
    if src == trg   h5repack(src)  end
    @static if Sys.iswindows()
        read(`h5repack.exe $src $trg`, String)
    end
    @static if Sys.islinux()
        read(`h5repack $src $trg`, String)
    end
end
function h5repack(filename::String)
    h5repack(filename, "tmp.h5")
    mv("tmp.h5",filename,force=true)
end


##############################################################
# delete this block when https://github.com/JuliaMath/AbstractFFTs.jl/pull/26 is merged
struct Frequencies <: AbstractVector{Float64}
    nreal::Int
    n::Int
    multiplier::Float64
end

unsafe_getindex(x::Frequencies, i::Int) =
    (i-1+ifelse(i <= x.nreal, 0, -x.n))*x.multiplier
function Base.getindex(x::Frequencies, i::Int)
    (i >= 1 && i <= x.n) || throw(BoundsError())
    unsafe_getindex(x, i)
end
if isdefined(Base, :iterate)
    function Base.iterate(x::Frequencies, i::Int=1)
        i > x.n ? nothing : (unsafe_getindex(x,i), i + 1)
    end
else
    Base.start(x::Frequencies) = 1
    Base.next(x::Frequencies, i::Int) = (unsafe_getindex(x, i), i+1)
    Base.done(x::Frequencies, i::Int) = i > x.n
end
Base.size(x::Frequencies) = (x.n,)
Base.step(x::Frequencies) = x.multiplier

"""
    fftfreq(n, fs=1)
Return discrete fourier transform sample frequencies. The returned
Frequencies object is an AbstractVector containing the frequency
bin centers at every sample point. `fs` is the sample rate of the
input signal.
"""
fftfreq(n::Int, fs::Real=1) = Frequencies(((n-1) >> 1)+1, n, fs/n)

"""
    rfftfreq(n, fs=1)
Return discrete fourier transform sample frequencies for use with
`rfft`. The returned Frequencies object is an AbstractVector
containing the frequency bin centers at every sample point. `fs`
is the sample rate of the input signal.
"""
rfftfreq(n::Int, fs::Real=1) = Frequencies((n >> 1)+1, (n >> 1)+1, fs/n)
FFTW.fftshift(x::Frequencies) = (x.nreal-x.n:x.nreal-1)*x.multiplier
##############################################################
