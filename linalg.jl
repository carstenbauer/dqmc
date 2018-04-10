#### SVD, i.e. UDV decomposition
function decompose_udv!(A::Matrix{T}) where T<:Number

  Base.LinAlg.LAPACK.gesvd!('A','A',A)

  # F = svdfact!(A) # based on Base.LinAlg.LAPACK.gesdd!('A',A)
  # return F[:U], F[:S], F[:Vt]
end

function decompose_udv(A::Matrix{T}) where T<:Number
  X = copy(A)
  decompose_udv!(X)
  return X
end


#### QR, i.e. UDT decomposition
function decompose_udt(A::AbstractMatrix{C}) where C<:Number
  Q, R, p = qr(A, Val{true}; thin=false)
  @views p[p] = collect(1:length(p))
  # D = abs.(real(diag(triu(R))))
  D = abs.(real(diag(R)))
  T = (spdiagm(1./D) * R)[:, p]
  return Q, D, T
end


#### Other
function expm_diag!(A::Matrix{T}) where T<:Number
  F = eigfact!(A)
  return F[:vectors] * spdiagm(exp(F[:values])) * ctranspose(F[:vectors])
end

function lu_det(M)
    L, U, p = lu(M)
    return prod(diag(L)) * prod(diag(U))
end

function multiply_safely(Ul,Dl,Tl, Ur,Dr,Tr)
  mat = Tl * Ur
  mat = spdiagm(Dl)*mat*spdiagm(Dr)
  U, D, T = decompose_udt(mat)
  return Ul*U, D, T*Tr
end

# See https://discourse.julialang.org/t/asymmetric-speed-of-in-place-sparse-dense-matrix-product/10256/3
import Base.A_mul_B!
function Base.A_mul_B!(Y::StridedMatrix{TY}, X::StridedMatrix{TX}, A::SparseMatrixCSC{TvA,TiA}) where {TY,TX,TvA,TiA}
    mX, nX = size(X)
    nX == A.m || throw(DimensionMismatch())
    fill!(Y, 0) # can't assume that Y is initialized with zeros
    rowval = A.rowval
    nzval = A.nzval
    @inbounds for multivec_row=1:mX, col = 1:A.n, k=A.colptr[col]:(A.colptr[col+1]-1)
        Y[multivec_row, col] += X[multivec_row, rowval[k]] * nzval[k]
    end
    Y
end