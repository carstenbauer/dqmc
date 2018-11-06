#### SVD, i.e. UDV decomposition
function decompose_udv!(A::Matrix{T}) where T<:Number
  LinearAlgebra.LAPACK.gesvd!('A','A',A) # was gesdd! at some point
end

function decompose_udv(A::Matrix{T}) where T<:Number
  X = copy(A)
  decompose_udv!(X)
end


#### QR, i.e. UDT decomposition
function decompose_udt(A::AbstractMatrix{C}) where C<:Number
  F = qr(A, Val(true))
  p = F.p
  @views p[p] = collect(1:length(p))
  # D = abs.(real(diag(triu(R))))
  D = abs.(real(diag(F.R)))
  T = (sparse(Diagonal(1 ./ D)) * F.R)[:, p]
  return F.Q, D, T
end

# function decompose_udt!(A::AbstractMatrix{C}, D) where C<:Number
#   Q, R, p = qr(A, Val{true}; thin=false)
#   @views p[p] = 1:length(p)
#   D .= abs.(real(diag(R)))
#   scale!(1 ./ D, R)
#   return Q, D, R[:, p]
# end

function decompose_udt!(A::AbstractMatrix{C}, D) where C<:Number
  F = qr!(A, Val(true))
  @views F.p[F.p] = 1:length(F.p)
  D .= abs.(real(diag(F.R)))
  R = F.R
  lmul!(Diagonal(1 ./ D), R)
  return Matrix(F.Q), R[:, F.p] # Q, (D is modified in-place), T  # was full(F.Q) before upgrade
end


#### Other
function expm_diag!(A::Matrix{T}) where T<:Number
  F = eigfact!(A)
  return F[:vectors] * spdiagm(exp(F[:values])) * adjoint(F[:vectors])
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
# UPGRADE: This is the relevant method in LinearAlgebra. How to rewrite it for Mat * SparseMat?
# import LinearAlgebra.mul!
# mul!(C::StridedVecOrMat, A::SparseMatrixCSC, B::StridedVecOrMat) =
#     mul!(C, A, B, one(eltype(B)), zero(eltype(C)))
# function mul!(C::StridedVecOrMat, A::SparseMatrixCSC, B::StridedVecOrMat, α::Number, β::Number)
#     A.n == size(B, 1) || throw(DimensionMismatch())
#     A.m == size(C, 1) || throw(DimensionMismatch())
#     size(B, 2) == size(C, 2) || throw(DimensionMismatch())
#     nzv = A.nzval
#     rv = A.rowval
#     if β != 1
#         β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
#     end
#     for k = 1:size(C, 2)
#         @inbounds for col = 1:A.n
#             αxj = α*B[col,k]
#             for j = A.colptr[col]:(A.colptr[col + 1] - 1)
#                 C[rv[j], k] += nzv[j]*αxj
#             end
#         end
#     end
#     C
# end
















# Calculates (UDVd)^-1, where U, D, Vd come from SVD decomp.
function inv_udv(U,D,Vd)
  m = adjoint(Vd)
  rmul!(m, Diagonal(1 ./ D))
  res = similar(m)
  mul!(res,m,adjoint(U))
  res
end

# Calculates (UDT)^-1, where U, D, T come from QR decomp.
function inv_udt(U,D,T)
  m = inv(T)
  res = similar(m)
  rmul!(m, Diagonal(1 ./ D))
  mul!(res, m, adjoint(U))
  res
end

# Calculates (1 + UDVd)^-1, where U, D, Vd come from SVD decomp.
# !! Breaks down for large spread in D (i.e. low temperatures).
function inv_one_plus_udv(U,D,Vd)
  inner = adjoint(Vd)
  inner .+= U * spdiagm(D)
  I = decompose_udv!(inner)
  u = adjoint(I[3] * Vd)
  d = 1 ./ I[2]
  vd = adjoint(I[1])

  rmul!(u,Diagonal(d))
  u*vd
end

# same as inv_one_plus_udt but separating both U AND Vd from D
# !! Breaks down for large spread in D (i.e. low temperatures). Slightly better than normal version.
function inv_one_plus_udv_alt(U,D,Vd)
  inner = adjoint(Vd*U)
  inner[diagind(inner)] .+= D
  u, d, vd = decompose_udv!(inner)

  t1 = adjoint(vd*Vd)
  t2 = adjoint(U*u)
  rmul!(t1, Diagonal(1 ./ d))
  t1*t2
end

# Calculates (1 + UDVd)^-1, where U, D, Vd come from SVD decomp.
# More controlled handling of scales, however also slower.
# This is Yoni's `stable_invert_B`
# Seems to work even for large spread in D (i.e. low temperatures)!
function inv_one_plus_udv_scalettar(U,D,Vd)
  Dp = max.(D,1.)
  Dm = min.(D,1.)
  Dpinv = 1 ./ Dp

  l = adjoint(Vd)
  rmul!(l, Diagonal(Dpinv))

  r = copy(U)
  rmul!(r, Diagonal(Dm))

  u, d, vd = decompose_udv!(l+r)

  m = inv_udv(u,d,vd)
  lmul!(Diagonal(Dpinv), m)
  u, d, vd = decompose_udv!(m)

  mul!(adjoint(m), Vd, u)
  # return m, d, vd
  rmul!(m, Diagonal(d))
  m*vd
end

# speed? Left in a hurry
function inv_one_plus_udt(U,D,T)
  m = adjoint(U) * inv(T)
  m[diagind(m)] .+= D
  u,d,t = decompose_udt(m)
  u = U*u
  t = t*T
  tinv = inv(t)
  rmul!(tinv, Diagonal(1 ./ d))
  tinv*adjoint(u)
end

function inv_one_plus_udt!(mc, res, U,D,T)
  m = mc.s.tmp
  d = mc.s.d
  u = mc.s.U
  t = mc.s.T

  mul!(m, adjoint(U), inv(T))
  m[diagind(m)] .+= D

  utmp,ttmp = decompose_udt!(m, d)
  mul!(u, U, utmp)
  mul!(t, ttmp, T)
  tinv = inv(t)
  rmul!(tinv, Diagonal(1 ./ d))
  mul!(res, tinv, adjoint(u))
  nothing
end

function UDV_to_mat!(mat, U, D, Vd, is_inv) 
    if !is_inv
        mat1 = copy(U)
        rmul!(mat1, Diagonal(D))
        mul!(mat,mat1,Vd)
    else #V D^(-1) Ud = (D^-1 *Vd)^(dagger) *Ud
        mat1 = copy(Vd)
        lmul!(Diagonal(1 ./ D), Vd)
        mul!(mat,adjoint(mat1),adjoint(U))
    end  
end

function UDT_to_mat!(mat, U, D, T; inv=false) 
    if !inv
        mat1 = copy(U)
        rmul!(mat1, Diagonal(D))
        mul!(mat,mat1,T)
    else # (DT)^-1 * U^dagger
        mat1 = copy(T)
        lmul!(Diagonal(D), mat1)
        mat .= mat1 \ adjoint(U)
    end  
end

# multiplies two UDVds -> UDVd
function mul_udvs(Ul,Dl,Vdl,Ur,Dr,Vdr)
  tmp = adjoint(Vdl)*Ur
  rmul!(tmp, Diagonal(Dr))
  lmul!(Diagonal(Dl), tmp)
  U, D, Vd = decompose_udv!(tmp)
  U = Ul*U
  Vd = Vd*Vdr
  return U, D, Vd
end

# Calculates (UaDaVda + UbDbVdb)^-1
function inv_sum_udvs(Ua, Da, Vda, Ub, Db, Vdb)
    
    d=length(Da)
    
    #DXp = max (X%D, 1) ,DXm = min (X%D, 1) and similarly for Y.
    Dap = max.(Da,1.)
    Dam = min.(Da,1.)
    Dbp = max.(Db,1.)
    Dbm = min.(Db,1.)

    #mat1= DXm * X%Vd * (Y%Vd)^dagger /DYp
    mat1 = Vda * adjoint(Vdb)
    for j in 1:d, k in 1:d
        mat1[j,k]=mat1[j,k] * Dam[j]/Dbp[k]
    end

    #mat2 = 1/(DXp) * (X%U)^dagger * Y%U * DYm
    mat2 = adjoint(Ua) * Ub
    for j in 1:d, k in 1:d
        mat2[j,k]=mat2[j,k] * Dbm[k]/Dap[j]
    end
    
    #mat1 = mat1+mat2
    mat1 = mat1 + mat2
    
    #invert mat1: mat1=mat1^(-1)
    U, D, Vd = decompose_udv!(mat1)
    UDV_to_mat!(mat1, U, D, Vd, true)

    #mat1 = 1/DYp * mat1 /DXp
    for j in 1:d, k in 1:d
        mat1[j,k]=mat1[j,k] / Dbp[j] / Dap[k]
    end

    #mat1 = U D Vd
    U, D, Vd = decompose_udv!(mat1)

    # U = (Y%Vd)^dagger * U , Vd = Vd * (X%U)^dagger
    mul!(mat1,adjoint(Vdb),U)
    mul!(mat2,Vd,adjoint(Ua))
    U=mat1
    Vd=mat2

    return U, D, Vd
end

# Calculates (UaDaTda + UbDbTdb)^-1
function inv_sum_udts(Ua,Da,Ta,Ub,Db,Tb)
  m1 = Ta * inv(Tb)
  lmul!(Diagonal(Da), m1)

  m2 = adjoint(Ua) * Ub
  rmul!(m2, Diagonal(Db))

  u,d,t = decompose_udt(m1 + m2)

  mul!(m1, Ua, u)
  mul!(m2, t, Tb)

  return inv(m2), 1 ./ d, adjoint(m1)
  # m3 = inv(m2)
  # scale!(m3, 1 ./ d)
  # return m3 * adjoint(m1)
end

function inv_sum_udts!(mc, res, Ua,Da,Ta,Ub,Db,Tb)
  d = mc.s.d
  m1 = mc.s.tmp
  m2 = mc.s.tmp2

  mul!(m1, Ta, inv(Tb))
  lmul!(Diagonal(Da), m1)

  mul!(m2, adjoint(Ua), Ub)
  rmul!(m2, Diagonal(Db))

  u,t = decompose_udt!(m1 + m2, d)

  mul!(m1, Ua, u)
  mul!(m2, t, Tb)

  m3 = inv(m2)
  rmul!(m3, Diagonal(1 ./ d))
  mul!(res, m3, adjoint(m1))

  nothing
end