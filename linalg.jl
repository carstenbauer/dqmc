function decompose_udv!{T<:Number}(A::Matrix{T})

  Base.LinAlg.LAPACK.gesvd!('A','A',A)

  # F = svdfact!(A) # based on Base.LinAlg.LAPACK.gesdd!('A',A)
  # return F[:U], F[:S], F[:Vt]
end

function decompose_udv{T<:Number}(A::Matrix{T})
  X = copy(A)
  return decompose_udv!(X)
end


function decompose_udt{X<:Number}(A::Matrix{X})
  Q, R, p = qr(A, Val{true}; thin=false)
  p_T = copy(p); p_T[p] = collect(1:length(p))
  D = abs(real(diag(triu(R))))
  T = (spdiagm(1./D) * R)[:, p_T]
  return Q, D, T
end


function expm_diag!{T<:Number}(A::Matrix{T})
  F = eigfact!(A)
  return F[:vectors] * spdiagm(exp(F[:values])) * ctranspose(F[:vectors])
end
