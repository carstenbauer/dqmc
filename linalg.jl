function decompose_udv!{T<:Number}(A::Matrix{T})

  Base.LinAlg.LAPACK.gesvd!('A','A',A)

  # F = svdfact!(A) # based on Base.LinAlg.LAPACK.gesdd!('A',A)
  # return F[:U], F[:S], F[:Vt]
end

function decompose_udv{T<:Number}(A::Matrix{T})
  X = copy(A)
  return decompose_udv!(X)
end

function expm_diag!{T<:Number}(A::Matrix{T})
  F = eigfact!(A)
  return F[:vectors] * spdiagm(exp(F[:values])) * ctranspose(F[:vectors])
end
