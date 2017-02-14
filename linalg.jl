function decompose_udv{T<:Number}(A::Matrix{T})
  svdfact(A)
end

function decompose_udv!{T<:Number}(A::Matrix{T})
  svdfact!(A)
end

function expm_diag!{T<:Number}(A::Matrix{T})
  F = eigfact!(A)
  return F[:vectors] * spdiagm(exp(F[:values])) * ctranspose(F[:vectors])
end
