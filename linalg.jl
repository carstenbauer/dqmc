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

function expm_diag!{T<:Number}(A::Matrix{T})
  F = eigfact!(A)
  return F[:vectors] * spdiagm(exp(F[:values])) * ctranspose(F[:vectors])
end

# function combine_udt(Ul, Dl, Tl, Ur, Dr, Tr)
#   M = spdiagm(Dl) * (Tl * Ur) * spdiagm(Dr)
#   Up, Dp, Tp = decompose_udt(M)
#   return Ul * Up, Dp, Tp * Tr
# end
