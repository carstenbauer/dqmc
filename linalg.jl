# function combine_udt(Ul, Dl, Tl, Ur, Dr, Tr)
#   M = spdiagm(Dl) * (Tl * Ur) * spdiagm(Dr)
#   Up, Dp, Tp = decompose_udt(M)
#   return Ul * Up, Dp, Tp * Tr
# end
