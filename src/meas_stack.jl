mutable struct MeasStack{G<:Number} # G = GreensEltype

  # ETPCs (equal-time pairing correlations)
  etpc_minus::Array{Float64, 2} # "P_(y,x)", i.e. d-wave
  etpc_plus::Array{Float64, 2} # "P+(y,x)", i.e. s-wave

  # ZFPCs (zero-frequency pairing correlations)
  zfpc_minus::Array{Float64, 2} # "P_(y,x)", i.e. d-wave
  zfpc_plus::Array{Float64, 2} # "P+(y,x)", i.e. s-wave


  # ETCDCs (equal-time charge density correlations)
  etcdc_minus::Array{ComplexF64, 2} # "C_(y,x)", i.e. d-wave
  etcdc_plus::Array{ComplexF64, 2} # "C+(y,x)", i.e. s-wave

  # ZFCDCs (zero-frequency charge density correlations)
  zfcdc_minus::Array{ComplexF64, 2} # "C_(y,x)", i.e. d-wave
  zfcdc_plus::Array{ComplexF64, 2} # "C+(y,x)", i.e. s-wave


  # ZFCCCs (zero-frequency current-current correlations)
  zfccc::Array{ComplexF64, 2} # "Λxx(y,x)"


  # TDGF
  Gt0::Vector{Matrix{G}}
  G0t::Vector{Matrix{G}}
  BT0Inv_u_stack::Vector{Matrix{G}}
  BT0Inv_d_stack::Vector{Vector{Float64}}
  BT0Inv_t_stack::Vector{Matrix{G}}
  BBetaT_u_stack::Vector{Matrix{G}}
  BBetaT_d_stack::Vector{Vector{Float64}}
  BBetaT_t_stack::Vector{Matrix{G}}
  BT0_u_stack::Vector{Matrix{G}}
  BT0_d_stack::Vector{Vector{Float64}}
  BT0_t_stack::Vector{Matrix{G}}
  BBetaTInv_u_stack::Vector{Matrix{G}}
  BBetaTInv_d_stack::Vector{Vector{Float64}}
  BBetaTInv_t_stack::Vector{Matrix{G}}

  MeasStack{G}() where G = new{G}()
end