function test_boson_action_diff(p,l)
  old_hsfield = copy(p.hsfield)
  new_hsfield = copy(old_hsfield)

  site = rand(1:l.sites)
  slice = rand(1:p.slices)
  new_op = rand(3)
  # new_op = [100.,0.2,1234.56]
  new_hsfield[:,site,slice] = new_op[:]

  Sbef = boson_action(p,l,old_hsfield)
  Saft = boson_action(p,l,new_hsfield)
  dS_direct = Saft - Sbef
  dS = boson_action_diff(p,l,site,new_op,old_hsfield,slice)
  if dS==dS_direct
    error("Inconsistency between boson_action and boson_action_diff!")
  end
  return true
end
