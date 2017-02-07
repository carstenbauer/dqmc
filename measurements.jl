
function measure_op(s, p, l)
  mean_abs_op = mean(abs(p.hsfield))
  mean_op = vec(mean(p.hsfield,[2,3]))
  return (mean_abs_op, mean_op)
end
