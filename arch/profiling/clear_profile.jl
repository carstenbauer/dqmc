for f in filter(x->endswith(x, ".mem"), readdir())
    rm(f)
end

rm("profile.bin")
rm("profile.svg")

cd("..")

for f in filter(x->endswith(x, ".mem"), readdir())
    rm(f)
end
