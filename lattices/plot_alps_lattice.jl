using LightXML
using PyPlot

lattice_file = ARGS[1]

lfile = parse_file(lattice_file)
graph = LightXML.root(lfile)

xmlvertices = get_elements_by_tagname(graph, "VERTEX")
x = Array{Float64}(length(xmlvertices))
y = Array{Float64}(length(xmlvertices))
for (i,v) in enumerate(xmlvertices)
  cords = map(x->parse(Float64,x), split(content(find_element(v, "COORDINATE")), " "))
  x[i] = cords[1]
  y[i] = cords[2]
end


scatter(x,y, c=1:length(x), cmap="Blues")
axis("equal")
savefig("$lattice_file.pdf")

close("all")
