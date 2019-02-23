using LightXML
using PyPlot

lattice_file = ARGS[1]

lfile = parse_file(lattice_file)
graph = LightXML.root(lfile)
N = parse(Int, attribute(graph, "vertices"; required=true))
L = Int(sqrt(N))

figure()

# # vertices
# xmlvertices = get_elements_by_tagname(graph, "VERTEX")
# x = Array{Float64}(length(xmlvertices))
# y = Array{Float64}(length(xmlvertices))
# for (i,v) in enumerate(xmlvertices)
#   cords = map(x->parse(Float64,x), split(content(find_element(v, "COORDINATE")), " "))
#   x[i] = cords[1]
#   y[i] = cords[2]
# end
# scatter(x,y, c=1:length(x), cmap="Blues")

# edges
xmledges = get_elements_by_tagname(graph, "EDGE")
x = Array{Float64}(length(xmledges))
y = Array{Float64}(length(xmledges))
sql = reshape(1:N, L, L)
switch = true
types = Int[]
labels = String[]

for (i,edge) in enumerate(xmledges)
    attrs = attributes_dict(edge)
    src = parse(Int, attrs["source"])
    trg = parse(Int, attrs["target"])
    typ = parse(Int, attrs["type"])
    if !(typ in types) push!(types, typ); push!(labels, "Type $(length(types))"); end
    # if typ != 2
    #     continue
    # end

    v1 = [reverse(ind2sub(sql,src))...]
    v2 = [reverse(ind2sub(sql,trg))...]

    if src > trg # pbc bond
      continue # leave out pbs bonds
      if (v1[2] == L && v1[1] != L) || (v1[2] == L && switch)
        v2 = copy(v1)
        v2[2] += 1
        if v1 == [L, L]
          switch = false
        end
      elseif (v1[2] != L && v1[1] == L) || (v1[2] == L && !switch)
        v2 = copy(v1)
        v2[1] += 1
      end
    end
    plot(zip(v1-1,v2-1)..., "-", color="C$typ")
    # plot(zip(v1-1,v2-1)..., "-")
    plot(zip(v1-1,v2-1)..., "o", color="black")
end

#Create custom artists
handles = [plt[:Line2D]((0,1),(0,0), color="C$(i-1)") for i in 1:length(types)]

#Create legend from custom artist/label lists
legend(handles,labels,loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=4, fancybox=true)
xlabel("x")
ylabel("y")

axis("equal")
savefig("$(lattice_file)_edges.pdf")

close("all")
