using LightXML

if length(ARGS) < 1 error("L argument missing") end

L=parse(Int, ARGS[1])
N=L^2

# create an empty XML document
xdoc = XMLDocument()

# create & attach a graph (root) node
graph = create_root(xdoc, "GRAPH")

function addvertex!(node, id, typ, coordinates)
    v = new_child(node, "VERTEX")
    set_attributes(v, Dict("id"=>id, "type"=>typ))

    vc = new_child(v, "COORDINATE")
    add_text(vc, "$(join(coordinates .- 1, " "))")
end

function addedge!(node, src, trg, id, typ, vector)
    edge = new_child(node, "EDGE")
    set_attributes(edge, Dict("id"=>id))
    set_attributes(edge, Dict("source"=>src, "target"=>trg))
    set_attributes(edge, Dict("type"=>typ, "vector"=>join(vector, " ")))
end

# nn
sql = reshape(1:N, L, L)
up = circshift(sql,(-1,0))
right = circshift(sql,(0,-1))
down = circshift(sql,(1,0))
left = circshift(sql,(0,1))
neighbors = vcat(up[:]',right[:]',down[:]',left[:]') # colidx = site, rowidx = up right down left

# next nn (xy, yx)
ur = circshift(sql,(-1,-1))
dr = circshift(sql,(1,-1))
dl = circshift(sql,(1,1))
ul = circshift(sql,(-1,1))
Nneighbors = vcat(ur[:]',dr[:]',dl[:]',ul[:]') # colidx = site, rowidx = ur dr dl ul

# next-next nn (xx, yy)
uu = circshift(sql,(-2,0))
rr = circshift(sql,(0,-2))
dd = circshift(sql,(2,0))
ll = circshift(sql,(0,2))
NNneighbors = vcat(uu[:]',rr[:]',dd[:]',ll[:]') # colidx = site, rowidx = uu rr dd ll


# vertices
vi = 1 # vertex counter
for x in 1:L
    for y in 1:L
        global vi
        addvertex!(graph, vi, 0, [x, y])
        vi += 1
    end
end

# edges
ei = 1 # edge counter
for site in 1:N
    global ei

    # nn
    up = neighbors[1,site]
    right = neighbors[2,site]
    addedge!(graph, site, up, ei, 0, [0, 1])
    ei += 1
    addedge!(graph, site, right, ei, 0, [1, 0])
    ei +=1

    # next nn
    ur = Nneighbors[1,site]
    dr = Nneighbors[2,site]
    addedge!(graph, site, ur, ei, 1, [1, 1])
    ei += 1
    addedge!(graph, site, dr, ei, 1, [1, -1])
    ei +=1

    # next-next nn
    uu = NNneighbors[1,site]
    rr = NNneighbors[2,site]
    addedge!(graph, site, uu, ei, 2, [0, 2])
    ei += 1
    addedge!(graph, site, rr, ei, 2, [2, 0])
    ei +=1

end

set_attributes(graph, Dict("dimension"=>2, "vertices"=>vi-1, "edges"=>ei-1))

# s = string(xdoc);
# print(xdoc)

save_file(xdoc, "NNsquare_L_$(L)_W_$(L).xml")
