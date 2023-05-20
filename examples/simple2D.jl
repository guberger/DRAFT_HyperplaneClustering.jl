module ExampleSimple2D

include("../src/HyperplaneClustering.jl")
HC = HyperplaneClustering
Datum = HC.Datum
evalf = HC.evalf

using JuMP
using Gurobi
using Plots

const GUROBI_ENV = Gurobi.Env()
solver() = Model(optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>false
))

np1 = 20
np2 = 20
xs = Vector{Float64}[]
for (x1, x2) in Iterators.product(range(0, 1, length=np1),
                                  range(0, 1, length=np2))
    push!(xs, [x1, x2])
end
data = Datum[]
for x in xs
    σ = 1
    if x[2] < 0.5
        if x[1] + x[2] < 0.66
            σ = 2
        else
            σ = 3
        end
    end
    push!(data, Datum(x, σ))
end

plt = plot()

for σ = 1:3
    xs = [datum.x for datum in filter(datum -> datum.σ == σ, data)]
    scatter!(getindex.(xs, 1), getindex.(xs, 2))
end

ϵ = 1e-4
afs, flag = HC.learn_hyperplanes(data, 2, 2, ϵ, solver)

@assert flag

x = 0:0.01:1
y = 0:0.01:1
for af in afs
    f(x, y) = evalf(af, (x, y))
    contour!(x, y, f, levels=[0], lw=2.0)
end

display(plt)

end # module