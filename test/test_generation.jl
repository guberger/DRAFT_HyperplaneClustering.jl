module MyTest

using JuMP
using CSDP
using Test
@static if isdefined(Main, :TestLocal) && Main.TestLocal
    include("../src/HyperplaneClustering.jl")
else
    using HyperplaneClustering
end
HC = HyperplaneClustering
Datum = HC.Datum
Node = HC.Node
nanafs = HC.nanafs

solver() = Model(optimizer_with_attributes(
    CSDP.Optimizer, "printlevel"=>0
))

xs = [[0, 0], [0, 1], [0, 2], [0, 3],
      [1, 0], [1, 1], [1, 2], [1, 3],
      [2, 0], [2, 1], [2, 2], [2, 3],
      [3, 0], [3, 1], [3, 2], [3, 3]]
datavec = Datum[]
for x in xs
    σ::Int = x[2] > 1.5 ? 1 : x[1] < 1.5 ? 2 : 3
    push!(datavec, Datum(x, σ))
end

N = 2
K = 2
afs = [nanafs(N) for k = 1:K]
xs_neg = Vector{Float64}[]
xs_pos = Vector{Float64}[]
node = Node([[1, 2, 13, 14], [9, 12, 13, 16]],
            [[3, 4, 15, 16], [1, 4, 5, 8]])
ϵ = 1e-3
rmax = 10

flag = HC.candidate_generation!(afs, xs_neg, xs_pos,
                                node, xs,
                                N, K, ϵ, rmax, solver)
                            
@testset "rectangle" begin
    @test flag
    @test afs[1].a ≈ [0, 1]
    @test afs[1].β ≈ -1.5
    @test afs[2].a ≈ [-1, 0]
    @test afs[2].β ≈ 1.5
end

afs = [nanafs(N) for k = 1:K]
node = Node([[1, 2, 13, 14], [9, 12, 13, 16]],
            [[3, 4, 15, 16], [10]])

flag = HC.candidate_generation!(afs, xs_neg, xs_pos,
                                node, xs,
                                N, K, ϵ, rmax, solver)
                            
@testset "infeasible" begin
    @test !flag
    @test afs[1].a ≈ [0, 1]
    @test afs[1].β ≈ -1.5
    @test all(isnan, afs[2].a)
    @test isnan(afs[2].β)
end

end # module