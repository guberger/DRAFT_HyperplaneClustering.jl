module MyTest

using Test
@static if isdefined(Main, :TestLocal) && Main.TestLocal
    include("../src/HyperplaneClustering.jl")
else
    using HyperplaneClustering
end
HC = HyperplaneClustering
AffForm = HC.AffForm
Datum = HC.Datum
signfs = HC.signfs

xs = [[0, 0], [0, 1], [0, 2], [0, 3],
      [1, 0], [1, 1], [1, 2], [1, 3],
      [2, 0], [2, 1], [2, 2], [2, 3],
      [3, 0], [3, 1], [3, 2], [3, 3]]
datavec = Datum[]
for x in xs
    σ::Int = x[2] > 1.5 ? 1 : x[1] < 1.5 ? 2 : 3
    push!(datavec, Datum(x, σ))
end

Q = 3
xlist = [Vector{Float64}[] for q = 1:Q]
afs = [AffForm([1, 0], -1.5), AffForm([1, 1], -3.0)]

dist, i1, i2 = HC.candidate_verification!(xlist, afs, datavec, xs, Q)

@testset "finite dist" begin
    @test dist ≈ 1
    @test datavec[i1].σ ≠ datavec[i2].σ
    @test signfs(afs, xs[i1]) == signfs(afs, xs[i2])
end

afs = [AffForm([1, 0], -1.5), AffForm([0, 2], -3.0)]

dist, i1, i2 = HC.candidate_verification!(xlist, afs, datavec, xs, Q)

@testset "infinite dist" begin
    @test isinf(dist)
    @test i1 == i2 == -1
end

afs = [AffForm([0, -1], 1.5), AffForm([-2, 0], 3.0)]

dist, i1, i2 = HC.candidate_verification!(xlist, afs, datavec, xs, Q)

@testset "infinite dist" begin
    @test isinf(dist)
    @test i1 == i2 == -1
end

end # module