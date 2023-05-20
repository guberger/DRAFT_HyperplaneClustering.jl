module MyTest

using Test
@static if isdefined(Main, :TestLocal) && Main.TestLocal
    include("../src/HyperplaneClustering.jl")
else
    using HyperplaneClustering
end
HC = HyperplaneClustering

x1s = [[0, 0], [0, 1], [0, 2], [0, 3],
       [1, 0], [1, 1], [1, 2], [1, 3],
       [2, 0], [2, 1], [2, 2], [2, 3],
       [3, 0], [3, 1], [3, 2], [3, 3]]
    
x2s = [[10, 10], [1.95, 2.36]]

dist, j1, j2 = HC.closest_distance(Vector{Float64}[], Vector{Float64}[])

@testset "both empty" begin
    @test isinf(dist)
    @test j1 == j2 == -1
end

dist, j1, j2 = HC.closest_distance(Vector{Float64}[], x2s)

@testset "x1s empty" begin
    @test isinf(dist)
    @test j1 == j2 == -1
end

dist, j1, j2 = HC.closest_distance(x1s, Vector{Float64}[])

@testset "x2s empty" begin
    @test isinf(dist)
    @test j1 == j2 == -1
end

dist, j1, j2 = HC.closest_distance(x1s, x2s)

@testset "both nonempty" begin
    @test dist ≈ sqrt(0.05^2 + 0.36^2)
    @test j1 == 11
    @test j2 == 2
end

dist, j1, j2 = HC.closest_distance(x2s, reverse(x2s))

@testset "same" begin
    @test abs(dist) < 1e-8
    @test j1 == 3 - j2
end

end # module