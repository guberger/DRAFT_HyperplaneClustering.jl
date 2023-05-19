using LinearAlgebra
using JuMP
using CSDP
using Test
@static if isdefined(Main, :TestLocal) && TestLocal
    include("../src/HyperplaneClustering.jl")
else
    using HyperplaneClustering
end
HC = HyperplaneClustering

solver() = Model(optimizer_with_attributes(
    CSDP.Optimizer, "printlevel"=>0
))

N = 2
rmax = 100.0

## Empty
xs_neg = Vector{Float64}[]
xs_pos = Vector{Float64}[]
af, r = HC.separating_hyperplane(xs_neg, xs_pos, N, rmax, solver)

@testset "empty" begin
    @test r ≈ rmax
end

## Set 2D #1
xs_neg = [[1.0, 1.0]]
xs_pos = [[0.0, 0.0]]
af, r = HC.separating_hyperplane(xs_neg, xs_pos, N, rmax, solver)

@testset "diagonal" begin
    @test r ≈ sqrt(0.5)
    @test norm(af.a - [-1, -1]/sqrt(2)) < 1e-8
    @test abs(af.β - sqrt(0.5)) < 1e-8
end

xs_neg = [[0.0, 0.0]]
xs_pos = Vector{Float64}[]
af, r = HC.separating_hyperplane(xs_neg, xs_pos, N, rmax, solver)

@testset "half empty" begin
    @test r ≈ rmax
end

## Set 2D #2 feasible
xs_neg = [[0.0, 0.0], [4.0, 0.0]]
xs_pos = [[8.0, 0.0]]
af, r = HC.separating_hyperplane(xs_neg, xs_pos, N, rmax, solver)

@testset "horizontal feasible" begin
    @test r ≈ 2
    @test norm(af.a - [1, 0]) < 1e-8
    @test abs(af.β + 6) < 1e-8
end

## Set 2D #2 infeasible
xs_neg = [[0.0, 0.0], [4.0, 0.0]]
xs_pos = [[2.0, 0.0]]
af, r = HC.separating_hyperplane(xs_neg, xs_pos, N, rmax, solver)

@testset "horizontal infeasible" begin
    @test abs(r) < 1e-8
    @test norm(af.a) < 1e-8
    @test abs(af.β) < 1e-8
end

## Simplex
function test_simplex(N)
    xs_neg = [zeros(N)]
    xs_pos = []
    for i = 1:N
        push!(xs_neg, [j == i ? 1.0 : 0.0 for j = 1:N])
        push!(xs_pos, [j == i ? 2.0 : 0.0 for j = 1:N])
    end
    af, r = HC.separating_hyperplane(xs_neg, xs_pos, N, rmax, solver)

    @testset "simplex feasible $(N)" begin
        @test r ≈ 0.5/sqrt(N)
        @test norm(af.a - ones(N)/sqrt(N)) < 1e-8
        @test abs(af.β + 1.5/sqrt(N)) < 1e-8
    end

    push!(xs_pos, ones(N)/(2*N))
    af, r = HC.separating_hyperplane(xs_neg, xs_pos, N, rmax, solver)

    @testset "simplex infeasible $(N)" begin
        @test abs(r) < 1e-8
        @test norm(af.a) < 1e-8
        @test abs(af.β) < 1e-8
    end
end

for N = 1:10
    test_simplex(N)
end

nothing