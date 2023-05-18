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
xsneg = Vector{Float64}[]
xspos = Vector{Float64}[]
af, r = HC.separating_hyperplane(xsneg, xspos, N, rmax, solver)

@testset "empty" begin
    @test r ≈ rmax
end

## Set 2D #1
xsneg = [[1.0, 1.0]]
xspos = [[0.0, 0.0]]
af, r = HC.separating_hyperplane(xsneg, xspos, N, rmax, solver)

@testset "diagonal" begin
    @test r ≈ sqrt(0.5)
    @test norm(af.a - [-1, -1]/sqrt(2)) < 1e-8
    @test abs(af.β - sqrt(0.5)) < 1e-8
end

xsneg = [[0.0, 0.0]]
xspos = Vector{Float64}[]
af, r = HC.separating_hyperplane(xsneg, xspos, N, rmax, solver)

@testset "half empty" begin
    @test r ≈ rmax
end

## Set 2D #2 feasible
xsneg = [[0.0, 0.0], [4.0, 0.0]]
xspos = [[8.0, 0.0]]
af, r = HC.separating_hyperplane(xsneg, xspos, N, rmax, solver)

@testset "horizontal feasible" begin
    @test r ≈ 2
    @test norm(af.a - [1, 0]) < 1e-8
    @test abs(af.β + 6) < 1e-8
end

## Set 2D #2 infeasible
xsneg = [[0.0, 0.0], [4.0, 0.0]]
xspos = [[2.0, 0.0]]
af, r = HC.separating_hyperplane(xsneg, xspos, N, rmax, solver)

@testset "horizontal infeasible" begin
    @test abs(r) < 1e-8
    @test norm(af.a) < 1e-8
    @test abs(af.β) < 1e-8
end

## Simplex
function test_simplex(N)
    xsneg = [zeros(N)]
    xspos = []
    for i = 1:N
        push!(xsneg, [j == i ? 1.0 : 0.0 for j = 1:N])
        push!(xspos, [j == i ? 2.0 : 0.0 for j = 1:N])
    end
    af, r = HC.separating_hyperplane(xsneg, xspos, N, rmax, solver)

    @testset "simplex feasible $(N)" begin
        @test r ≈ 0.5/sqrt(N)
        @test norm(af.a - ones(N)/sqrt(N)) < 1e-8
        @test abs(af.β + 1.5/sqrt(N)) < 1e-8
    end

    push!(xspos, ones(N)/(2*N))
    af, r = HC.separating_hyperplane(xsneg, xspos, N, rmax, solver)

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