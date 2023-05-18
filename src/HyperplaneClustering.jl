module HyperplaneClustering

using LinearAlgebra
using DataStructures
using JuMP

struct Datum
    x::Vector{Float64}
    σ::Int
end

struct AffForm
    a::Vector{Float64}
    β::Float64
end

evalf(af::AffForm, x::AbstractVector) = dot(af.a, x) + β

function separating_hyperplane(xsneg, xspos, N, rmax, solver)
    model = solver()
    a = @variable(model, [1:N])
    β = @variable(model)
    r = @variable(model, upper_bound=rmax)
    @constraint(model, sum(x -> x^2, a) ≤ 1)
    for x in xsneg
        @constraint(model, dot(a, x) + β + r ≤ 0)
    end
    for x in xspos
        @constraint(model, dot(a, x) + β - r ≥ 0)
    end
    @objective(model, Max, r)
    optimize!(model)
    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT
    return AffForm(value.(a), value(β)), value(r)
end

function learn_hyperplanes(data, N, M, ϵ, solver)
    println("to do")    
end

end # module HyperplaneClustering
