module HyperplaneClustering

using LinearAlgebra
using DataStructures
using JuMP

struct AffForm
    a::Vector{Float64}
    β::Float64
end

evalf(af::AffForm, x::AbstractVector) = dot(af.a, x) + β

function separating_hyperplane(xs_neg, xs_pos, N, rmax, solver)
    model = solver()
    a = @variable(model, [1:N])
    β = @variable(model)
    r = @variable(model, upper_bound=rmax)
    @constraint(model, sum(x -> x^2, a) ≤ 1)
    for x in xs_neg
        @constraint(model, dot(a, x) + β + r ≤ 0)
    end
    for x in xs_pos
        @constraint(model, dot(a, x) + β - r ≥ 0)
    end
    @objective(model, Max, r)
    optimize!(model)
    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT
    return AffForm(value.(a), value(β)), value(r)
end

struct Datum
    x::Vector{Float64}
    σ::Int
end

struct Node
    indsets_neg::Vector{Vector{Int}}
    indsets_pos::Vector{Vector{Int}}
end

function processdata(data)
    M = maximum(datum -> datum.σ, data, init=0)::Int
    datavec = [datum for datum in data]::Vector{Datum}
    xs = [datum.x for datum in datavec]
    classes = [Int[] for q = 1:M]
    for (i, datum) in enumerate(datavec)
        push!(classes[datum.σ], i)
    end
    return classes, xs
end

nanafs(N) = AffForm(fill(NaN, N), NaN)

function populatexs!(xs_dest, indset, xs_all)
    empty!(xs_dest)
    for i in indset
        push!(xs_dest, xs_all[i])
    end
    return nothing
end

function candidate_generation!(afs, xs_neg, xs_pos,
                               node, xs,
                               N, K, ϵ, rmax, solver)
    for k = 1:K
        populatexs!(xs_neg, node.indsets_neg[k], xs)
        populatexs!(xs_pos, node.indsets_pos[k], xs)
        af, r = separating_hyperplane(xs_neg, xs_pos, N, rmax, solver)
        r < ϵ && return false
        afs[k] = af
        return true
    end
end

oppositesign(v, w) = ((v > 1e-8) && (w < 1e-8)) || ((v < 1e-8) && (w > 1e-8))

function are_separate_vectors(afs, x1, x2)
    for af in afs
        oppositesign(evalf(af, x1), evalf(af, x2)) && return true
    end
    return false
end

function are_separate_sets(afs, x1s, x2s)
    for (j1, x1) in enumerate(x1s)
        for (j2, x2) in enumerate(x2s)
            !are_separate_vectors(afs, x1, x2) && return j1, j2
        end
    end
    return -1, -1
end

function candidate_verification!(x1s, x2s, afs, xs, classes)
    Q = length(classes)
    for q1 = 1:Q
        populatexs!(x1s, classes[q1], xs)
        for q2 = (q1 + 1):Q
            populatexs!(x2s, classes[q2], xs)
            j1, j2 = are_separate_sets(afs, x1s, x2s)
            if j1 > 0 && j2 > 0
                return classes[q1][j1], classes[q2][j2]
            end
        end
    end
    return -1, -1
end

function learn_hyperplanes(data, N, K, ϵ, solver)
    classes, xs = processdata(data)
    root = Node([Int[] for k = 1:K], [Int[] for k = 1:K])
    tree = [root]
    afs = [nanafs(N) for k = 1:K]
    xs_neg = Vector{Float64}[]
    xs_pos = Vector{Float64}[]
    x1s = Vector{Float64}[]
    x2s = Vector{Float64}[]
    rmax = 1.0

    while !isempty(tree)
        node = pop!(tree)

        # Candidate generation
        !candidate_generation!(afs, xs_neg, xs_pos,
                               node, xs,
                               N, Kd, ϵ, rmax, solver) && continue

        # Candidate verification
        i1, i2 = candidate_verification!(x1s, x2s, afs, xs, classes)
        if i1 < 0 || i2 < 0
            return afs, true
        end

        # Branching
        for k = 1:K
            if isempty(node.indsets_neg[k]) && isempty(node.indsets_pos[k])
                indsets_neg = copy(node.indsets_neg)
                indsets_neg[k] = union(indsets_neg[k], [i1])
                indsets_pos = copy(node.indsets_pos)
                indsets_pos[k] = union(indsets_pos[k], [i2])
                push!(tree, Node(indsets_neg, indsets_pos))
                break
            elseif i1 ∈ node.indsets_pos[k]
                indsets_neg = copy(node.indsets_neg)
                indsets_neg[k] = union(indsets_neg[k], [i2])
                push!(tree, Node(indsets_neg, node.indsets_pos))
            elseif i1 ∈ node.indsets_neg[k]
                indsets_pos = copy(node.indsets_pos)
                indsets_pos[k] = union(indsets_pos[k], [i2])
                push!(tree, Node(node.indsets_neg, indsets_pos))
            elseif i2 ∈ node.indsets_pos[k]
                indsets_neg = copy(node.indsets_neg)
                indsets_neg[k] = union(indsets_neg[k], [i1])
                push!(tree, Node(indsets_neg, node.indsets_pos))
            elseif i2 ∈ node.indsets_neg[k]
                indsets_pos = copy(node.indsets_pos)
                indsets_pos[k] = union(indsets_pos[k], [i1])
                push!(tree, Node(node.indsets_neg, indsets_pos))
            else
                indsets_neg = copy(node.indsets_neg)
                indsets_neg[k] = union(indsets_neg[k], [i1])
                indsets_pos = copy(node.indsets_pos)
                indsets_pos[k] = union(indsets_pos[k], [i2])
                push!(tree, Node(indsets_neg, indsets_pos))
                indsets_neg = copy(node.indsets_neg)
                indsets_neg[k] = union(indsets_neg[k], [i2])
                indsets_pos = copy(node.indsets_pos)
                indsets_pos[k] = union(indsets_pos[k], [i1])
                push!(tree, Node(indsets_neg, indsets_pos))
            end
        end
    end
end

return afs, false

end # module HyperplaneClustering
