module HyperplaneClustering

using LinearAlgebra
using JuMP

struct AffForm
    a::Vector{Float64}
    β::Float64
end

evalf(af::AffForm, x) = dot(af.a, x) + af.β

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
    Q = maximum(datum -> datum.σ, data, init=0)::Int
    datavec = [datum for datum in data]::Vector{Datum}
    xs = [datum.x for datum in datavec]
    return datavec, xs, Q
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
    end
    return true
end

signfs(afs::Vector{AffForm}, x) = [evalf(af, x) > 0 for af in afs]

function classify(afs, datavec, Q)
    classes = Dict{UInt,Vector{Vector{Int}}}()
    for (i, datum) in enumerate(datavec)
        c = hash(signfs(afs, datum.x))
        indsets = get(classes, c, [Int[] for q = 1:Q])
        push!(indsets[datum.σ], i)
        classes[c] = indsets
    end
    return classes
end

function closest_distance(x1s, x2s)
    distmin = Inf
    j1opt, j2opt = -1, -1
    for (j1, x1) in enumerate(x1s)
        for (j2, x2) in enumerate(x2s)
            dist = norm(x1 - x2)
            if dist < distmin
                distmin = dist
                j1opt, j2opt = j1, j2
            end
        end
    end
    return distmin, j1opt, j2opt
end

function candidate_verification!(xlists, afs, datavec, xs, Q)
    classes = classify(afs, datavec, Q)
    distmin = Inf
    i1opt, i2opt = -1, -1
    for indsets in values(classes)
        for q = 1:Q
            populatexs!(xlists[q], indsets[q], xs)
        end
        for q1 = 1:Q
            for q2 = (q1 + 1):Q
                dist, j1, j2 = closest_distance(xlists[q1], xlists[q2])
                if dist < distmin
                    distmin = dist
                    i1opt, i2opt = indsets[q1][j1], indsets[q2][j2]
                end
            end
        end
    end
    return distmin, i1opt, i2opt
end

function copypush(A::Vector{T}, x::T) where T
    L = length(A) + 1
    B = Vector{T}(undef, L)
    @inbounds copyto!(B, A)
    @inbounds B[L] = x
    return B
end

function child_node(node, k, ineg::Int, ipos::Int)
    indsets_neg = copy(node.indsets_neg)
    indsets_neg[k] = copypush(indsets_neg[k], ineg)
    indsets_pos = copy(node.indsets_pos)
    indsets_pos[k] = copypush(indsets_pos[k], ipos)
    return Node(indsets_neg, indsets_pos)
end

function child_node(node, k, ineg::Int, ::Nothing)
    indsets_neg = copy(node.indsets_neg)
    indsets_neg[k] = copypush(indsets_neg[k], ineg)
    return Node(indsets_neg, node.indsets_pos)
end

function child_node(node, k, ::Nothing, ipos::Int)
    indsets_pos = copy(node.indsets_pos)
    indsets_pos[k] = copypush(indsets_pos[k], ipos)
    return Node(node.indsets_neg, indsets_pos)
end

function add_nodes!(tree, node, k, i1, i2)::Bool
    if isempty(node.indsets_neg[k]) && isempty(node.indsets_pos[k])
        push!(tree, child_node(node, k, i1, i2))
        return true
    end
    for (j1, j2) in ((i1, i2), (i2, i1))
        if j1 ∈ node.indsets_neg[k]
            push!(tree, child_node(node, k, nothing, j2))
            return false
        end
        if j1 ∈ node.indsets_pos[k]
            push!(tree, child_node(node, k, j2, nothing))
            return false
        end
    end
    push!(tree, child_node(node, k, i1, i2))
    push!(tree, child_node(node, k, i2, i1))
    return false
end

function add_children!(tree, node, i1, i2, K)
    for k = 1:K
        add_nodes!(tree, node, k, i1, i2) && break
    end
end

function learn_hyperplanes(data, N, K, ϵ, solver)
    datavec, xs, Q = processdata(data)
    root = Node([Int[] for k = 1:K], [Int[] for k = 1:K])
    tree = [root]
    afs = [nanafs(N) for k = 1:K]
    xs_neg = Vector{Float64}[]
    xs_pos = Vector{Float64}[]
    xlists = [Vector{Float64}[] for q = 1:Q]
    rmax = 1.0

    while !isempty(tree)
        node = pop!(tree)

        # Candidate generation
        !candidate_generation!(afs, xs_neg, xs_pos,
                               node, xs,
                               N, K, ϵ, rmax, solver) && continue

        # Candidate verification
        dist, i1, i2 = candidate_verification!(xlists, afs, datavec, xs, Q)
        @assert dist > 0
        if i1 < 0 || i2 < 0
            return afs, true
        end

        # Branching
        add_children!(tree, node, i1, i2, K)
    end

    return afs, false
end

end # module HyperplaneClustering
