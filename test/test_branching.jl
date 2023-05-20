module MyTest

using Test
@static if isdefined(Main, :TestLocal) && Main.TestLocal
    include("../src/HyperplaneClustering.jl")
else
    using HyperplaneClustering
end
HC = HyperplaneClustering
Node = HC.Node

Base.:(==)(n1::Node, n2::Node) = (n1.indsets_neg == n2.indsets_neg &&
                                  n1.indsets_pos == n2.indsets_pos)

K = 3
tree = Node[]
node = Node([Int[], Int[], Int[]],
            [Int[], Int[], Int[]])

HC.add_children!(tree, node, 1, 2, K)

@testset "1" begin
    @test tree == [Node([[1], Int[], Int[]],
                        [[2], Int[], Int[]])]
end

node = tree[1]
empty!(tree)
HC.add_children!(tree, node, 3, 2, K)

@testset "2" begin
    @test tree == [Node([[1, 3], Int[], Int[]],
                        [[2], Int[], Int[]]),
                   Node([[1], Int[3], Int[]],
                        [[2], Int[2], Int[]])]
end

empty!(tree)
HC.add_children!(tree, node, 3, 4, K)

@testset "3" begin
    @test tree == [Node([[1, 3], Int[], Int[]],
                        [[2, 4], Int[], Int[]]),
                   Node([[1, 4], Int[], Int[]],
                        [[2, 3], Int[], Int[]]),
                   Node([[1], Int[3], Int[]],
                        [[2], Int[4], Int[]])]
end

end # module