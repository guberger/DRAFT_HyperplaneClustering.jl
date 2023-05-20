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
afs = [AffForm([1, 0], -1.5), AffForm([1, 1], -3.0)]

classes = HC.classify(afs, datavec, Q)

@testset "main" begin
    length(keys(classes)) == 4
    #
    @test sort(classes[hash([true, true])][1]) == [11, 12, 15, 16]
    @test sort(classes[hash([true, true])][2]) == Int[]
    @test sort(classes[hash([true, true])][3]) == [14]
    #
    @test sort(classes[hash([true, false])][1]) == Int[]
    @test sort(classes[hash([true, false])][2]) == Int[]
    @test sort(classes[hash([true, false])][3]) == [9, 10, 13]
    #
    @test sort(classes[hash([false, true])][1]) == [8]
    @test sort(classes[hash([false, true])][2]) == Int[]
    @test sort(classes[hash([false, true])][3]) == Int[]
    #
    @test sort(classes[hash([false, false])][1]) == [3, 4, 7]
    @test sort(classes[hash([false, false])][2]) == [1, 2, 5, 6]
    @test sort(classes[hash([false, false])][3]) == Int[]
end

end # module