include("function.jl")
using Metrics
using Plots: plot
using ProgressBars
mutable struct NeuralNetwork
    W::Array
    b::Array
    α::Float64
    number_per_layer::Array
    """
    init(x::Neuron, X::AbstractMatrix)
Initialisation of a neural network with random weights and bias
# Arguments:
- `x::Neuron`: the neuron to initialize
- `X::AbstractMatrix`: the input data of size (m, n)
- `number_per_layer::Array`: the number of neurons per layer 
"""
    function NeuralNetwork(number_of_feature, number_per_layer::Array)
        layer = [number_of_feature, number_per_layer...]
        W = []
        b = []
        for i in 2:size(layer)[1]
            push!(W, randn((layer[i], layer[i-1])))
            push!(b, randn((layer[i], 1)))
        end
        new(W, b, number_per_layer, 0.01)
    end
end



function forward(x::NeuralNetwork, X::AbstractMatrix)
    A = []
    for i in 1:size(x.number_per_layer)[1]

        answer = i == 1 ? X : A[i-1]
        @show size(A)
        Z = z(x.W[i], answer, x.b[i])
        @show size(Z)
        push!(A, a(Z))
    end
    return A
end
x = NeuralNetwork()
init(x, [1 4 5 6], [2, 1])
@show forward(x, [1 4 5 6])
