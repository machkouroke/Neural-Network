include("function.jl")
using Metrics
using Plots: plot
using ProgressBars
mutable struct NeuralNetwork
    W::Array{Matrix}
    b::Array{Matrix}
    α::Float64
    number_per_layer::Array
    """
    init(x::Neuron, X::AbstractMatrix)
    Initialisation of a neural network with random weights and bias
    # Arguments:
    - `number_of_feature::Int64`: number of features of the input data
    - `number_per_layer::Array`: the number of neurons per layer 
    """
    function NeuralNetwork(number_of_feature::Int64, number_per_layer::Array)
        layer = [number_of_feature, number_per_layer...]
        W = [randn((layer[i], layer[i-1])) for i in 2:size(layer)[1]]
        b = [randn((layer[i], 1)) for i in 2:size(layer)[1]]
        @show layer
        new(W, b, 0.01, layer)
    end
end



function forward_propagation(x::NeuralNetwork, X::Array)::Array
    A = []
    for i in 1:size(x.number_per_layer)[1]-1
        answer = i == 1 ? X : A[i-1]
        Z = z(x.W[i], answer, x.b[i])
        push!(A, a(Z))
    end
    return [X, A...]
end

function back_propagation(x::NeuralNetwork, X::Array, Y::Array, A::Array)
    number_of_layer = length(x.number_per_layer) - 1
    dZ = Array{Any}(undef, number_of_layer)
    dW = Array{Any}(undef, number_of_layer)
    db = Array{Any}(undef, number_of_layer)
    m = size(Y)[1]
    for i in number_of_layer:-1:1
        dZ[i] = i == number_of_layer ? A[i] .- Y : (x.W[i+1] * dZ[i+1]) .* (A[i] .* (1 .- A[i]))
        dW[i] = i == 1 ? (1/m) * (dZ[i] * X') : (1/m) * (dZ[i] * A[i-1]')
        db[i] = (1 / m) .* sum(dZ[i], dims=2)
    end
    return dW, db
end

function predict(x::NeuralNetwork, data::Array)
    final_result = forward_propagation(x, data)[end][1]
    return final_result > 0.5 
end

function gradient(network::NeuralNetwork, data::AbstractMatrix, y::AbstractMatrix; iter::Int64=10000)
    loss = []
    accuracy = []
    println("Start of gradient")
    for i in ProgressBar(1:iter)
        A = forward_propagation(network, data)
        dW, db = back_propagation(network, data, y, A)
        neuron.W, neuron.b = update(dW, db, neuron.W, neuron.b, network.α)
        y_pred = predict(network, data)
        push!(accuracy, binary_accuracy(y_pred, y))
        push!(loss, log_loss(A[end], y))
        dW, db = ∂LW(X, y, A), ∂Lb(X, y, A)
        neuron.W, neuron.b = update(dW, db, neuron.W, neuron.b, α)
    end
    y_pred = predict(neuron, X)
    println("Accuracy Train Set: ", binary_accuracy(y_pred, y))
    return W, b, loss, accuracy
end
x = NeuralNetwork(2, [2, 1])
@show predict(x, [4; 8])
# forw = forward_propagation(x, [4; 8])

# w, b = back_propagation(x, [4; 8], [1; 0], forw)
# @show w
# @show b
