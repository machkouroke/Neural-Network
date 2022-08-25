include("function.jl")
using MLJ
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
        new(W, b, 0.0001, layer)
    end
end



function forward_propagation(x::NeuralNetwork, X::Array)::Array
    A = []
    for i in 1:size(x.number_per_layer)[1]-1
        answer = i == 1 ? X : A[i-1]
        Z = z(x.W[i], answer, x.b[i])
        push!(A, a(Z))
    end
    return A
end

function back_propagation(x::NeuralNetwork, X::Array, Y::Array, A::Array)
    number_of_layer = size(x.number_per_layer)[1] - 1
    dZ = Array{Any}(undef, number_of_layer)
    dW = Array{Any}(undef, number_of_layer)
    db = Array{Any}(undef, number_of_layer)
    m = size(Y)[2]
    for i in number_of_layer:-1:1
        dZ[i] = i == number_of_layer ? A[i] .- Y : (x.W[i+1]' * dZ[i+1]) .* (A[i] .* (1 .- A[i]))
        dW[i] = i == 1 ? (1/m) .* (dZ[i] * X') : (1/m) .* (dZ[i] * A[i-1]')
        db[i] = (1 / m) .* sum(dZ[i], dims=2)
    end
    return dW, db
end

function predict(x::NeuralNetwork, data::Array)
    final_result = forward_propagation(x, data)[end]
    return final_result .>= 0.5 
end
function fit(x::NeuralNetwork, data::Array, output::Array, data_test, y_test)
    loss, accuracy, accuracy_test = gradient(x, data, output, data_test, y_test)
    return loss, accuracy, accuracy_test
end
function score(x::NeuralNetwork, data::AbstractMatrix, output::AbstractMatrix)
    return sum(predict(x, data) .== output) / size(output)[2]
end


function gradient(network::NeuralNetwork, data::AbstractMatrix, y::AbstractMatrix, data_test, y_test; iter::Int64=10000)
    loss = []
    accuracy = []
    accuracy_test = []
    println("Start of gradient")
    for i in ProgressBar(1:iter)
        A = forward_propagation(network, data)
        y_pred = predict(network, data)
        dW, db = back_propagation(network, data, y, A)
        neuron.W, neuron.b = update(dW, db, neuron.W, neuron.b, network.α)
        push!(accuracy, binary_accuracy(y_pred', y'))
        push!(loss, log_loss(A[end], y))
        y_test_pred = predict(network, data_test)
        push!(accuracy_test, binary_accuracy(y_test_pred', y_test'))
    end
    return loss, accuracy, accuracy_test
end
# n_feature = 4096
# n_element = 1200
# n_neural_per_layer = [2, 2, 1]
# X, y = make_blobs(n_element, n_feature; centers=2, as_table=false)
# X = permutedims(X, (2, 1))
# y = [i == 1 ? 0 : 1 for i in reshape(y, (1, n_element))]
# X_train, y_train, X_test, y_test = X[:, 1:1000], y[:, 1:1000], X[:, 1001:1200], y[:, 1001:1200]
# @show size(X_train)
# @show size(X_test)
# @show size(y_test)
# @show size(y_train)
# neuron = NeuralNetwork(n_feature, n_neural_per_layer)
# loss, accuracy, accuracy_test = fit(neuron, X_train, y_train, X_test, y_test)

# @show accuracy_test[end]
# @show accuracy[end]
# p1 = plot(loss, title="Loss")
# p2 = plot(accuracy, title="Accuracy")
# p3 = plot(accuracy_test, title="Accuracy Test")
# plot(p1, p2, p3, layout=(1,3))