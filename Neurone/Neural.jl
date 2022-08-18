include("function.jl")
using Metrics
using Plots: plot
Base.@kwdef mutable struct Neuron
    W::Array = []
    b::Float64=0
    α::Float64=0.0001
end
function Base.show(io::IO, x::Neuron)
    print(io, "W:$(x.W) \nb:$(x.b)\nα:$(x.α)")
end
"""
    init(x::Neuron, X::AbstractMatrix)
Initialisation of a neuron with random weights and bias
# Arguments:
- `x::Neuron`: the neuron to initialize
- `X::AbstractMatrix`: the input data 
"""
function init(x::Neuron, X::AbstractMatrix)
    x.W = randn((size(X)[2], 1))
    x.b = randn()
    return Nothing
end

"""
    Make the gradient descent step
# Arguments:
- `neuron::Neuron`: the neuron in which we pass the data
- `X::AbstractMatrix`: the input data
- `y::AbstractMatrix`: the expected output
- `α::Float64`: the learning rate
- `iter::Int64`: the number of iterations
"""
function gradient(X::AbstractMatrix, y::AbstractMatrix, neuron::Neuron,  α::Float64; iter::Int64=10000)
    W, b = neuron.W, neuron.b
    loss = []
    accuracy = []
    println("Start of gradient")
    for i in 1:iter
        # println("Iteration:$(i)")
        A = a(z(neuron.W, X, neuron.b))
        push!(loss, log_loss(A, y))
        dW, db = ∂LW(X, y, A), ∂Lb(X, y, A)
        neuron.W, neuron.b = update(dW, db, neuron.W, neuron.b, α)
    end
    y_pred = predict(neuron, X)

    return W, b, loss
end
"""
    Fit the neuron to the data
# Arguments:
- `neuron::Neuron`: the neuron to fit
- `data::AbstractMatrix`: the input data
- `output::AbstractMatrix`: the expected output
"""
function fit(x::Neuron, data::AbstractMatrix, output::AbstractMatrix)
    x.W, x.b, loss = gradient(data, output, x, x.α)
    return loss
end

"""
    predict(x::Neuron, data::AbstractMatrix)
Predict the output of the neuron for a given input
# Arguments:
- `neuron::Neuron`: the neuron to predict
- `data::AbstractMatrix`: the data for which we predict the output. 
    The shape of the array must be (n, m) where n is the number of samples and m is the number of features.

"""
function predict(x::Neuron, data::AbstractMatrix)
    return a(z(x.W, data, x.b)) .>= 0.5
end

"""
    Compute the accuracy of the neuron for a given input and output
# Arguments:
- `neuron::Neuron`: the neuron to evaluate
- `data::AbstractMatrix`: the data for which we evaluate the accuracy
    The shape of the array must be (n, m) where n is the number of samples and m is the number of features.
- `output::AbstractMatrix`: the expected output for the data
    The shape of the output must be (n, 1) where n is the number of samples.
"""
function score(x::Neuron, data::AbstractMatrix, output::AbstractMatrix)
    return sum(predict(x, data) .== output) / size(output)[1]
end

