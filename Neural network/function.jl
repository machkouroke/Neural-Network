
using LinearAlgebra


"""Fonction d'aggrégation (Forward propagation)"""
function z(W::Array, X::AbstractMatrix, b::Array)::Array
    return [W[i] * X .+ b[i] for i in 1:size(W)[1]]
end



"""Fonction d'activation"""
function a(Z::Array)
    return 1 ./ (1 .+ exp.(-Z[i]))
end



"""Fonction de cout"""
function log_loss(A::AbstractMatrix, y::AbstractMatrix, ϵ::Float64=1e-15)
    m = size(y)[1]
    return -(1/m) * sum(y .* log.(A .+ ϵ) + (1 .- y) .* log.(1 .- A .+ ϵ))
end



"""Gradient"""
function ∂LW(X::AbstractMatrix, y::AbstractMatrix, A::AbstractMatrix)
    m = size(y)[1]
    return m^-1 * (X' * (A - y))
end


function ∂Lb(X::AbstractMatrix, y::AbstractMatrix, A::AbstractMatrix)
    m = size(y)[1]
    return m^-1 * sum(A - y)
end

function update(dW::AbstractMatrix, db::Float64, W::AbstractMatrix, b::Float64, α::Float64)
    W = W - α * dW
    b = b - α * db
    return W, b
end

