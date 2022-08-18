
using LinearAlgebra


"""Fonction d'aggrégation"""
function z(W::AbstractMatrix, X::AbstractMatrix, b::Float64)
    return X * W .+ b
end



"""Fonction d'activation"""
function a(Z::AbstractMatrix)
    println(findmax(Z)[1], findmin(Z)[1])

    if any(y -> y >= 710, Z)
        print("Overflow")
    end
    return 1 ./ (1 .+ exp.(-Z))
end



"""Fonction de cout"""
function log_loss(A::AbstractMatrix, y::AbstractMatrix)
    m = size(y)[1]
    return -(1/m) * sum(y .* log.(A) + (1 .- y) .* log.(1 .- A))
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

