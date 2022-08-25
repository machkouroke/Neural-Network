
using LinearAlgebra


"""Fonction d'aggrégation (Forward propagation)"""
function z(W::Matrix, X::Array, b::Array)::Matrix
    return W * X .+ b
end



"""Fonction d'activation"""
function a(Z::Array)::Matrix
    return 1 ./ (1 .+ exp.(-Z))
end
"""Fonction de cout"""
function log_loss(A::AbstractMatrix, y::AbstractMatrix, ϵ::Float64=1e-15)
    m = size(y)[1]
    return -(1/m) * sum(y .* log.(A .+ ϵ) + (1 .- y) .* log.(1 .- A .+ ϵ))
end


function update(dW::Array, db::Array, W::Array, b::Array, α::Float64)::Tuple
    new_W = []
    new_b = []
    for i in eachindex(dW)
        push!(new_W, W[i] .- α .* dW[i])
        push!(new_b, b[i] .- α .* db[i])
    end
    return new_W, new_b
end


