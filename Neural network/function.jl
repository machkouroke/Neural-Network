
using LinearAlgebra


"""Fonction d'aggrégation (Forward propagation)"""
function z(W::Matrix, X::Array, b::Array)::Matrix
    return W * X .+ b
end



"""Fonction d'activation"""
function a(Z::Array)::Matrix
    return 1 ./ (1 .+ exp.(-Z))
end

function update(dW::Array, db::Array, W::Array, b::Array, α::Array)
    new_W = []
    new_b = []
    for i in 1:eachindex(dW)
        push!(new_W, W[i] - α .* dW[i])
        push!(new_b, b[i] - α .* db[i])
    end
    return new_W, new_b
end




