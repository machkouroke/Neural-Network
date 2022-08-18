using HDF5
using Plots: plot
function data_load()
    permute(a) = permutedims(read(a), collect(reverse(1:ndims(a))))
    X_train = permute(h5open("data/trainset.hdf5", "r")["X_train"])
    y_train = permute(h5open("data/trainset.hdf5", "r")["Y_train"])
    X_test = permute(h5open("data/testset.hdf5", "r")["X_test"])
    y_test = permute(h5open("data/testset.hdf5", "r")["Y_test"])
    return X_train, y_train, X_test, y_test
end

function normalize(x::Array)
    return x ./ findmax(x)[1]
end
function flatten_image(x::Array)
    flatten_array = reshape(x, (size(x)[begin], :))
    return flatten_array
end



function preprocess_data()
    X_train, y_train, X_test, y_test = data_load()
    X_train, X_test = flatten_image(X_train), flatten_image(X_test)
 
    X_train, X_test = normalize.((X_train, X_test))
    

    return X_train, y_train, X_test, y_test
end

