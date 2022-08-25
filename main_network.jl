using MLJ
using LopShelve: open!
include("Neural network/Network.jl")
include("load_data.jl")

trans(X) = permutedims(X, (2, 1))
X_train, y_train, X_test, y_test = trans.(preprocess_data())
n_feature = size(X_train)[1]
n_element = size(X_train)[2]
n_neural_per_layer = [40, 30, 1]
# Initialisation du réseau de neurones
neuron = NeuralNetwork(n_feature, n_neural_per_layer)
loss, accuracy, accuracy_test = fit(neuron, X_train, y_train, X_test, y_test)

p1 = plot(loss, title="Loss")
p2 = plot(accuracy, title="Accuracy")
p3 = plot(accuracy_test, title="Accuracy Test")
plot(p1, p2, p3, layout=(1,3))
