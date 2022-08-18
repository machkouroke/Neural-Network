using MLJ
using LopShelve: open!
include("Neurone/Neural.jl")
include("load_data.jl")

open!("cat_dog") do blobs_neuron 
    
    X_train, y_train, X_test, y_test = preprocess_data()

    # Initialisation du réseau de neurones
    neuron = Neuron()
    init(neuron, X_train)
    loss, accuracy = fit(neuron, X_train, y_train)

    final_score_test = score(neuron, X_test, y_test)
    println("Score_test:$(final_score_test)")
    # print(loss)
    p1 = plot(loss)
    p2 = plot(accuracy)
    plot(p1, p2, layout=(1,2))
end 

using MLJ
using LopShelve: open!
include("Neurone/Neural.jl")
include("load_data.jl")

open!("cat_dog") do blobs_neuron 
    
    X_train, y_train, X_test, y_test = preprocess_data()

    # Initialisation du réseau de neurones
    neuron = Neuron()
    init(neuron, X_train)
    loss, accuracy = fit(neuron, X_train, y_train)
    y_pred = predict(neuron, X_test)
    println("Accuracy Test Set: ", binary_accuracy(y_pred, y_test))
    final_score_test = score(neuron, X_test, y_test)
    println("Score_test:$(final_score_test)")

    # print(loss)
    p1 = plot(loss)
    p2 = plot(accuracy)
    plot(p1, p2, layout=(1,2))
end 

