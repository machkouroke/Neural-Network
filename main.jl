using MLJ
using LopShelve: open!
include("Neurone/Neural.jl")
include("load_data.jl")

open!("cat_dog") do blobs_neuron 
    
    X_train, y_train, X_test, y_test = preprocess_data()

    # Initialisation du réseau de neurones
    neuron = Neuron()
    init(neuron, X_train)
    loss = fit(neuron, X_train, y_train)

    final_score_test = score(neuron, X_test, y_test)
    println("Score_test:$(final_score_test)")
    # print(loss)
    plot(loss)
end 

