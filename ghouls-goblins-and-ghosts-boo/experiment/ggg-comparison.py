from subprocess import call

def main():
    DNN = "../ggg_dnn.py"
    OPTIMIZERS = ["Adadelta", "Adagrad", "Adam", "Ftrl", "GradientDescent",
                  "ProximalAdagrad", "ProximalGradientDescent", "RMSProp"]
    ACTIVATION_FNS = ["relu", "relu6", "crelu", "elu", "selu", "softplus",
                      "softsign", "sigmoid", "tanh"]
    HL_GNN = ["5", "10"]

    for hl in HL_GNN:
        for opt in OPTIMIZERS:
            for act in ACTIVATION_FNS:
                call(["python", DNN, "-hl=" + hl, "-steps=10000", "-opt=" + opt, "-act=" + act])


if __name__ == '__main__':
    main()
