from subprocess import call

def main():
    DNNS = ["/home/victor/gitProjects/tfg/ghouls-goblins-and-ghosts-boo/ggg_dnn.py", "/home/victor/gitProjects/tfg/forest-cover-type-prediction/forest_cover_type_dnn.py"]
    OPTIMIZERS = ["Adadelta", "Adagrad", "Adam", "Ftrl", "GradientDescent",
                  "ProximalAdagrad", "ProximalGradientDescent", "RMSProp"]
    ACTIVATION_FNS = ["relu", "relu6", "crelu", "elu", "selu", "softplus",
                      "softsign", "sigmoid", "tanh"]
    HL_GNN = ["5", "10", "5,5", "128,256,512,1024"]
    HL_FCT = ["30", "64,128,256", "512,1024,1024,512"]

    # DNN GNN
    for hl in HL_GNN:
        print(hl)
        call(["python", DNNS[0], "-hl=" + hl, "-steps=10000", "-opt=RMSProp", "-act=softsign"])

    # DNN FCT
    for hl in HL_FCT:
        print(hl)
        call(["python", DNNS[1], "-hl=" + hl, "-steps=10000", "-opt=RMSProp", "-act=softplus"])



if __name__ == '__main__':
    main()
