import tensorflow as tf
import pandas as pd

tf.logging.set_verbosity(tf.logging.INFO)

# Data sets
GGG_TRAINING = "/home/victor/gitProjects/tfg/ghouls-goblins-and-ghosts-boo/train.csv"
GGG_TEST = "/home/victor/gitProjects/tfg/ghouls-goblins-and-ghosts-boo/test.csv"

FEATURES = ["bone_length", "rotting_flesh", "hair_length",
            "has_soul", "color"]
LABEL = "type"


def get_input_fn(data_set, y_none=False, num_epochs=None, shuffle=True):
    if y_none:
        y = None
    else:
        y = pd.Series(data_set[LABEL].values)
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
        y=y,
        num_epochs=num_epochs,
        shuffle=shuffle)


def main(hl, steps, optimizer, activation_fn):
    # Hidden layer and its units
    HIDDEN_LAYERS = hl
    # Steps for train the model
    STEPS = steps
    # Optimizer
    if optimizer == 'Adadelta':
        OPTIMIZER = 'tf.train.' + optimizer + 'Optimizer()'
    elif optimizer == 'Adam':
        OPTIMIZER = 'tf.train.' + optimizer + 'Optimizer()'
    else:
        OPTIMIZER = 'tf.train.' + optimizer + 'Optimizer(learning_rate=0.001)'
    # Activation fuction
    if activation_fn == 'sigmoid':
        ACTIVATION_FN = 'tf.' + activation_fn
    elif activation_fn == 'tanh':
        ACTIVATION_FN = 'tf.' + activation_fn
    else:
        ACTIVATION_FN = 'tf.nn.' + activation_fn

    # Load datasets
    training_set = pd.read_csv(GGG_TRAINING, index_col=0)
    test_set = pd.read_csv(GGG_TEST, index_col=0)

    # Transform the label column to int values in training set
    training_set['type'] = training_set['type'].str.replace('Ghoul', '0')
    training_set['type'] = training_set['type'].str.replace('Goblin', '1')
    training_set['type'] = training_set['type'].str.replace('Ghost', '2')
    training_set['type'] = pd.to_numeric(training_set['type'])

    # Feature columns
    bone_length = tf.feature_column.numeric_column("bone_length")
    rotting_flesh = tf.feature_column.numeric_column("rotting_flesh")
    hair_length = tf.feature_column.numeric_column("hair_length")
    has_soul = tf.feature_column.numeric_column("has_soul")
    color = tf.feature_column.categorical_column_with_vocabulary_list(
        "color", vocabulary_list=["white", "black", "clear", "blue", "green", "blood"])
    columns = [bone_length,
               rotting_flesh,
               hair_length,
               has_soul,
               tf.feature_column.indicator_column(color)]

    # Build 3 layer DNN with 64, 128, 256 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=columns,
        hidden_units=[hl for hl in HIDDEN_LAYERS],
        n_classes=3,
        optimizer=eval(OPTIMIZER),
        activation_fn=eval(ACTIVATION_FN),
        model_dir="/tmp/ggg_model_{}_{}_{}_{}".format('_'.join(str(x) for x in HIDDEN_LAYERS),
                                             optimizer,
                                             ACTIVATION_FN,
                                             STEPS))

    # Define the training inputs
    train_input_fn = get_input_fn(training_set)

    # Train model.
    classifier.train(input_fn=train_input_fn, steps=STEPS)

    # Define the test inputs
    # test_input_fn = get_input_fn(prediction_set, num_epochs=1, shuffle=False)

    # Evaluate for one step (one pass through the test data).
    # results = classifier.evaluate(input_fn=test_input_fn)

    # Print the stats for the evaluation.
    # for key in sorted(results):
    #     print("%s: %s" % (key, results[key]))

    # Predict the samples in test.csv
    test_input_fn = get_input_fn(test_set, y_none=True, num_epochs=1, shuffle=False)
    predictions = list(classifier.predict(input_fn=test_input_fn))
    predicted_classes = [p["classes"][0] for p in predictions]
    # print("New Samples, Class Predictions:    {}\n".format(predicted_classes))

    # Replace int values by creature name for submission
    for p in range(len(predicted_classes)):
        if predicted_classes[p] == "0":
            predicted_classes[p] = "Ghoul"
        if predicted_classes[p] == "1":
            predicted_classes[p] = "Goblin"
        if predicted_classes[p] == "2":
            predicted_classes[p] = "Ghost"

    # Create submission file
    with open('submission_dnn_{}_{}_{}_{}.csv'.format('-'.join(str(x) for x in HIDDEN_LAYERS),
                                                   optimizer,
                                                   ACTIVATION_FN,
                                                   STEPS), 'w') as f:
        f.write("id,type\n")
        i = 0
        for index, monster in test_set.iterrows():
            f.write("{},{}\n".format(index, predicted_classes[i]))
            i = i + 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create a DNN')
    parser.add_argument('-hl', '--hidden-layers', required=True,
                        help='the number of units in each hidden layer. Ex: 10,10,10')
    parser.add_argument('-steps', default=5000, type=int,
                        help='the number of steps in training. By default 5000 steps')
    parser.add_argument('-opt', '--optimizer', default='Adagrad',
                        help='the optimizer in training. By default "Adagrad"')
    parser.add_argument('-act', '--activation-fn', default='relu',
                        help='the optimizer in training. By default "relu"')
    args = parser.parse_args()
    # Transform into int list
    hl = args.hidden_layers
    hl = hl.split(',')
    for i in range(len(hl)):
        hl[i] = int(hl[i])
    main(hl=hl, steps=args.steps, optimizer=args.optimizer, activation_fn=args.activation_fn)
