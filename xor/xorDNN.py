import numpy as np
import tensorflow as tf

# Data sets
XOR_TRAINING = "xorTrain.csv"

def main():
    # Load datasets.
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=XOR_TRAINING,
        target_dtype=np.int32,
        features_dtype=np.int32)

    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column("x", shape=[2])]

    # Configure the DNN
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[4],
                                            n_classes=2,
                                            model_dir="/tmp/xor_model")
    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)

    # Train model.
    classifier.train(input_fn=train_input_fn, steps=2000)

    # Classify two xor samples.
    new_samples = np.array(
        [[0, 0],
         [0, 1]], dtype=np.int32)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": new_samples},
        num_epochs=1,
        shuffle=False)

    predictions = list(classifier.predict(input_fn=predict_input_fn))
    predicted_classes = [p["classes"][0] for p in predictions]

    print("New Samples: ", new_samples)
    print(
        "New Samples, Class Predictions:    {}\n"
            .format(predicted_classes))


if __name__ == "__main__":
    main()
