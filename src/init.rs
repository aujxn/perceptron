use crate::{Sample, Weights};
use anyhow::Result;
use csv::Reader;
use nalgebra::{DMatrix, DVector};
use rand::{thread_rng, Rng};
use std::fs::File;

/* Loads the mnist data from the csv and normalizes the data to be between 0 and 1
 * Returns a tuple, first element is the training set and the second is the test set */
pub fn read_and_normalize(num_training_examples: usize) -> Result<(Vec<Sample>, Vec<Sample>)> {
    let test_rdr = Reader::from_path("./data/mnist_test.csv")?;
    let train_rdr = Reader::from_path("./data/mnist_train.csv")?;

    // closure that takes a reader and produces a vector of samples
    let read = |mut rdr: Reader<File>| -> Vec<Sample> {
        rdr.records()
            .take(num_training_examples)
            .map(|sample| {
                let sample = sample.unwrap();

                /* Create target vector */
                // Create the target vector, initialized to 10 0.1's
                let mut target = vec![0.1f64; 10];
                // The first item in the record is the label for the sample
                let label = sample[0].parse::<usize>().unwrap();
                // Change the value in the target vector at the index of the label to 0.9
                target[label] = 0.9;
                let target = DVector::from_vec(target);

                /* Create input vector */
                let input = DVector::from_vec(
                    // Start with a 1.0 as the bias for the input into the hidden layer
                    vec![1.0]
                        .into_iter()
                        // Chain the rest of the data onto this vector to create the input
                        .chain(
                            sample
                                .iter()
                                // Skip the first value because that is the label
                                .skip(1)
                                // Normalize the byte values into floats by dividing by 255.0
                                .map(|string| string.parse::<usize>().unwrap() as f64 / 255.0),
                        )
                        .collect(),
                );
                Sample {
                    label,
                    input,
                    target,
                }
            })
            .collect()
    };

    Ok((read(train_rdr), read(test_rdr)))
}

/* Creates matrices with random values within +-range for weights and momentum matrices
 * with initial values of 0 */
pub fn init_random_weights(hidden_dimension: usize, range: f64) -> Weights {
    let mut rng = thread_rng();

    // The hidden weights is a 785 by hidden dimension matrix, the input vector
    // has a length of 784 but a bias node of 1.0 is inserted into the input features
    // meaning 785 weights are needed. Each row of the matrix corresponds to a value
    // in the feature and each column corresponds to neuron in the network.
    let iter = (0..785 * hidden_dimension).map(|_| rng.gen_range(-range, range));
    let hidden_weights = DMatrix::from_iterator(785, hidden_dimension, iter);
    let hidden_momentum = DMatrix::<f64>::zeros(785, hidden_dimension);

    // The output weights is a hidden dimension + 1 (for bias) by 10 matrix. Each row
    // is for a hidden neuron and each column is for and output value (0-9).
    let iter = (0..(hidden_dimension + 1) * 10).map(|_| rng.gen_range(-range, range));
    let output_weights = DMatrix::from_iterator(hidden_dimension + 1, 10, iter);
    let output_momentum = DMatrix::<f64>::zeros(hidden_dimension + 1, 10);

    Weights(
        hidden_weights,
        hidden_momentum,
        output_weights,
        output_momentum,
    )
}
