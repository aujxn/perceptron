use crate::init::*;
use crate::Sample;
use crate::{sigmoid, TestResult, Weights};
use anyhow::Result;
use chrono::Local;
use nalgebra::{DMatrix, DVector};
use rand::seq::SliceRandom;
use rand::thread_rng;

/* Trains and runs a variety of different networks and returns the results */
pub fn run_variety(epochs: usize) -> Result<Vec<TestResult>> {
    let mut test_results = vec![];
    test_results.append(&mut train_and_run(20, 0.05, 0.1, 0.9, epochs, 60000)?);
    test_results.append(&mut train_and_run(50, 0.05, 0.1, 0.9, epochs, 60000)?);
    test_results.append(&mut train_and_run(100, 0.05, 0.1, 0.9, epochs, 60000)?);
    test_results.append(&mut train_and_run(100, 0.05, 0.1, 0.0, epochs, 60000)?);
    test_results.append(&mut train_and_run(100, 0.05, 0.1, 0.25, epochs, 60000)?);
    test_results.append(&mut train_and_run(100, 0.05, 0.1, 0.5, epochs, 60000)?);
    test_results.append(&mut train_and_run(100, 0.05, 0.1, 0.9, epochs, 30000)?);
    test_results.append(&mut train_and_run(100, 0.05, 0.1, 0.9, epochs, 15000)?);
    Ok(test_results)
}

/* Trains a network with the provided parameters, tests it at each epoch, and returns the results */
pub fn train_and_run(
    hidden_dimension: usize,
    weight_range: f64,
    learning_rate: f64,
    momentum: f64,
    epochs: usize,
    training_examples: usize,
) -> Result<Vec<TestResult>> {
    let mut test_results = vec![];

    // Load the data
    println!("Loading data...");
    let (mut train_set, test_set) = read_and_normalize(training_examples)?;

    // Initialize random weights and set momentums to 0
    println!("Initializing random weights...");
    let Weights(mut hidden_weights, mut hidden_momentum, mut output_weights, mut output_momentum) =
        init_random_weights(hidden_dimension, weight_range);

    println!(
        "Starting training loop. Network information:
              Hidden layer size (not including bias): {}
              Initial weight range: {:.3} - {:.3}
              Learning rate: {:.3}
              Momentum Coefficient: {:.3}
              Number of epochs: {}
              Input size: 784",
        hidden_dimension, -weight_range, weight_range, learning_rate, momentum, epochs,
    );

    /* Training loop */
    for epoch in 0..epochs {
        let start_time = Local::now();
        // Shuffle the training data so its a different order each epoch
        train_set.shuffle(&mut thread_rng());

        for sample in train_set.iter() {
            /***********************/
            /* Forward Propagation */
            /***********************/

            // Transpose of the hidden weights times the input (as column vector) to get
            // pre-activated values
            let hidden: DVector<f64> = (hidden_weights.transpose() * &sample.input)
                .map(sigmoid) // Map the sigmoid activation function across ouput
                .insert_row(0, 1.0); // insert the bias at the start of resulting vector

            // Transpose of the output weights times the activated values of the hidden layer and
            // activated again with sigmoid
            let output = (output_weights.transpose() * &hidden).map(sigmoid);

            /*****************/
            /* Backward Prop */
            /*****************/

            let output_error = DVector::from_vec(
                // Take the expected label and calculate the error's for the output layer using the
                // differentiated formula: d_k = o_k(1-o_k)(t_k - o_k)
                sample
                    .target
                    .iter()
                    .zip(output.iter())
                    .map(|(target, output)| (output * (1.0 - output)) * (target - output))
                    .collect(),
            );

            // Calculate the change in the hidden weights before momentum by taking the post
            // activation hidden layer values (h_j) times the learning rate and scale the output
            // error vector by each one of these values. This results in a matrix with the desired
            // change in output weight before adding the momentum.
            let delta_output_weights: Vec<DVector<f64>> = hidden
                .iter()
                .map(|h_j| &output_error * (learning_rate * h_j))
                .collect();
            // Add scaled momentum to the change in weights to get the new momentum values.
            // Add these new momentum values the the weights and then replace old momentums with new.
            let new_output_momentum = DMatrix::from_columns(&delta_output_weights).transpose()
                + (output_momentum.scale(momentum));
            output_weights = output_weights + &new_output_momentum;
            output_momentum = new_output_momentum;

            // Calculate the error for the hidden layer. Skipping the bias neuron, take the dot
            // product of the output weights with the output error vector. Multiply this product
            // by the result of h_j * (1 - h_j) formula to get errors.
            let hidden_error = DVector::from_vec(
                (1..hidden_dimension + 1)
                    .map(|i| {
                        let h_j = hidden[i];
                        let sum = output_weights.row(i).dot(&output_error.transpose());
                        h_j * (1.0 - h_j) * sum
                    })
                    .collect(),
            );

            // Calculate the change in weights before momentum for the input to hidden layer
            // weights by doing the same thing as with hidden to output but with the new error
            // values instead of output errors and input values instead of post activation hidden
            // values.
            let delta_hidden_weights: Vec<DVector<f64>> = sample
                .input
                .iter()
                .map(|x_i| &hidden_error * (learning_rate * x_i))
                .collect();
            // Same as before but now for hidden layer: add momentum, update weights, update momentum
            let new_hidden_momentum = DMatrix::from_columns(&delta_hidden_weights).transpose()
                + (momentum * &hidden_momentum);
            hidden_weights = hidden_weights + &new_hidden_momentum;
            hidden_momentum = new_hidden_momentum;
        }

        /*******************/
        /* Network Testing */
        /*******************/

        let end_time = Local::now();
        let test_set_size = test_set.len();
        let training_time = (end_time - start_time).num_seconds();

        // Calculate accuracy on training data and testing data sets and save results
        let (test_accuracy, test_confusion) = test(&test_set, &hidden_weights, &output_weights);
        let (train_accuracy, train_confusion) = test(&train_set, &hidden_weights, &output_weights);

        let test_result = TestResult {
            epoch,
            hidden_dimension: hidden_dimension,
            learning_rate: learning_rate,
            momentum: momentum,
            note: String::from("test set"),
            accuracy: test_accuracy,
            train_set_size: training_examples,
            test_set_size,
            training_time,
            confusion_matrix: test_confusion,
        };
        let train_result = TestResult {
            epoch,
            hidden_dimension: hidden_dimension,
            learning_rate: learning_rate,
            momentum: momentum,
            note: String::from("train set"),
            accuracy: train_accuracy,
            train_set_size: training_examples,
            test_set_size,
            training_time,
            confusion_matrix: train_confusion,
        };

        test_results.push(test_result);
        test_results.push(train_result);

        println!(
            "Epoch {:03} of {:03}:    test set accuracy - {:.2}%    train set accuracy - {:.2}%",
            epoch,
            epochs,
            test_accuracy * 100.0,
            train_accuracy * 100.0
        )
    }

    Ok(test_results)
}

/* Tests a network and returns the accuracy value with a confusion matrix */
pub fn test(
    input: &Vec<Sample>,
    hidden_weights: &DMatrix<f64>,
    output_weights: &DMatrix<f64>,
) -> (f64, Vec<Vec<usize>>) {
    let mut num_correct = 0.0;
    let total = input.len();
    let mut confusion_matrix = vec![vec![0; 10]; 10];

    for sample in input.iter() {
        let hidden: DVector<f64> = (hidden_weights.transpose() * &sample.input)
            .map(sigmoid)
            .insert_row(0, 1.0); // insert the bias
        let output = (output_weights.transpose() * hidden).map(sigmoid);

        let class = output
            .iter()
            .enumerate()
            .fold(
                (0, 0.0),
                |acc, (label, val)| {
                    if *val > acc.1 {
                        (label, *val)
                    } else {
                        acc
                    }
                },
            );

        if class.0 == sample.label {
            num_correct += 1.0;
        }

        confusion_matrix[sample.label][class.0] += 1;
    }

    (num_correct / total as f64, confusion_matrix)
}
