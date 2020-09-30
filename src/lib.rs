use nalgebra::{DMatrix, DVector};
use serde::Serialize;

pub mod init;
pub mod network;

// Stores information about a network's accuracy at a particular epoch in training
#[derive(Serialize)]
pub struct TestResult {
    epoch: usize,
    hidden_dimension: usize,
    learning_rate: f64,
    momentum: f64,
    note: String, // Test data or train data? or other like restricted training data etc.
    accuracy: f64,
    train_set_size: usize,
    test_set_size: usize,
    training_time: i64, // In seconds
    confusion_matrix: Vec<Vec<usize>>,
}

// TypeDef for weights and momentums for the network
// Order: (hidden weights, hidden_momentum, output_weights, output_momentum)
pub struct Weights(DMatrix<f64>, DMatrix<f64>, DMatrix<f64>, DMatrix<f64>);

// Represents a single "number" from the mnist data set (training or testing)
pub struct Sample {
    // What number it is
    label: usize,
    // Input has been normalize to be between 0 and 1 and bias has been added at index 0
    input: DVector<f64>,
    // A construct target vector with 0.1 in the wrong spots and 0.9 in the correct one
    target: DVector<f64>,
}

/* Sigmoid activation function */
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (x * -1.0).exp())
}
