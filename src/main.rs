use anyhow::Result;
use perceptron::network::{run_variety, train_and_run};
use std::io::Write;
use structopt::StructOpt;

// Configuration for command line args and help tool
#[derive(StructOpt, Debug)]
enum Opt {
    /// Trains and tests a custom network with the given parameters.
    /// Results are saved to data/output.json.
    Custom {
        /// Number of neurons in hidden layer of network
        #[structopt(short, long, default_value = "50")]
        hidden_dimension: usize,

        /// Range in which random weights are created from (between +range and -range)
        #[structopt(short, long, default_value = "0.05")]
        weight_range: f64,

        /// Learning rate coefficient for training
        #[structopt(short, long, default_value = "0.1")]
        learning_rate: f64,

        /// Momentum coefficient for training
        #[structopt(short, long, default_value = "0.9")]
        momentum: f64,

        /// Number of passes through the training data
        #[structopt(short, long, default_value = "50")]
        num_epochs: usize,

        /// Number of training examples to use
        #[structopt(short, long, default_value = "60000")]
        training_examples: usize,
    },
    /// Runs a variety of network configurations.
    /// Results are saved to data/output.json.
    All {
        /// Number of passes through the training data
        #[structopt(short, long, default_value = "50")]
        num_epochs: usize,
    },
}

fn main() -> Result<()> {
    let test_results = match Opt::from_args() {
        Opt::Custom {
            hidden_dimension,
            weight_range,
            learning_rate,
            momentum,
            num_epochs,
            training_examples,
        } => train_and_run(
            hidden_dimension,
            weight_range,
            learning_rate,
            momentum,
            num_epochs,
            training_examples,
        )?,
        Opt::All { num_epochs } => run_variety(num_epochs)?,
    };

    let mut file = std::fs::File::create("./data/output.json")?;
    let serialized = serde_json::to_string(&test_results)?;
    file.write_all(serialized.as_bytes())?;

    Ok(())
}
