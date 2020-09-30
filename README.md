## Purpose
This program is an implementation of a multilayer perceptron with one hidden layer
for classifying the MNIST data set. Program was completed for Dr. Anthony Rhodes'
CS445/545 Machine Learning class in the Fall of 2020.

## Usage
If you want to run my network you must have python3 and cargo/rust installed.

Clone the repo:
```
git clone https://github.com/aujxn/perceptron
cd perceptron
```

Download the dataset from: https://www.kaggle.com/oddrationale/mnist-in-csv and put the files
(mnist_train.csv and mnist_test.csv) into the data directory.

Create python environment and load python dependencies:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Compile network:
```
cargo build --release
```

Program has two subcommands. Usage info:
```
cargo run --release -- custom --help    // Run a custom network config
cargo run --release -- all --help       // Run a variety of configs
```

Running the program will overwrite the output.json file with information about the network.
If run with the "all" subcommand, the python program can be used to create some interactive
graphs:
```
python main.py
```

This wont work for the custom network, though.

## License
MIT - see LICENSE file
