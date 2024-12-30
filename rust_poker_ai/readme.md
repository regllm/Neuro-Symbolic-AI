# Rust Poker AI

This repository contains a clustering section of the Python library "Poker AI."

## Build

To build the package in an optimized mode, run this command:
```bash
cargo build --release
```

## Run

### Cluster

To start the clustering process, run this command:
```bash
cargo run --release cluster
```

If you want to simply test with the short deck with only 4 ranks available,
add "--short" option when you run:
```bash
cargo run --release -- cluster --short
```

### Run Lookup Table Server

To start the lookup table server, run this command:
```bash
cargo run --release -- table
```

You can specify the path to read the data files for the lookup table.
If you do not specify, the default path is `./output`. 
```bash
cargo run --release -- table --input ./custom_path
```

You can also specify the host and port for the server.
```bash
cargo run --release -- table --host 127.0.0.1 --port 8989
```

When you run *Poker AI train*, You can specify the lut_path as a URI start with `lut://`.

For example, with the server executed with the arguments above,
you may want to start training the model with this command:
```bash
poker_ai train start --lut_path lut://127.0.0.1:8989
```
