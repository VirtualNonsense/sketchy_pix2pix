# sketchy pix2pix
I'm currently trying to adapt the pix2pix network described [here](https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/) to work with the [sketchydb dataset](https://sketchy.eye.gatech.edu) (i'm using the download link that is provided within this [repository](https://github.com/CDOTAD/SketchyDatabase) to get the data).

# Setup
For your convinience i wrote a small xtask program to setup the environment.

Download the dataset
```bash
cargo xtask download-sketchy-db
``` 

Extract the dataset

```bash
cargo xtask extract-sketchy-db
``` 

Install [rerun](https://rerun.io) to see intermediate results and some nice graphs.
```
cargo binstall rerun-cli
```

# training
- Start rerun
- Run the training
    ```bash
    cargo run --package sketchy_pix2pix --bin train_pix2pix --release
    ``` 

## troubleshooting
You may encounter this [error message](https://github.com/rerun-io/rerun/issues/9159) due to an issue in one of the dependencies of rerun.
For me it worked to exclude xtask from the [workspace](./cargo.toml#L5) and run ```cargo update chrono --precise 0.4.34```
