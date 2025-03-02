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

# training
Run the training
```bash
cargo run --bin sketchy_pix2pix
``` 

