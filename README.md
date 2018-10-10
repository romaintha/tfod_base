# TensorFlow Object Detection API base structure
This repository is meant to be used as a starting point for object detection tasks involving
the TensorFlow object detection API.

## Getting Started
Before to be able to use the TensorFlow object detection API, they are several steps that needs 
to be followed.

### Conda environment
To installed an already prepared conda environment, run:

```conda create --name myenv --file spec-file.txt```

### TensorFlow Models
Clone [this GitHub repository](https://github.com/tensorflow/models) somewhere in your ```home``` folder.

You also need to update your PYTHONPATH variable to include the TensorFlow object detection 
API import. 

```
export PYTHONPATH=$PYTHONPATH:/folder_where_you_cloned_the_above_repo/models/research:/folder_where_you_cloned_the_above_repo/models/research/slim
```

You will eventually need to run it every time you open a new terminal. 

Finally the TensorFlow object detection API framework needs to be compiled. Go into the ```models/research```
folder and run :

```
protoc object_detection/protos/*.proto --python_out=.
```

To test the framework is correcly installed simply run from the ```models/research``` folder:

```
python object_detection/builders/model_builder_test.py
```

