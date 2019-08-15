# Cars palette detection

Python project to segment car's pallete and identify car's number. 

## Getting Started
Project is still in progress. Now it is possible to detect car's pallete position, extract 
and normilize it i.e convert pallete into top-view format. As an example one can examine cars_test.ipynb file.

### Prerequisites

In order run unet_simple.py one needs to have a GPU acceleration, otherwise the 
training process may take some time.


### Installing

* Python 3.7.3
* opencv 3.4.2
* tensorflow 1.14
* keras 2.2.4

### Project's structure:
	├── input
		├── data
			├── train
			├── test
			├── train.json
			├── val.json
		├── resized
			├── train
				├── images_512
				├── masks_512
			├── val
				├── images_512
				├── masks_512
	├── models
		├──unet_1.h5
	├── src
		├── dataPrep.py
		├── imageFunc.py
		├── unet_simple.py

* Input/Model folders are ignored due to sizes
## Deployment
[ADD]


## Authors

* **Aziz Abdraimov** - *Initial work* - [Aziko13](https://github.com/Aziko13)

