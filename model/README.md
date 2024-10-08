
# Classification Model
Custom CNN model to be used in OCR system. I utilized dataset from [bhmsds's Repo](https://github.com/wblachowski/bhmsds). This model achieved an accuracy of `%98.2` on the test set of the dataset (splitted with fixed seed for reproducibility).

![Model Architecture](https://github.com/user-attachments/assets/745d3c8b-53ec-4265-acaa-36e0ccff6fc9)

## Installation
you can install the packages needed with `conda` by using [`requirements.txt`](https://github.com/Azimi2kht/camculator/blob/master/requirements.txt).

```
conda create --name camculator --file requirements.txt
```

## Documentation
You can customize the model settings in [`config.yaml`](https://github.com/Azimi2kht/camculator/blob/master/model/src/config.yaml) file.

### Adding a Custom Part
Any customizations can be added by changing the corresponding file in the project. for example if you want to add and use another loss function change the [`loss.py`](https://github.com/Azimi2kht/camculator/blob/master/model/src/model/loss.py) file, then change the name in [`config.yaml`](https://github.com/Azimi2kht/camculator/blob/master/model/src/config.yaml). Or if you want to add another model change the [`models.py`](https://github.com/Azimi2kht/camculator/blob/master/model/src/model/models.py) file and add the model class, then change the field in [`config.yaml`](https://github.com/Azimi2kht/camculator/blob/master/model/src/config.yaml).



