# StableMiss & StableMiss+

Official code for both StableMiss and StableMiss+.

- StableMiss: [Prediction with Incomplete Data under Agnostic Mask Distribution Shift](https://www.ijcai.org/proceedings/2023/0525.pdf) (IJCAI 2023)
- StableMiss+: [StableMiss+: Prediction with Incomplete Data under Agnostic Mask Distribution Shift](https://ieeexplore.ieee.org/document/10446980) (ICASSP 2024)

StableMiss+ is an improved version of StableMiss. The difference lies in the way of sample reweighting.

## Configuration

Configure the settings of datasets, models, and optimization process in `reweight_[stablemiss/stablemiss+].py` and `train.py`.

## Training

- Step 1: Conduct sample reweighting and obtain the weights for decorrelation.

```Python
python reweight_[stablemiss/stablemiss+].py
```

- Step 2: Train the prediction framework based on the weights from step 1.

```Python
python train.py
```
