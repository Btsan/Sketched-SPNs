<meta name="robots" content="noindex">

## Requirements

Install python dependencies

```
pip3 install -r requirements.txt
```

Also, install the KWiseHash package from [https://github.com/mikeheddes/fast-multi-join-sketch](https://github.com/mikeheddes/fast-multi-join-sketch/tree/c66c1486679f6eea4f78d703550050a83153360b/kwisehash)

Run the following script to download the Stats and IMDb datasets used to train SPNs

```
bash datasets.sh
```

## Usage

Launch training and evaluation with [sketched_spn.py](/sketched_spn.py).
