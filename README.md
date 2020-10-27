### CS5242 Kaggle Project
	
Liu Mingzhe - e0575807@u.nus.edu  
Jeremy Tan - e0573157@u.nus.edu  


#### Installation and Usage
`pip install -r requirements.txt`
`python train.py`

#### Data
The data is to be downloaded from kaggle and placed in the `/data` folder and extracted there.  
The expected filepath is:
```
data/
	nus-cs5242/
		test_image/
		train_image/
		sample_submission.csv
		train_label.csv
```

#### TODO:
1. Data augmentation for imbalanced set
2. Inference script in `train.py`