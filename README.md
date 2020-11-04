### CS5242 Kaggle Project
	
Liu Mingzhe - e0575807@u.nus.edu  
Jeremy Tan - e0573157@u.nus.edu  


#### Installation and Usage
1. Install dependencies. Note the `find-links` for pytorch, added for your convenience.  
`pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html`    

2. Save data file in  `/data` folder and extract there.   
The expected filepath is:
	```
	data/

	  nus-cs5242/

		test_image/
		train_image/
		sample_submission.csv
		train_label.csv
	```   

3. Run train script.  
This will augment the data, train the model, and output predictions in `/predictions` folder.  
`python train.py`  
On the terminal, the expected output should look like this:  
<img src="misc/expected_output.png">  
Of course, running the full 100 epochs will give a better result.

#### Expected Output of predictions
The predictions will be saved in the `predictions\<timestamp>` folder in csv.  
The model will likewise be saved in the `models\<timestamp>` folder.