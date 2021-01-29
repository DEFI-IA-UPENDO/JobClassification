# AI Challenge Repository

### Team name: France-INSA/ENSEEIHT/VALDOM-UPENDO
### Team Members: 
* GHOMSI KONGA Serge <vr> 
* KEITA Alfousseyni
* RIDA Moumni
* SANOU Désiré
* WAFFA PAGOU Brondon


The goal of the *Défi-IA* was to  assign the correct job category (among 28 categories) to a job description. 
The link to the kaggle page of the the competition is : https://www.kaggle.com/c/defi-ia-insa-toulouse .<br>
For this project, we tested several models including BERT, Logistic Regression, Embedding,lstm-gru-cnn-glove and SVC. 
With individual models, we didn't get the accuracy we were excepting. 
Therefore, we chose those giving the best accuracies ( i.e Bert, SVC and lstm-gru-cnn-glove), and performed a majority voting on them. <br> 
BERT had the best accuracy among the three models, so we gave it the priority in case all the three predictions are different. <br>
This reports focuses on the principles of the three algorithms, the reasons behind our choice and the results we obtained from them.

## Achieved Results
The final accuracies we had were:
* Public score:  0.78049
* Private score: 0.78013

## Computation time and Engine?
**BERT** 
* Colab GPU: ~ 2h
* Local (using GPU): ~ 5.2h 

**SVC**
* Local (using CPU): 3 ~ 5min

**Bi-GRU-LSTM-CNN-Glove**
* Kaggle: ~ 12min  (GPU = 16GB, CPU = 13GB)
* Local : ~ 12h  (Intel ® Core(™) i7-6500U CPU @ 2.50GHz 2.59GHz, Ram : 16Go, System : 64 bits)
 
## Technical requirements

You should install the following packages. <br>
They are all mentioned in the requirements.txt file. <br>
To install all the packages, on your command line, type: 
> pip install -r requirements.txt

### Make prediction with our model
In case you want to make new prediction using our model, type.
> python main.py 

and follow the instructions
### Reproduce training
If you want to reproduce all the training,on your command line, type:
> python main.py 

and follow the instructions :

### Docker :

<p>In the current directory of the application, execute the  following commands : </p> <br>

```
docker build -t job_classification .
docker run -i job_classification
```

### Conda :

 ```
 conda create -n JobClassification python=3.7.9
conda activate JobClassification
pip install -r requirements.txt 
>python app/main.py  
```


