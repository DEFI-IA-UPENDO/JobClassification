# AI Challenge Repository

### Team name: France-INSA/ENSEEIHT/VALDOM-UPENDO
### Team Members: 
* GHOMSI KONGA Serge <vr> 
* KEITA Alfousseyni
* RIDA Moumni
* SANOU Désiré
* WAFFA PAGOU Brondon


The goal of the *Défi-IA* was to  assign the correct job category (among 28 categories) to a job description. 
The link to the kaggle pqge of the the competition is : https://www.kaggle.com/c/defi-ia-insa-toulouse .<br>
For this project, we tested several models including BERT, Logistic Regression, Embedding,lstm-gru-cnn-glove and SVC. 
With individual models, we didn't get the accuracy we were excepting. 
Therefore, we chosed those giving the best accuracies ( i.e Bert, SVC and lstm-gru-cnn-glove), and performed a majority voting on them. <br> 
BERT had the best accuracy among the three models, so we gave it the priority in case all the three predictions are different. <br>
This reports focuses on the principles of the three algorithms, the reasons behind our choice and the results we obtained from them.

## Achieved Results
The final accuracies we had were:
* Public score:  0.78049
* Private score: 0.78013

## Computation time?
 *TODO: add computation time here*
 
## Technical requirements
*TODO:engine used*

You should install the following packages.They are all mentionned in the requirements.txt file: <br>
*TODO: add the content of the requirements.txt.*

To install all the packages, on your command line, type: 
> pip install -r requirements.txt

### Make prediction with our model
In case you want to make new prediction using our model, type
> *TODO: complete the command line*
### Reproduce training
If you want to reproduce all the training,on your command line, type:
> *TODO:complete the command line* 

To run the training,on your command line, type:
> *TODO:complete the command line* 
