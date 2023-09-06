# CR-age-related-conditions-predictions
Though not recognized, this one would have scored a bronze in Kaggle contest https://www.kaggle.com/competitions/icr-identify-age-related-conditions

ICR Age Related Conditions Prediction of Class 1 (has one or more conditions, or Class 0 does not have conditions)
Author, Thomas Gamet, released under the Apache 2.0 license

Turned off SMOTE
Now using MLP, TabPFN and LGB with 0.18 as the base public score I have added a TabPFN trained/tuned to detect the errors in the training and plan to apply its tunning to the public (and private) runs to detect possible samples that may be errors. For all those that are not possible errors it will boost the confidence as long as the tunning run on the training data's threshold for values that help to tune are met.

XGB was the least performant network and it has been removed.

Greeks were used, the Gamma column, to try and tune XGB. Greeks' Gamma column is used in SMOTE, though at this time the SMOTE code is marked down.

There are several versions of the balanced_log_loss implemented for tensor flow. One works with MLP and CNN which has dimensionality 2 for y predictions. The LGB Model and TabPFN (PyTorch) also have their own logloss implementation.

Key notes:

MLP tuning with drops of weights causes division by 0 and out of bounds log calculations, just don't bother.
Batch sizes can be decreased to about 101, below that is still a problem for the train.csv set (attempting greeks categoricals only works with batch of 617) 2.1. If a batch is missing all of one or the other you'll get a 0 in N0 or N1 which will cause problems. 2.2. Stratefied batches are included (class1 is mixed into the class0 rows).
The balanced_log_loss was updated to follow guidance from the Contest page. 3.1. It was further updated on 6/24/2023 to work for N>=2 categories.
LGB Model seems to outperform the MLP with the same set of features.
A voter based on taking the mean is used
The voted output has its confidence of results boosted 6.1. First, a TabPFN is trained to detect possible training errors 6.2. Second, thresholds for tuning confidence is determined experimentally. Any confidence above (or below) the class 0 and class 1 threshol respectively is converted to full confidence (likely not a false positive or false negative, and if also above the experimentally tested threshold (+- 0.02 for a safety net) then make a strong assertion. 6.3. The above failed leading me to conclude that the attempt to classify possible failures did not generalize to the public data set.
Modify the voter to boost confidence of only Class 1 predictions when it is at least 0.75 (the voter is producing 0.69 for class 0 when it has no real reason to be sure of the right probability like in the test submissions, and going with a small saftey margin for boostingto 0.999 we'ere using 0.75 to rule out class 0 when predicting class 1). Also, experimental evidence suggests that class 0 predictions is above .9 for many cases, and the one misclassification was close to .5, however the cost of being wrong with a class 1 (predicting a 0) is 5 times higher so I'm setting the boosting threshold for class 1 to > 0.90 (mistakes will cost 3 times more boosted, but correct answers will only help by 0.05 - it would take ~60 correct boosts to balance one mistaken boost). Boost to .999 as boost to 1 punishes 5 times harder (log 1e-3 versus log 1e-15 is 3 versus 15).

Plan:

See which model shows best initial promise - so far MLP (two layers, recommending between 90 and 128 nodes - currently running with 128 in the 1st layer, and 64 in the second layer).
See if means and stddev do better than 0.0 in filling NaN values in features (done and yes).
Can greeks.csv help do a better job of feature preparation (maybe)?
SMOTE is non-trivial, using impbalance over_sample module to generate synthetic numeric features, then tuning validation data selection, epochs, and features. SMOTE did not work out, scores became 2x or more times worse.
Observed that the Greek's data for Gamma maps to Class 0 for Gamma categories M and N, and to class 1 for all other 6 categories. 5.1. First pass to see if Gamma can provide a corrective prediction failed. 5.2. I am considering alternative approaches to Classifying Gamma (until its classification outperforms the train.csv set it is likely not yet suitable for a voter role).
Are the features causing any confusion - do we have too many (maybe, we are trying 14 features based on MIscore>0.04 (based on analysis of 9 with MIscore>0.05 and 19 with MIscore>0.03 not being significantly better)). 6.1. Possibly over tuning on 6/3/2023 - added highest 38 MI scores less 'AH' which caused log loss to increase (become worse) as features from 33 to 38 were added. 6.2. The best score with the MLP defined below was with 34 features on submission 18, but unfortunately there was some randomness at the time so I can now only hard code in the values.
Do we need stability - make an ensemble work for just MLP (running 5 times, but instead of averaging I'm going to test if the ends or middle are best).
Tune MLP. 8.1. Tracking this in the MLP declaration and new MI scoring code section 8.2. Competion is focused on Public score which is with the real validation set 8.3. Must use all training data to optimize Public score and graph its success, however for stopping conditions a good split was determined for 0.05 testing data. 8.4. Tune the model after finding cross over of training loss against public score 8.5. A voter for the 5 selected models (starting at basemodel) was written and the hope is that best probability can be chosen from the 3 to 5 agreeing votes based upon the size of the majority vote (unanimous uses the highest probability, 4 of 5 majority vote uses the second highest probability, and a simple vote uses the middle (also second of 3 of 5) probabilities.) 8.5.1. It looks like the middle 5 models produce about the same class output with slightly different probabilities (voting for the largest for each class seems to help). This voting technique does not appear to work when mixing with LGB model output that produces different class selections per row.
Try to tune CNN also - notes left elsewhere, it under performed MLP
Try to tune LSTM also - the long short term memory does not seem to help - and the scores are less favorable than MLP
See if an ensemble of the two best models helps. Yes, we are voting on an ensemble of 5.
Added LGB as a second model and aiming to use means of the predictions from both models to see if the MLP .20 and LGB .16 do better together or average between them (the training data test for accuracy suggests accuracy will improve) 12.1 First try with taking the mean of MLP/LGB outputs produced 0.16 (no change) 12.2 Second try voting with the max/min for each class failed wiht 0.27 as the score 12.3 Going to try model 18's 34 features as it produced an MLP that scored 0.17 - it produced 0.15 as the public score. The index was a bit off, so I'm rerunning it and the score stayed 0.15. One more run with twice the bags and twice the folds and updated parameters is being tried (merged voter suggests it will be better, let's see if it was actually overfit), then on to 12.4 12.3.1 It finally occurred to me to use greeks data to stratify Class_1 resampling for SMOTE so A, B, C (or A/B,E/F,G/H) are not mixed up when creating synthetic data. 12.4 Gut feeling: Either at XGB, or try slightly different features, or use LGB or XGB to classify on greeks Gamma to see if we can predict M or N for class 0 and the rest for class 1. 12.4.1. XGB altered to classify using softmax on the Gamma column of the Greeks data
I see notebooks using "from tabpfn import TabPFNClassifier" which looks like a worth investigation
