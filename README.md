# ICR-age-related-conditions-predictions
Though not recognized, this one would have scored a bronze in Kaggle contest https://www.kaggle.com/competitions/icr-identify-age-related-conditions

Lesson Learned, this had a private scored of 0.41 between Sliver to Bronze score ranges, if I had just let it pick the best two scores.¶
TRY old school RFC and XGBoost on Gamma to see if it outperforms the current MLP on training CSV (which has a .2 ish, once scored .17).
ICR Age Related Conditions Prediction of Class 1 (has one or more conditions, or Class 0 does not have conditions)
Author, Thomas Gamet, released under the Apache 2.0 license
Not yet using greeks and currently filling in Null and NaN with standard deviation and mean based normalization. Three models were experimented with:

Multi Layer Perceptron - seems the most promising at 128 and 64 neuron layers
Convolutional Neural Network - several attempts made, underperformed MLP
Long Short Term Memory Neural Network - underperformed MLP
There are two versions of the balanced_log_loss implemented for tensor flow. One works with MLP and CNN which has dimensionality 2 for y predictions. The other is for LSMT which uses dimensionality 3 for y predictions.

Key notes:

Tuning with drops of weights causes division by 0 and out of bounds log calculations, just don't bother.
All batch sizes are set to all training data so that we never run without having both class 1 and class 0. 2.1. If a batch is missing all of one or the other you'll get a 0 in N0 or N1 which will cause problems. 2.2. I have not experimented with creating my own stratified batches.
The balanced_log_loss was updated to follow guidance from the Contest page. 3.1. It was further updated on 6/24/2023 to work for N>=2 categories.
Plan:

See which model shows best initial promise - so far MLP (two layers, recommending between 90 and 128 nodes - currently running with 128 in the 1st layer, and 64 in the second layer).
See if means and stddev do better than 0.0 in filling NaN values in features (done and yes).
Can greeks.csv help do a better job of feature preparation (maybe)?
SMOTE is non-trivial, using impbalance over_sample module to generate synthetic numeric features, then tuning validation data selection, epochs, and features. SMOTE did not work out, scores became 2x or more times worse.
Observed that the Greek's data for Gamma maps to Class 0 for Gamma categories M and N, and to class 1 for all other 6 categories. 5.1. First pass to see if Gamma can provide a corrective prediction failed. 5.2. I am considering alternative approaches to Classifying Gamma (until its classification outperforms the train.csv set it is likely not yet suitable for a voter role).
Are the features causing any confusion - do we have too many (maybe, we are trying 14 features based on MIscore>0.04 (based on analysis of 9 with MIscore>0.05 and 19 with MIscore>0.03 not being significantly better)). 6.1. Possibly over tuning on 6/3/2023 - added highest 38 MI scores less 'AH' which caused log loss to increase (become worse) as features from 33 to 38 were added. 6.2. May search the first 33 features for any which when removed improve the internal balanced log loss???
Do we need stability - make an ensemble work for just MLP (running 5 times, but instead of averaging I'm going to test if the ends or middle are best).
Tune MLP. 8.1. Tracking this in the MLP declaration and new MI scoring code section 8.2. Competion is focused on Public score which is with the real validation set 8.3. Must use all training data to optimize Public score and graph its success, however for stopping conditions a good split was determined for 0.05 testing data. 8.4. Tune the model after finding cross over of training loss against public score 8.5. A voter for the 5 selected models (starting at basemodel) was written and the hope is that best probability can be chosen from the 3 to 5 agreeing votes based upon the size of the majority vote (unanimous uses the highest probability, 4 of 5 majority vote uses the second highest probability, and a simple vote uses the middle (also second of 3 of 5) probabilities.)
Try to tune CNN also - notes left elsewhere, it under performed MLP
Try to tune LSTM also - the long short term memory does not seem to help - and the scores are less favorable than MLP
See if an ensemble of the two best models helps. Yes, we are voting on an ensemble of 5.
