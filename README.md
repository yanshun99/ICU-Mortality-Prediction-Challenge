Feature Extraction and Preprocessing
In my project, I implemented an advanced feature extraction method to capture a more comprehensive picture of 
each patient’s condition. For static variables (such as age, gender, etc.), I treated numeric and categorical 
variables differently. For numeric features, I stored both the original value and, when the value was positive, its 
log transform to help mitigate skewness (following suggestions from Ng, 2004). For categorical variables—most 
notably, ICUType—I applied one-hot encoding. Based on the PhysioNet Challenge 2012 guidelines, I created 
four binary features corresponding to the four known ICU types: "Med-Surg", "Cardiac", "Neuro", and 
"Surgical". This approach prevents the model from imposing an arbitrary ordinal relationship on these 
categories.
For time-series variables (recorded over the first 48 hours after ICU admission), I extracted several statistics to 
capture different aspects of a patient’s stay. Specifically, I computed: • Full 48-hour statistics: mean, standard 
deviation, minimum, maximum, median, interquartile range (IQR), and count. • Segmented 24-hour statistics: I 
split the 48-hour window into the first 24 hours and the last 24 hours, computed the mean for each segment, and 
derived a “change” feature defined as the difference between the last 24-hour mean and the first 24-hour mean. 
This change feature is intended to capture any deterioration in the patient’s condition. • Recent 12-hour statistics: 
To further capture the dynamics in the patient’s most recent data, I also computed the mean and standard 
deviation over the last 12 hours.
If only a subset of the data is used (via a use_subset flag), then I retain only the most recent 12-hour features. 
This rich set of features allows my classifier to consider both overall trends and short-term changes in patient 
measurements.
Before model training, I imputed missing values using the median, which is more robust to outliers than the 
mean. In addition, for each feature I appended a missing indicator so that the model is aware of which values 
were originally missing. Finally, I normalized the feature matrix using MinMaxScaler, which I found to produce 
the best empirical results.
Choice of Algorithm and Hyperparameter Tuning
For classification, I selected logistic regression because it is particularly well-suited for binary classification and 
offers interpretable parameters. Logistic regression is computationally efficient and provides a clear decision 
boundary, which is valuable for understanding the model’s predictions.
I performed hyperparameter tuning using my custom function select_param_logreg, which iterates over a 
candidate range of the regularization parameter C (in my case, a geometric sequence from 0.01 to 100 with 5 
candidate values) and the type of regularization (L1 vs. L2). I optimized the model based on cross-validation 
performance using F1-score as the metric. In my cross-validation experiments on a split of the challenge training 
data (approximately 10,000 samples), the best hyperparameters were found to be C = 100.0 and penalty = "l2".
In addition to tuning C and the penalty, I further fine-tuned the class weights. Since my validation set indicated 
that the positive class was underrepresented, I computed balanced class weights using sklearn’s 
compute_class_weight and then manually tested several candidate positive weights (while keeping the negative 
class weight fixed at 1). My experiments revealed that a positive class weight of 5 (i.e. class weights = {-1: 1, 1: 
5}) produced the best F1-score on the validation set.
Decision Threshold Optimization
Logistic regression uses a default decision threshold of 0 (or 0.5 when working with probabilities) to generate 
binary predictions. However, to further optimize the F1-score, I automatically searched for the optimal 
threshold on the validation set. By examining 50 candidate thresholds between the minimum and maximum 
decision scores, I identified an optimal threshold of 0.3969, which maximized the F1-score on my validation 
set.
Final Model and Confusion Matrix
After tuning, I trained my final model on the full challenge training data using the following selected 
parameters: • C = 100.0 and penalty = "l2" (as determined by my hyperparameter selection) • Class weights: 
{-1: 1, 1: 5} • Optimal decision threshold: 0.3969.
I evaluated the final model’s performance on the validation set using my custom performance function. The 
final model achieved: • F1-score: 0.5391 • AUROC: 0.8684.
The 2×2 confusion matrix on the challenge training data was:<img width="402" height="54" alt="Screenshot 2025-09-07 at 2 49 39 PM" src="https://github.com/user-attachments/assets/37cc1c94-8c03-484d-a93a-7a635f17d523" />

 This confusion matrix indicates that the model correctly classified 6784 negative cases and 1195 positive 
cases, while misclassifying 1676 negatives as positives and 345 positives as negatives. The high AUROC 
demonstrates that the model ranks cases well, and the improved F1-score reflects the careful balancing of 
precision and recall achieved through fine-tuning of both class weights and the decision threshold.
Conclusion
My approach combines extensive feature engineering, hyperparameter tuning, and decision threshold 
optimization to achieve a well-balanced model that optimizes both AUROC and F1-score. The detailed 
confusion matrix and performance metrics provide evidence of the effectiveness of these methods. In 
particular, the chosen model (with C = 100.0, L2 penalty, class weights {-1: 1, 1: 5}, and an optimal threshold 
of 0.3969) produced an AUROC of 0.8684 and an F1-score of 0.5391, demonstrating a significant 
improvement over simpler baseline models.
