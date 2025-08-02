## **Table of Contents**

| Preface<br>.<br><b>XV</b> |                                               |    |
|---------------------------|-----------------------------------------------|----|
| Part I.                   | The Fundamentals of Machine Learning          |    |
|                           | 1. The Machine Learning Landscape             |    |
|                           | What Is Machine Learning?                     | 4  |
|                           | Why Use Machine Learning?                     | 5  |
|                           | <b>Examples of Applications</b>               | 8  |
|                           | Types of Machine Learning Systems             | 9  |
|                           | <b>Training Supervision</b>                   | 10 |
|                           | Batch Versus Online Learning                  | 17 |
|                           | Instance-Based Versus Model-Based Learning    | 21 |
|                           | Main Challenges of Machine Learning           | 27 |
|                           | <b>Insufficient Quantity of Training Data</b> | 27 |
|                           | Nonrepresentative Training Data               | 28 |
|                           | Poor-Quality Data                             | 30 |
|                           | <b>Irrelevant Features</b>                    | 30 |
|                           | Overfitting the Training Data                 | 30 |
|                           | Underfitting the Training Data                | 33 |
|                           | <b>Stepping Back</b>                          | 33 |
|                           | Testing and Validating                        | 34 |
|                           | Hyperparameter Tuning and Model Selection     | 34 |
|                           | Data Mismatch                                 | 35 |
|                           | Exercises                                     | 37 |

{5}------------------------------------------------

| 2. End-to-End Machine Learning Project.          | 39  |
|--------------------------------------------------|-----|
| Working with Real Data                           | 39  |
| Look at the Big Picture                          | 41  |
| Frame the Problem                                | 41  |
| Select a Performance Measure                     | 43  |
| Check the Assumptions                            | 46  |
| Get the Data                                     | 46  |
| Running the Code Examples Using Google Colab     | 46  |
| Saving Your Code Changes and Your Data           | 48  |
| The Power and Danger of Interactivity            | 49  |
| Book Code Versus Notebook Code                   | 50  |
| Download the Data                                | 50  |
| Take a Quick Look at the Data Structure          | 51  |
| Create a Test Set                                | 55  |
| Explore and Visualize the Data to Gain Insights  | 60  |
| Visualizing Geographical Data                    | 61  |
| Look for Correlations                            | 63  |
| Experiment with Attribute Combinations           | 66  |
| Prepare the Data for Machine Learning Algorithms | 67  |
| Clean the Data                                   | 68  |
| Handling Text and Categorical Attributes         | 71  |
| Feature Scaling and Transformation               | 75  |
| <b>Custom Transformers</b>                       | 79  |
| <b>Transformation Pipelines</b>                  | 83  |
| Select and Train a Model                         | 88  |
| Train and Evaluate on the Training Set           | 88  |
| Better Evaluation Using Cross-Validation         | 89  |
| Fine-Tune Your Model                             | 91  |
| Grid Search                                      | 91  |
| Randomized Search                                | 93  |
| Ensemble Methods                                 | 95  |
| Analyzing the Best Models and Their Errors       | 95  |
| Evaluate Your System on the Test Set             | 96  |
| Launch, Monitor, and Maintain Your System        | 97  |
| Try It Out!                                      | 100 |
| Exercises                                        | 101 |
| 3. Classification.                               | 103 |
| <b>MNIST</b>                                     | 103 |
| Training a Binary Classifier                     | 106 |
| Performance Measures                             | 107 |

{6}------------------------------------------------

| Measuring Accuracy Using Cross-Validation | 107 |
|-------------------------------------------|-----|
| <b>Confusion Matrices</b>                 | 108 |
| Precision and Recall                      | 110 |
| The Precision/Recall Trade-off            | 111 |
| The ROC Curve                             | 115 |
| Multiclass Classification                 | 119 |
| Error Analysis                            | 122 |
| Multilabel Classification                 | 125 |
| Multioutput Classification                | 127 |
| Exercises                                 | 129 |
| 4. Training Models.                       | 131 |
| Linear Regression                         | 132 |
| The Normal Equation                       | 134 |
| <b>Computational Complexity</b>           | 137 |
| <b>Gradient Descent</b>                   | 138 |
| <b>Batch Gradient Descent</b>             | 142 |
| <b>Stochastic Gradient Descent</b>        | 145 |
| Mini-Batch Gradient Descent               | 148 |
| Polynomial Regression                     | 149 |
| Learning Curves                           | 151 |
| Regularized Linear Models                 | 155 |
| Ridge Regression                          | 156 |
| Lasso Regression                          | 158 |
| <b>Elastic Net Regression</b>             | 161 |
| Early Stopping                            | 162 |
| Logistic Regression                       | 164 |
| <b>Estimating Probabilities</b>           | 164 |
| Training and Cost Function                | 165 |
| <b>Decision Boundaries</b>                | 167 |
| Softmax Regression                        | 170 |
| Exercises                                 | 173 |
| 5. Support Vector Machines.               | 175 |
| Linear SVM Classification                 | 175 |
| Soft Margin Classification                | 176 |
| Nonlinear SVM Classification              | 178 |
| Polynomial Kernel                         | 180 |
| <b>Similarity Features</b>                | 181 |
| Gaussian RBF Kernel                       | 181 |
| SVM Classes and Computational Complexity  | 183 |

{7}------------------------------------------------

| <b>SVM Regression</b>                        | 184        |
|----------------------------------------------|------------|
| Under the Hood of Linear SVM Classifiers     | 186        |
| The Dual Problem                             | 189        |
| <b>Kernelized SVMs</b>                       | 190        |
| Exercises                                    | 193        |
| 6. Decision Trees<br>.                       | 195        |
| Training and Visualizing a Decision Tree     | 195        |
| <b>Making Predictions</b>                    | 197        |
| <b>Estimating Class Probabilities</b>        | 199        |
| The CART Training Algorithm                  | 199        |
| Computational Complexity                     | 200        |
| Gini Impurity or Entropy?                    | 201        |
| Regularization Hyperparameters               | 201        |
| Regression                                   | 204        |
| Sensitivity to Axis Orientation              | 206        |
| Decision Trees Have a High Variance          | 207        |
| Exercises                                    | 208        |
| 7. Ensemble Learning and Random Forests.     | 211        |
| <b>Voting Classifiers</b>                    | 212        |
| Bagging and Pasting                          | 215        |
| Bagging and Pasting in Scikit-Learn          | 217        |
| Out-of-Bag Evaluation                        | 218        |
| Random Patches and Random Subspaces          | 219        |
| Random Forests                               | 220        |
| Extra-Trees                                  | 220        |
| Feature Importance                           | 221        |
| Boosting                                     | 222        |
| AdaBoost                                     | 222        |
| <b>Gradient Boosting</b>                     | 226        |
| Histogram-Based Gradient Boosting            | 230        |
| Stacking<br>Exercises                        | 232<br>235 |
|                                              |            |
| 8. Dimensionality Reduction.                 | 237        |
| The Curse of Dimensionality                  | 238        |
| Main Approaches for Dimensionality Reduction | 239        |
| Projection                                   | 239        |
| Manifold Learning                            | 241        |
| <b>PCA</b>                                   | 243        |

{8}------------------------------------------------

| Preserving the Variance                            | 243 |
|----------------------------------------------------|-----|
| Principal Components                               | 244 |
| Projecting Down to d Dimensions                    | 245 |
| Using Scikit-Learn                                 | 246 |
| <b>Explained Variance Ratio</b>                    | 246 |
| Choosing the Right Number of Dimensions            | 247 |
| PCA for Compression                                | 249 |
| Randomized PCA                                     | 250 |
| Incremental PCA                                    | 250 |
| Random Projection                                  | 252 |
| <b>LLE</b>                                         | 254 |
| Other Dimensionality Reduction Techniques          | 256 |
| Exercises                                          | 257 |
| 9. Unsupervised Learning Techniques.               | 259 |
| Clustering Algorithms: k-means and DBSCAN          | 260 |
|                                                    |     |
| k-means                                            | 263 |
| Limits of k-means                                  | 272 |
| Using Clustering for Image Segmentation            | 273 |
| Using Clustering for Semi-Supervised Learning      | 275 |
| <b>DBSCAN</b>                                      | 279 |
| Other Clustering Algorithms                        | 282 |
| <b>Gaussian Mixtures</b>                           | 283 |
| Using Gaussian Mixtures for Anomaly Detection      | 288 |
| Selecting the Number of Clusters                   | 289 |
| Bayesian Gaussian Mixture Models                   | 292 |
| Other Algorithms for Anomaly and Novelty Detection | 293 |
| Exercises                                          | 294 |
