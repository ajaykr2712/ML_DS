# Machine Learning Project Checklist
This checklist can guide you through your Machine Learning projects. There are
eight main steps:
1. Frame the problem and look at the big picture.
2. Get the data.
3. Explore the data to gain insights.
4. Prepare the data to better expose the underlying data patterns to Machine Learn‐
ing algorithms.
5. Explore many different models and short-list the best ones.
6. Fine-tune your models and combine them into a great solution.
7. Present your solution.
8. Launch, monitor, and maintain your system.
Obviously, you should feel free to adapt this checklist to your needs.


## Frame the Problem and Look at the Big Picture
 Define the objective in business terms.
How will your solution be used?
What are the current solutions/workarounds (if any)?
How should you frame this problem (supervised/unsupervised, online/offline,etc.)?
How should performance be measured?
Is the performance measure aligned with the business objective?
What would be the minimum performance needed to reach the business objec‐tive?
What are comparable problems? Can you reuse experience or tools?
Is human expertise available?
How would you solve the problem manually?
List the assumptions you (or others) have made so far.
 Verify assumptions if possible.

 ## Get the Data

 Note: automate as much as possible so you can easily get fresh data.
 List the data you need and how much you need.
Find and document where you can get that data.
Check how much space it will take.
Check legal obligations, and get authorization if necessary.
 Get access authorizations.
Create a workspace (with enough storage space).
 Get the data.
 Convert the data to a format you can easily manipulate (without changing the data itself).
 Ensure sensitive information is deleted or protected (e.g., anonymized).
Check the size and type of data (time series, sample, geographical, etc.).
Sample a test set, put it aside, and never look at it (no data snooping!).

## Explore the Data
note: try to gain insights from the Experts of these steps


Create a copy of the data for exploration (sampling it down to a manageable size if necessary).
Create a Jupyter notebook to keep a record of your data exploration.
Study each attribute and its characteristics:
• Name
• Type (categorical, int/float, bounded/unbounded, text, structured, etc.)
• % of missing values
• Noisiness and type of noise (stochastic, outliers, rounding errors, etc.)
• Possibly useful for the task?
• Type of distribution (Gaussian, uniform, logarithmic, etc.)
For supervised learning tasks, identify the target attribute(s).
 Visualize the data.
Study the correlations between attributes.
Study how you would solve the problem manually.
Identify the promising transformations you may want to apply.
Identify extra data that would be useful (go back to “Get the Data” on page 498).
Document what you have learned.

## Prepare the Data
Notes: 
• Work on copies of the data (keep the original dataset intact).
• Write functions for all data transformations you apply, for five reasons:
— So you can easily prepare the data the next time you get a fresh dataset
— So you can apply these transformations in future projects
— To clean and prepare the test set
— To clean and prepare new data instances once your solution is live
— To make it easy to treat your preparation choices as hyperparameters
1. Data cleaning:
• Fix or remove outliers (optional).
• Fill in missing values (e.g., with zero, mean, median…) or drop their rows (or
columns).
2. Feature selection (optional):
• Drop the attributes that provide no useful information for the task.
3. Feature engineering, where appropriate:
• Discretize continuous features.
• Decompose features (e.g., categorical, date/time, etc.).
• Add promising transformations of features (e.g., log(x), sqrt(x), x^2, etc.).
• Aggregate features into promising new features.
Feature scaling: standardize or normalize features.

## Short-List Promising Models

Notes:
• If the data is huge, you may want to sample smaller training sets so you can train
many different models in a reasonable time (be aware that this penalizes complex
models such as large neural nets or Random Forests).
• Once again, try to automate these steps as much as possible.
1. Train many quick and dirty models from different categories (e.g., linear, naive
Bayes, SVM, Random Forests, neural net, etc.) using standard parameters.
2. Measure and compare their performance.
• For each model, use N-fold cross-validation and compute the mean and stan‐
dard deviation of the performance measure on the N folds.
3. Analyze the most significant variables for each algorithm.
4. Analyze the types of errors the models make.
• What data would a human have used to avoid these errors?
5. 6. Have a quick round of feature selection and engineering.
Have one or two more quick iterations of the five previous steps.
7. Short-list the top three to five most promising models, preferring models that
make different types of errors.


## Fine-Tune the System
Notes:
• Y ou will want to use as much data as possible for this step, especially as you move
toward the end of fine-tuning.
• As always automate what you can.

1. Fine-tune the hyperparameters using cross-validation.
• Treat your data transformation choices as hyperparameters, especially when
you are not sure about them (e.g., should I replace missing values with zero or
with the median value? Or just drop the rows?).
• Unless there are very few hyperparameter values to explore, prefer random
search over grid search. If training is very long, you may prefer a Bayesian
optimization approach (e.g., using Gaussian process priors, as described by
Jasper Snoek, Hugo Larochelle, and Ryan Adams).1
2. Try Ensemble methods. Combining your best models will often perform better
than running them individually.
3. Once you are confident about your final model, measure its performance on the
test set to estimate the generalization error.