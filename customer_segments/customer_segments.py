
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Unsupervised Learning
# ## Project: Creating Customer Segments

# Welcome to the third project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `'TODO'` statement. Please be sure to read the instructions carefully!
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ## Getting Started
# 
# In this project, you will analyze a dataset containing data on various customers' annual spending amounts (reported in *monetary units*) of diverse product categories for internal structure. One goal of this project is to best describe the variation in the different types of customers that a wholesale distributor interacts with. Doing so would equip the distributor with insight into how to best structure their delivery service to meet the needs of each customer.
# 
# The dataset for this project can be found on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers). For the purposes of this project, the features `'Channel'` and `'Region'` will be excluded in the analysis — with focus instead on the six product categories recorded for customers.
# 
# Run the code block below to load the wholesale customers dataset, along with a few of the necessary Python libraries required for this project. You will know the dataset loaded successfully if the size of the dataset is reported.

# In[143]:


# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
get_ipython().magic(u'matplotlib inline')

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"


# ## Data Exploration
# In this section, you will begin exploring the data through visualizations and code to understand how each feature is related to the others. You will observe a statistical description of the dataset, consider the relevance of each feature, and select a few sample data points from the dataset which you will track through the course of this project.
# 
# Run the code block below to observe a statistical description of the dataset. Note that the dataset is composed of six important product categories: **'Fresh'**, **'Milk'**, **'Grocery'**, **'Frozen'**, **'Detergents_Paper'**, and **'Delicatessen'**. Consider what each category represents in terms of products you could purchase.

# In[144]:


# Display a description of the dataset
display(data.describe())


# ### Implementation: Selecting Samples
# To get a better understanding of the customers and how their data will transform through the analysis, it would be best to select a few sample data points and explore them in more detail. In the code block below, add **three** indices of your choice to the `indices` list which will represent the customers to track. It is suggested to try different sets of samples until you obtain customers that vary significantly from one another.

# In[145]:


# Select three indices of your choice you wish to sample from the dataset
indices = [30,124,307]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "Chosen samples of wholesale customers dataset:"
display(samples)


# ### Question 1
# Consider the total purchase cost of each product category and the statistical description of the dataset above for your sample customers.  
# 
# * What kind of establishment (customer) could each of the three samples you've chosen represent?
# 
# **Hint:** Examples of establishments include places like markets, cafes, delis, wholesale retailers, among many others. Avoid using names for establishments, such as saying *"McDonalds"* when describing a sample customer as a restaurant. You can use the mean values for reference to compare your samples with. The mean values are as follows:
# 
# * Fresh: 12000.2977
# * Milk: 5796.2
# * Grocery: 3071.9
# * Detergents_paper: 2881.4
# * Delicatessen: 1524.8
# 
# Knowing this, how do your samples compare? Does that help in driving your insight into what kind of establishments they might be? 
# 

# **Answer:
# 
# The first customer selected could be a fresh stall(vegetable,fruits,etc) based on their higher  average purchase costs across all product categories.
# 
# 
# The second customer chosen appears to be a SuperMarket (retailer Grocery Store) based on their higher average purchase costs than that of  delicatessen (that is meats), and lower purchase costs of Frozens, Detergent, and other restaurant(milk)items.
# 
# 
# The third customer chosen appears to be a Coffee Shop based on their higher than average purchase costs of Milk and Groceries (i.e. snacks, other ingredients), and lower than average purchase costs of  Frozens, Detergent, and Delicatessen.
# 
# **

# ### Implementation: Feature Relevance
# One interesting thought to consider is if one (or more) of the six product categories is actually relevant for understanding customer purchasing. That is to say, is it possible to determine whether customers purchasing some amount of one category of products will necessarily purchase some proportional amount of another category of products? We can make this determination quite easily by training a supervised regression learner on a subset of the data with one feature removed, and then score how well that model can predict the removed feature.
# 
# In the code block below, you will need to implement the following:
#  - Assign `new_data` a copy of the data by removing a feature of your choice using the `DataFrame.drop` function.
#  - Use `sklearn.cross_validation.train_test_split` to split the dataset into training and testing sets.
#    - Use the removed feature as your target label. Set a `test_size` of `0.25` and set a `random_state`.
#  - Import a decision tree regressor, set a `random_state`, and fit the learner to the training data.
#  - Report the prediction score of the testing set using the regressor's `score` function.

# In[146]:


from sklearn.cross_validation import train_test_split as Tts
from sklearn.tree import DecisionTreeRegressor
# Make a copy of the DataFrame, using the 'drop' function to drop the given feature
# Possible features=['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicatessen']
feature_dropped='Milk'

new_data = data.drop(feature_dropped,axis=1)
labels=data[feature_dropped]

# Split the data into training and testing sets using the given feature as the target
X_train, X_test, y_train, y_test = Tts(new_data, labels, test_size=0.25, random_state=30)

# Create a decision tree regressor and fit it to the training set
regressor = DecisionTreeRegressor(random_state=30)
regressor.fit(X_train,y_train)

# Report the score of the prediction using the testing set
score = regressor.score(X_test,y_test)

print score


# ### Question 2
# 
# * Which feature did you attempt to predict? 
# * What was the reported prediction score? 
# * Is this feature necessary for identifying customers' spending habits?
# 
# **Hint:** The coefficient of determination, `R^2`, is scored between 0 and 1, with 1 being a perfect fit. A negative `R^2` implies the model fails to fit the data. If you get a low score for a particular feature, that lends us to beleive that that feature point is hard to predict using the other features, thereby making it an important feature to consider when considering relevance.

# **Answer:
# 
# I attempted to predict the "Milk" product category. The R^2 score obtained was 0.488020874412 which is scored between 0 and 1.The score 0.0488 is at half point between 0 and 1.
# From this result we can infer that the "Milk" feature is that necessary in our dataset, and if it is removed, our model will not accurately identify customers' spending habits.This is due to losing relevant information that is not correlated to the remaining features in the dataset, therefore making it hard to predict based on those remaining feature, as proven by the resulting medium R^2 score. 
# 
# 
# **

# ### Visualize Feature Distributions
# To get a better understanding of the dataset, we can construct a scatter matrix of each of the six product features present in the data. If you found that the feature you attempted to predict above is relevant for identifying a specific customer, then the scatter matrix below may not show any correlation between that feature and the others. Conversely, if you believe that feature is not relevant for identifying a specific customer, the scatter matrix might show a correlation between that feature and another feature in the data. Run the code block below to produce a scatter matrix.

# In[147]:


# Produce a scatter matrix for each pair of features in the data
pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# ### Question 3
# * Using the scatter matrix as a reference, discuss the distribution of the dataset, specifically talk about the normality, outliers, large number of data points near 0 among others. If you need to sepearate out some of the plots individually to further accentuate your point, you may do so as well.
# * Are there any pairs of features which exhibit some degree of correlation? 
# * Does this confirm or deny your suspicions about the relevance of the feature you attempted to predict? 
# * How is the data for those features distributed?
# 
# **Hint:** Is the data normally distributed? Where do most of the data points lie? You can use [corr()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html) to get the feature correlations and then visualize them using a [heatmap](http://seaborn.pydata.org/generated/seaborn.heatmap.html)(the data that would be fed into the heatmap would be the correlation values, for eg: `data.corr()`) to gain further insight.

# **Answer:
# 
# All of the features appears to be Skewed right (positive), with the majority data points landing towards the right of the mode and a long tail left, with the exceptions of Detergent_Paper and Grocery category pairs, which has an almost symmetric (normal) distribution across the x and y axis centering on the mode.
# 
# From the scatter matrix, it can be observed that that the pair (Grocery, Detergents_Paper) seems to have the strongest correlation between the features. The pair (Grocery, Milk) also seem to exhibit some degree of correlation. This scatter matrix also confirms my initial suspicions that the "Milk" product category does not have significant correlations to the remaining features and therefore, its information is necessary to accurately predict customers' behavior to an extend.
# 
# 
# **

# ## Data Preprocessing
# In this section, you will preprocess the data to create a better representation of customers by performing a scaling on the data and detecting (and optionally removing) outliers. Preprocessing data is often times a critical step in assuring that results you obtain from your analysis are significant and meaningful.

# ### Implementation: Feature Scaling
# If data is not normally distributed, especially if the mean and median vary significantly (indicating a large skew), it is most [often appropriate](http://econbrowser.com/archives/2014/02/use-of-logarithms-in-economics) to apply a non-linear scaling — particularly for financial data. One way to achieve this scaling is by using a [Box-Cox test](http://scipy.github.io/devdocs/generated/scipy.stats.boxcox.html), which calculates the best power transformation of the data that reduces skewness. A simpler approach which can work in most cases would be applying the natural logarithm.
# 
# In the code block below, you will need to implement the following:
#  - Assign a copy of the data to `log_data` after applying logarithmic scaling. Use the `np.log` function for this.
#  - Assign a copy of the sample data to `log_samples` after applying logarithmic scaling. Again, use `np.log`.

# In[148]:


#  Scale the data using the natural logarithm
log_data = np.log(data)

#  Scale the sample data using the natural logarithm
log_samples = np.log(samples)

# Produce a scatter matrix for each pair of newly-transformed features
pd.plotting.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# ### Observation
# After applying a natural logarithm scaling to the data, the distribution of each feature should appear much more normal. For any pairs of features you may have identified earlier as being correlated, observe here whether that correlation is still present (and whether it is now stronger or weaker than before).
# 
# Run the code below to see how the sample data has changed after having the natural logarithm applied to it.

# In[149]:


# Display the log-transformed sample data
display(log_samples)


# ### Implementation: Outlier Detection
# Detecting outliers in the data is extremely important in the data preprocessing step of any analysis. The presence of outliers can often skew results which take into consideration these data points. There are many "rules of thumb" for what constitutes an outlier in a dataset. Here, we will use [Tukey's Method for identfying outliers](http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/): An *outlier step* is calculated as 1.5 times the interquartile range (IQR). A data point with a feature that is beyond an outlier step outside of the IQR for that feature is considered abnormal.
# 
# In the code block below, you will need to implement the following:
#  - Assign the value of the 25th percentile for the given feature to `Q1`. Use `np.percentile` for this.
#  - Assign the value of the 75th percentile for the given feature to `Q3`. Again, use `np.percentile`.
#  - Assign the calculation of an outlier step for the given feature to `step`.
#  - Optionally remove data points from the dataset by adding indices to the `outliers` list.
# 
# **NOTE:** If you choose to remove any outliers, ensure that the sample data does not contain any of these points!  
# Once you have performed this implementation, the dataset will be stored in the variable `good_data`.

# In[150]:


all_outliers = np.array([], dtype='int64')

# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    
    #  Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature], 25)
    
    #  Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature], 75)
    
    #  Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = (Q3 - Q1) * 1.5
    
    outlier_points = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    all_outliers = np.append(all_outliers, outlier_points.index.values.astype('int64'))
    # Display the outliers
    print "Data points considered outliers for the feature '{}':".format(feature)
    display(outlier_points)

all_outliers, indices = np.unique(all_outliers, return_inverse=True)
counts = np.bincount(indices)
outliers = all_outliers[counts>1]

print outliers

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)


# ### Question 4
# * Are there any data points considered outliers for more than one feature based on the definition above? 
# * Should these data points be removed from the dataset? 
# * If any data points were added to the `outliers` list to be removed, explain why.
# 
# ** Hint: ** If you have datapoints that are outliers in multiple categories think about why that may be and if they warrant removal. Also note how k-means is affected by outliers and whether or not this plays a factor in your analysis of whether or not to remove them.

# **Answer:
# 
# There are several data points which are outliers for more than one feature [65, 66, 75, 128, 154].With 154 having the most outlying features with Milk, Grocery and Delicatessen. They should be removed from the dataset since they are outside of the expected range and skewing the analysis for our attempt to cluster the data to split the customers into segments. 
# 
# 
# I believe they should be removed because they represent truly unusual points with respect to the rest of the data, and therefore will be of little use in forming clusters of similar customers. For this reason I only chose to remove the 5 datapoints that are considered outliers in more than one feature to reduce the potential of skewing our results, and removing the outliers will allow us to more easily cluster the dataset features and hopefully reduce dimensionality
# 
# **

# ## Feature Transformation
# In this section you will use principal component analysis (PCA) to draw conclusions about the underlying structure of the wholesale customer data. Since using PCA on a dataset calculates the dimensions which best maximize variance, we will find which compound combinations of features best describe customers.

# ### Implementation: PCA
# 
# Now that the data has been scaled to a more normal distribution and has had any necessary outliers removed, we can now apply PCA to the `good_data` to discover which dimensions about the data best maximize the variance of features involved. In addition to finding these dimensions, PCA will also report the *explained variance ratio* of each dimension — how much variance within the data is explained by that dimension alone. Note that a component (dimension) from PCA can be considered a new "feature" of the space, however it is a composition of the original features present in the data.
# 
# In the code block below, you will need to implement the following:
#  - Import `sklearn.decomposition.PCA` and assign the results of fitting PCA in six dimensions with `good_data` to `pca`.
#  - Apply a PCA transformation of `log_samples` using `pca.transform`, and assign the results to `pca_samples`.

# In[151]:


#  Apply PCA by fitting the good data with the same number of dimensions as features
from sklearn.decomposition import PCA
pca = PCA(n_components=6)
pca.fit(good_data)

#  Transform the sample log-data using the PCA fit above
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
pca_results = vs.pca_results(good_data, pca)

# Print cumulative sums of the variances explained by the PCA as suggested by reviewer.
print "PCA dimensions 1&2 cumulative sums: ", pca_results['Explained Variance'].cumsum()[1]
print "PCA dimensions 1,2,3&4 cumulative sums: ", pca_results['Explained Variance'].cumsum()[3]


# ### Question 5
# 
# * How much variance in the data is explained* **in total** *by the first and second principal component? 
# * How much variance in the data is explained by the first four principal components? 
# * Using the visualization provided above, talk about each dimension and the cumulative variance explained by each, stressing upon which features are well represented by each dimension(both in terms of positive and negative variance explained). Discuss what the first four dimensions best represent in terms of customer spending.
# 
# **Hint:** A positive increase in a specific dimension corresponds with an *increase* of the *positive-weighted* features and a *decrease* of the *negative-weighted* features. The rate of increase or decrease is based on the individual feature weights.

# **Answer:
# 
# The first principal component explains 44.3% of the variance, and the second principal component is responsible for 26.4%. Together the first 2 account for 70.7% of all variance in the data. When you include the 3rd and 4th principal component (12.3% and 10.1% respectively) we've now accounted for 93.1%.
#  
# 
# Regarding spending, the first component (dimension 1 in the diagram),which makes up 44.3% of total variance.A customer with higher values on the first dimension would spend much more on Fresh and Frozen, whereas, with lower values on first diamension would spend on Detergents_Paper, Milk and Groceries, but, these are negative-weighted featues.Customers with Low values in this component purchase a lot in these three features, while those with high values in the component purchase a lot of Fresh and Frozen.
# 
# The second principal component (diamension 2), which makes up 26.4% of total variance,represents a pattern of above average spending on none of the features/products, but more interestingly, way below average spending on Fresh, Milk, Frozen and Delicatessen. Customers avoid purchasing Fresh, Milk, Frozen and Delicatessen , while those with Low values in the component do not follow specific this pattern.
# 
# The third principal component (dimension 3), which makes up 12.38% of spending patterns, represents a pattern of above average spending on Delicatessen, Frozen and milk features/products, but more interestingly, way below average spending on Detergents_paper and Fresh. Customers with High values in this component purchase a lot of Delicatessen, Frozen and milk and avoid purchasing Fresh and Detergents_Paper, while those with Low values in the component do not follow specific spending pattern.
# 
# The fourth principal component (dimension 4), which makes up 10.12% of the spending patterns, represents a pattern of above average spending on Frozen and Detergents_paper, but more interestingly, way below average spending on Froz Delicatessen and Fresh. Customers with High values in this component purchase a lot of Frozen, Grocery and Detergents_paper and avoid purchasing Delicatessen and Fresh, while those with Low values in the component do not follow this spending pattern.
# 
# **

# ### Observation
# Run the code below to see how the log-transformed sample data has changed after having a PCA transformation applied to it in six dimensions. Observe the numerical value for the first four dimensions of the sample points. Consider if this is consistent with your initial interpretation of the sample points.

# In[152]:


# Display sample log-data after having a PCA transformation applied
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))


# ### Implementation: Dimensionality Reduction
# When using principal component analysis, one of the main goals is to reduce the dimensionality of the data — in effect, reducing the complexity of the problem. Dimensionality reduction comes at a cost: Fewer dimensions used implies less of the total variance in the data is being explained. Because of this, the *cumulative explained variance ratio* is extremely important for knowing how many dimensions are necessary for the problem. Additionally, if a signifiant amount of variance is explained by only two or three dimensions, the reduced data can be visualized afterwards.
# 
# In the code block below, you will need to implement the following:
#  - Assign the results of fitting PCA in two dimensions with `good_data` to `pca`.
#  - Apply a PCA transformation of `good_data` using `pca.transform`, and assign the results to `reduced_data`.
#  - Apply a PCA transformation of `log_samples` using `pca.transform`, and assign the results to `pca_samples`.

# In[153]:



#  Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components=2).fit(good_data)

#  Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

#  Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])


# ### Observation
# Run the code below to see how the log-transformed sample data has changed after having a PCA transformation applied to it using only two dimensions. Observe how the values for the first two dimensions remains unchanged when compared to a PCA transformation in six dimensions.

# In[154]:


# Display sample log-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))


# ## Visualizing a Biplot
# A biplot is a scatterplot where each data point is represented by its scores along the principal components. The axes are the principal components (in this case `Dimension 1` and `Dimension 2`). In addition, the biplot shows the projection of the original features along the components. A biplot can help us interpret the reduced dimensions of the data, and discover relationships between the principal components and original features.
# 
# Run the code cell below to produce a biplot of the reduced-dimension data.

# In[155]:


# Create a biplot
vs.biplot(good_data, reduced_data, pca)


# ### Observation
# 
# Once we have the original feature projections (in red), it is easier to interpret the relative position of each data point in the scatterplot. For instance, a point the lower right corner of the figure will likely correspond to a customer that spends a lot on `'Milk'`, `'Grocery'` and `'Detergents_Paper'`, but not so much on the other product categories. 
# 
# From the biplot, which of the original features are most strongly correlated with the first component? What about those that are associated with the second component? Do these observations agree with the pca_results plot you obtained earlier?

# ## Clustering
# 
# In this section, you will choose to use either a K-Means clustering algorithm or a Gaussian Mixture Model clustering algorithm to identify the various customer segments hidden in the data. You will then recover specific data points from the clusters to understand their significance by transforming them back into their original dimension and scale. 

# ### Question 6
# 
# * What are the advantages to using a K-Means clustering algorithm? 
# * What are the advantages to using a Gaussian Mixture Model clustering algorithm? 
# * Given your observations about the wholesale customer data so far, which of the two algorithms will you use and why?
# 
# ** Hint: ** Think about the differences between hard clustering and soft clustering and which would be appropriate for our dataset.

# **Answer:
# 
# 
# k-means clustering is rather easy to implement and apply even on large data sets, particularly when using heuristics such as Lloyd's algorithm . It has been successfully used in various topics, including market segmentation, computer vision, geo statistics, astronomy and agriculture. The main advantages are listed below:
# •	If variables are huge, then  K-Means most of the times computationally faster than hierarchical clustering, if we keep k smalls.
# •	K-Means produce tighter clusters than hierarchical clustering, especially if the clusters are globular.
# •	simple to understand
# •	fast
# •	more scalable
# A Gaussian mixture model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. One can think of mixture models as generalizing k-means clustering to incorporate information about the covariance structure of the data as well as the centres of the latent Gaussians. The advantages of Gaussian mixture model are listed below:
# •	Speed - It is the fastest algorithm for learning mixture models
# •	Agnostic - As this algorithm maximizes only the likelihood, it will not bias the means towards zero, or bias the cluster sizes to have specific structures that might or might not apply.
# The advantage of a Gaussian Mixture Model (GMM) over the K-mean is its capability of incorporating the covariance between the points into the model to identify more complex clusters. Unlike K-means which assumes, during each iteration, that any given point can only belong to a specific cluster, GMM also takes into account the level of certainty with which a point belongs to a given cluster. This uncertainty is also revised during each iteration making the algorithm more flexible when assigning points to a cluster and capable of performing well on in less clearly defined datasets.
# From the biplot, it can be observed that the data points are mostly densily packed on an area of the plot but do not form clearly deliniated clusters as certain points seem to be in the border bettween two or more groups. We can also observe that certain dimensions in the data (i.e. Milk-Grocery-Detergents and Fresh-Frozen) have a strong degree of correlation between each other. Based on these facts and on the previous discussion, we can safely conclude that applying a Gausian Mixture Model will produce the best outcome for the problem at hand.
# 
# 
# **

# ### Implementation: Creating Clusters
# Depending on the problem, the number of clusters that you expect to be in the data may already be known. When the number of clusters is not known *a priori*, there is no guarantee that a given number of clusters best segments the data, since it is unclear what structure exists in the data — if any. However, we can quantify the "goodness" of a clustering by calculating each data point's *silhouette coefficient*. The [silhouette coefficient](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) for a data point measures how similar it is to its assigned cluster from -1 (dissimilar) to 1 (similar). Calculating the *mean* silhouette coefficient provides for a simple scoring method of a given clustering.
# 
# In the code block below, you will need to implement the following:
#  - Fit a clustering algorithm to the `reduced_data` and assign it to `clusterer`.
#  - Predict the cluster for each data point in `reduced_data` using `clusterer.predict` and assign them to `preds`.
#  - Find the cluster centers using the algorithm's respective attribute and assign them to `centers`.
#  - Predict the cluster for each sample data point in `pca_samples` and assign them `sample_preds`.
#  - Import `sklearn.metrics.silhouette_score` and calculate the silhouette score of `reduced_data` against `preds`.
#    - Assign the silhouette score to `score` and print the result.

# In[156]:


from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

#  Apply your clustering algorithm of choice to the reduced data 
clusterer = GaussianMixture(n_components=2).fit(reduced_data)

#  Predict the cluster for each data point
preds = clusterer.predict(reduced_data)

#  Find the cluster centers
centers = clusterer.means_

#  Predict the cluster for each transformed sample data point
sample_preds = clusterer.predict(pca_samples)

#  Calculate the mean silhouette coefficient for the number of clusters chosen
score = silhouette_score(reduced_data,preds)

print score


# ### Question 7
# 
# * Report the silhouette score for several cluster numbers you tried. 
# * Of these, which number of clusters has the best silhouette score?

# **Answer:
# 
# 
# 
# 
# I have  tried cluster numbers from 2 to 7. Below is a table that summarizes the results as well as graphical representations shown above. Of the cluster numbers we tried, 2 has the best silhouette score of 0.411818864386.
# 
# 
# CLUSTER NUMBER	-  SILHOUETTE SCORE
# 
#       2	            0.411818864386
# 
#       3	            0.373560747175
# 
#       4	            0.308878605178
# 
#       5	            0.280983407051
# 
#       6	            0.270498049376
# 
#       7             0.325023833619
# 
# As the summaries show, we generally got better silhouette scores as we reduced the number of clusters.
# 
# **

# ### Cluster Visualization
# Once you've chosen the optimal number of clusters for your clustering algorithm using the scoring metric above, you can now visualize the results by executing the code block below. Note that, for experimentation purposes, you are welcome to adjust the number of clusters for your clustering algorithm to see various visualizations. The final visualization provided should, however, correspond with the optimal number of clusters. 

# In[157]:


# Display the results of the clustering from implementation
vs.cluster_results(reduced_data, preds, centers, pca_samples)


# ### Implementation: Data Recovery
# Each cluster present in the visualization above has a central point. These centers (or means) are not specifically data points from the data, but rather the *averages* of all the data points predicted in the respective clusters. For the problem of creating customer segments, a cluster's center point corresponds to *the average customer of that segment*. Since the data is currently reduced in dimension and scaled by a logarithm, we can recover the representative customer spending from these data points by applying the inverse transformations.
# 
# In the code block below, you will need to implement the following:
#  - Apply the inverse transform to `centers` using `pca.inverse_transform` and assign the new centers to `log_centers`.
#  - Apply the inverse function of `np.log` to `log_centers` using `np.exp` and assign the true centers to `true_centers`.
# 

# In[158]:



#  Inverse transform the centers

log_centers = pca.inverse_transform(centers)

#  Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)


# ### Question 8
# 
# * Consider the total purchase cost of each product category for the representative data points above, and reference the statistical description of the dataset at the beginning of this project(specifically looking at the mean values for the various feature points). What set of establishments could each of the customer segments represent?
# 
# **Hint:** A customer who is assigned to `'Cluster X'` should best identify with the establishments represented by the feature set of `'Segment X'`. Think about what each segment represents in terms their values for the feature points chosen. Reference these values with the mean values to get some perspective into what kind of establishment they represent.

# **Answer:
# 
# 
# A customer assigned to Segment 0 would most likely represent some type of restaurant, coffee houses or dinners  as they have higher than average purchase costs of freshs. This can be concluded based on their same purchase costs across the milk, grocery and frozen product categories and higher than average purchase costs of freshs.
# 
# On the other hand, a customer assigned to Segment 1 most likely represents some type of market, vegetable,supermarket or covinience store as they have higher than average or close to average purchase costs of Groceries and Milk and detergent_paper.
# 
# **

# ### Question 9
# 
# * For each sample point, which customer segment from* **Question 8** *best represents it? 
# * Are the predictions for each sample point consistent with this?*
# 
# Run the code block below to find which cluster each sample point is predicted to be.

# In[159]:


# Display the predictions
for i, pred in enumerate(sample_preds):
    print "Sample point", i, "predicted to be in segment", pred


# **Answer:
# 
# For the 3 samples that we picked earlier,firstly, sample points 0 is placed in  segment 1, which have high Milk, Grocery and Detergents_Paper feature spending patterns. This is consistant for sample 0, since they also purchase high quantities of Milk, Grocery and Detergents_Paper products.Sample point 0 is best represented by a market, vegetable,supermarket or covinience store as it have higher than average or close to average purchase costs of Groceries, Milk and detergent_paper.
# 
# sample points 1 and 2 are placed in  segment 0, which have high Fresh feature spending patterns.Sample point 1 and 2  is best represented by a restaurant, coffee houses or dinners.Segment 0 would most likely represent some type of market, vegetable or a covinience store as they have higher than average purchase costs of freshs.This can be concluded based on their same purchase costs across the milk, grocery and frozen product categories and higher than average purchase costs of freshs.**

# ## Conclusion

# In this final section, you will investigate ways that you can make use of the clustered data. First, you will consider how the different groups of customers, the ***customer segments***, may be affected differently by a specific delivery scheme. Next, you will consider how giving a label to each customer (which *segment* that customer belongs to) can provide for additional features about the customer data. Finally, you will compare the ***customer segments*** to a hidden variable present in the data, to see whether the clustering identified certain relationships.

# ### Question 10
# Companies will often run [A/B tests](https://en.wikipedia.org/wiki/A/B_testing) when making small changes to their products or services to determine whether making that change will affect its customers positively or negatively. The wholesale distributor is considering changing its delivery service from currently 5 days a week to 3 days a week. However, the distributor will only make this change in delivery service for customers that react positively. 
# 
# * How can the wholesale distributor use the customer segments to determine which customers, if any, would react positively to the change in delivery service?*
# 
# **Hint:** Can we assume the change affects all customers equally? How can we determine which group of customers it affects the most?

# **Answer:
# 
# By identifying the underlying type of customer segments (through clustering), the wholesaler will be able to draw more meaningfull hypothesis about expected behavior of the customers in each segment prior to performing the A/B test. Then these hypothesis may be tested on each segment separately to find more meaningful conclusions and understand the impact level on each independent customer segment.
# 
# 
# For example, by observing the definitions of these two customer segments, the wholesaler could draw the preliminary hypothesis that "market/convenient store" customers(i.e. segment 1) and "restaurant" customers (i.e. segment 0) will react different to a reduction in number of deliveries.
# 
#   •	It is likely that customers from segment 0 who serve lots of fresh     food are going to want 5-day weeks in order to keep food as fresh       as possible
#   •	Segment 1 could be more flexible - they buy a more wide variety of     perishable and non-perishable goods so do not necessarilly need a       daily delivery.
# 
# However, those are just guesses, and an effective A/B testing program will be able to statistically prove if changing the delievery service will make different customer segments react positively or negatively to the change, or none at all. This can be done by randomly selecting a sample set of customers from each customer segment and apply the new delivery services as they are designed. Their feedback on the new services will dictate if the services should be rolled out on other customers of the same segments depending on if the service is successful with the A/B testing group.
# 
# If a trend is found in a particular segment, it allows a business to make educated and targeted decisions that would benefit their customers going forward depending on their profile.
# 
# **

# ### Question 11
# Additional structure is derived from originally unlabeled data when using clustering techniques. Since each customer has a ***customer segment*** it best identifies with (depending on the clustering algorithm applied), we can consider *'customer segment'* as an **engineered feature** for the data. Assume the wholesale distributor recently acquired ten new customers and each provided estimates for anticipated annual spending of each product category. Knowing these estimates, the wholesale distributor wants to classify each new customer to a ***customer segment*** to determine the most appropriate delivery service.  
# * How can the wholesale distributor label the new customers using only their estimated product spending and the **customer segment** data?
# 
# **Hint:** A supervised learner could be used to train on the original customers. What would be the target variable?

# **Answer:
# 
# The wholesale distributor could develop a unsupervised clustering and then a supervised clustering technique.
# 
# By first running an unsupervised clustering approach, such as GMM, we first establish clusters and use this as a new feature - which cluster they are in. We can call this feature 'Customer Segment'.
# 
# Afterwhich, the wholesale distributor could train a supervised machine learning classification algorithm (e.g. SVC, or decision tree classifier, etc) with the initial dataset's customer product spending as inputs and the customer segments (as obtained from GMM clustering) as the target variable. Once the classifier is trained it can be used to predict the customer segment for new customers which would then determine the most appropriate delivery service (3 days per week or 5 days per week).
# 
# Once trained on the existing customer dataset using the new labels, the new supervised learner can be used to predict new customers based on their estimated product spending.Standard Supervised Learning optimizations could be used to tune the model - boosting, cross-validation etc.
# 
# 
# **

# ### Visualizing Underlying Distributions
# 
# At the beginning of this project, it was discussed that the `'Channel'` and `'Region'` features would be excluded from the dataset so that the customer product categories were emphasized in the analysis. By reintroducing the `'Channel'` feature to the dataset, an interesting structure emerges when considering the same PCA dimensionality reduction applied earlier to the original dataset.
# 
# Run the code block below to see how each data point is labeled either `'HoReCa'` (Hotel/Restaurant/Cafe) or `'Retail'` the reduced space. In addition, you will find the sample points are circled in the plot, which will identify their labeling.

# In[160]:


# Display the clustering results based on 'Channel' data
vs.channel_results(reduced_data, outliers, pca_samples)


# ### Question 12
# 
# * How well does the clustering algorithm and number of clusters you've chosen compare to this underlying distribution of Hotel/Restaurant/Cafe customers to Retailer customers? 
# * Are there customer segments that would be classified as purely 'Retailers' or 'Hotels/Restaurants/Cafes' by this distribution? 
# * Would you consider these classifications as consistent with your previous definition of the customer segments?

# **Answer:
# 
# The clustering results from the Gaussian Mixture Model algorithm predicted the cluster/segmented correctly.Both the GMM algorithm and the number of clusters/segments chosen are highly comparable to the underlying distribution shown in the plot above. The red
# 
# 
#   •	A customer assigned to Segment 0 would most likely represent some       type of restaurant, coffee houses or dinners  as they have higher       than average purchase costs of freshs. This can be concluded based     on their same purchase costs across the milk, grocery and frozen       product categories and higher than average purchase costs of freshs.
#   
#   • On the other hand, a customer assigned to Segment 1 most likely  m       represents some type of market, vegetable,supermarket or covinience     store as they have higher than average or close to average purchase     costs of Groceries and Milk and detergent_paper..
# 
# The customer segments as classified here closely match those I previously defined in question 8(also in above 2 paragraphs)(i.e Green points: Market/vegetable/Convenience Store, Red Points: Restaurants/coffee shops/cafe).
# 
# **

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
