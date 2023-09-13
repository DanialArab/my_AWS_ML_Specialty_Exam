# Exam date: ‎Sunday, November 26, 2023‎

1. [AWS free materials/courses](#1)
   1. [Process Model: CRISP-DM on the AWS Stack](#2)
   2. [Exam Readiness: AWS Certified Machine Learning - Specialty](#3)
  

<a name="1"></a>
## AWS free materials/courses

<a name="2"></a>
## Process Model: CRISP-DM on the AWS Stack

CRISP-DM (Cross Industry Standard Process -  Data Mining)

+ Business understanding
+ Data understanding
+ Data Preparation 
+ Modeling
+ Evaluation
+ Deployment 

### AWS Services for data understanding:
In the past, data exploration often meant loading the dataset into redshift tableau or ec2 instances then diving into them through libraries such as ggplot or matplotlib. Today some of these tasks can be addressed in AWS without the need to load the data into a data warehouse or hadoop masternode or downloading the data onto your box. Here are the key servises helping us with data quality and visualizations:

+ Athena
+ QuickSight
+ Glue

#### Glue:
+ managed serverless ETL service
+ includes three components:
  	+ build data catalog
  	+ generate and edit transformations
  	+ schedule and run jobs
#### Athena:
+ interactive query service
+ runs interactive SQL queries on S3 data
+ schema-on-read
+ supports ANSI SQL operators and functions
+ serverless

#### QuickSight
+ cloud-powered BI service
+ scale to hundreds of thousands of users
+ 1/10th of the cost of traditional BI solutions
+ secure sharing and collaboration (StoryBoard)

<a name="3"></a>
## Exam Readiness: AWS Certified Machine Learning - Specialty

### Data Engineering
+ create data repositories for ML
+ identify and implement a data-ingestion solution
+ identify and implement a data-transformation solution

#### First domain in Data Engineering: create data repositories for ML

+ You need a way to store your data in a **centralized repository**. **Data lake** is a key solution to this challenge. With a data lake, you can store structured and unstructured data.
+ AWS Lake Formation is your data lake solution, and Amazon S3 is the preferred storage option for data science processing on AWS
+ Use Amazon S3 storage classes to reduce the cost of data storage.
+ Amazon S3 is integrated with Amazon SageMaker to store your training data and model training output.
+ However, Amazon S3 isn't your only storage solution for model training. 
+ Amazon FSx for Lustre: When your training data is already in Amazon S3 and you plan to run training jobs several times using different algorithms and parameters, consider using Amazon FSx for Lustre, a file system service. FSx for Lustre speeds up your training jobs by serving your Amazon S3 data to Amazon SageMaker at high speeds. The first time you run a training job, FSx for Lustre automatically copies data from Amazon S3 and makes it available to Amazon SageMaker. You can use the same Amazon FSx file system for subsequent iterations of training jobs, preventing repeated downloads of common Amazon S3 objects.
+ Amazon S3 with Amazon EFS: Alternatively, if your training data is already in Amazon Elastic File System (Amazon EFS), we recommend using that as your training data source. Amazon EFS has the benefit of directly launching your training jobs from the service without the need for data movement, resulting in faster training start times. This is often the case in environments where data scientists have home directories in Amazon EFS and are quickly iterating on their models by bringing in new data, sharing data with colleagues, and experimenting with including different fields or labels in their dataset. For example, a data scientist can use a Jupyter notebook to do initial cleansing on a training set, launch a training job from Amazon SageMaker, then use their notebook to drop a column and re-launch the training job, comparing the resulting models to see which works better.
+ When choosing a file system, take into consideration the training load time: The table below shows an example of some different file systems and the relative rate that they can transfer images to a compute cluster. This performance is only a single measurement and is only a suggestion for which file systems you could use for a given workload. The specifics of a given workload might change these results.

![](https://github.com/DanialArab/images/blob/main/AWS_ML_Specialty_Exam/comparison%20of%20training%20load%20time.PNG)

+ Topics related to this subdomain: Here are some topics you may want to study for more in-depth information related to this subdomain:

  + AWS Lake Formation
  + Amazon S3 (as storage for a data lake)
  + Amazon FSx for Lustre
  + Amazon EFS
  + Amazon EBS volumes
  + Amazon S3 lifecycle configuration
  + Amazon S3 data storage options

#### Second domain in Data Engineering: Identify and implement a data ingestion solution

To use this data for ML, you need to ingest it into a service like Amazon S3

One of the core benefits of a data lake solution is the ability to quickly and easily ingest multiple types of data. In some cases, your data will reside outside your Amazon S3 data lake solution, in databases, on-premises storage platforms, data warehouses, and other locations. To use this data for ML, you may need to ingest it into a storage service like Amazon S3.

+ **Batch and stream processing are two kinds of data ingestion**

##### Batch processing

Batch processing periodically collects and groups source data

With batch processing, the ingestion layer periodically collects and groups source data and sends it to a destination like Amazon S3. You can process groups based on any logical ordering, the activation of certain conditions, or a simple schedule. Batch processing is typically used when there is no real need for real-time or near-real-time data, because it is generally easier and more affordably implemented than other ingestion options.

Several services can help with batch processing into the AWS Cloud

For batch ingestions to the AWS Cloud, you can use services like AWS Glue, an ETL (extract, transform, and load) service that you can use to categorize your data, clean it, enrich it, and move it between various data stores. AWS Database Migration Service (AWS DMS) is another service to help with batch ingestions. This service reads from historical data from source systems, such as relational database management systems, data warehouses, and NoSQL databases, at any desired interval. You can also automate various ETL tasks that involve complex workflows by using AWS Step Functions.

##### Stream processing

Stream processing manipulates and loads data as it’s recognized

Stream processing, which includes real-time processing, involves no grouping at all. Data is sourced, manipulated, and loaded as soon as it is created or recognized by the data ingestion layer. This kind of ingestion is less cost-effective, since it requires systems to constantly monitor sources and accept new information. But you might want to use it for real-time predictions using an Amazon SageMaker endpoint that you want to show your customers on your website or some real-time analytics that require continually refreshed data, like real-time dashboards.

+ Amazon Kinesis is a platform for streaming data on AWS

AWS recommends that you capture and ingest this fast-moving data using Amazon Kinesis, a platform for streaming data on AWS. Amazon Kinesis gives you the opportunity to build custom streaming data applications for specialized needs, and it offers several services focused on making it easier to load and analyze your streaming data.

Amazon Kinesis:
+ Amazon Kinesis Video Streams
+ Amazon Kinesis Data Streams
+ Amazon Kinesis Data Firehose
+ Amazon Kinesis Data Analytics

  
###### Amazon Kinesis Video Streams
You can use Amazon Kinesis Video Streams to ingest and analyze video and audio data. For example, a leading home security provider ingests audio and video from their home security cameras using Kinesis Video Streams. They then attach their own custom ML models running in Amazon SageMaker to detect and analyze objects to build richer user experiences.

###### Amazon Kinesis Data Streams
With Amazon Kinesis Data Streams, you can use the Kinesis Producer Library (KPL), an intermediary between your producer application code and the Kinesis Data Streams API data, to write to a Kinesis data stream. With the Kinesis Client Library (KCL), you can build your own application to preprocess the streaming data as it arrives and emit the data for generating incremental views and downstream analysis.

###### Amazon Kinesis Data Firehose
As data is ingested in real time, you can use Amazon Kinesis Data Firehose to easily batch and compress the data to generate incremental views. Kinesis Data Firehose also allows you to execute custom transformation logic using AWS Lambda before delivering the incremental view to Amazon S3.

###### Amazon Kinesis Data Analytics
Amazon Kinesis Data Analytics provides the easiest way to process and transform the data that is streaming through Kinesis Data Streams or Kinesis Data Firehose using SQL. This lets you gain actionable insights in near-real time from the incremental stream before storing it in Amazon S3.


+ Topics related to this subdomain: Here are some topics you may want to study for more in-depth information related to this subdomain:
  
  + Amazon Kinesis Data Streams
  + Amazon Kinesis Data Firehose
  + Amazon Kinesis Data Analytics
  + Amazon Kinesis Video Streams
  + AWS Glue
  + Apache Kafka


#### Third domain in Data Engineering: Identify and implement a data transformation solution

Raw ingested data is not ML ready

The raw data ingested into a service like Amazon S3 is usually not ML ready as is. The data needs to be transformed and cleaned, which includes deduplication, incomplete data management, and attribute standardization. Data transformation can also involve changing the data structures, if necessary, usually into an OLAP model to facilitate easy querying of data. Doing this in the context of ML, while using key services that help you with data transformation, is the focus of this subdomain.

Transforming your data for ML

Data transformation is often necessary to deal with huge amounts of data. Distributed computation frameworks like MapReduce and Apache Spark provide a protocol of data processing and node task distribution and management. They also use algorithms to split datasets into subsets and distribute them across nodes in a compute cluster.

Using Apache Spark on Amazon EMR provides a managed framework

Using Apache Spark on Amazon EMR provides a managed framework that can process massive quantities of data. Amazon EMR supports many instance types that have proportionally high CPU with increased network performance, which is well suited for HPC (high-performance computing) applications.

A key step in data transformation for ML is partitioning your dataset

Datasets required for ML applications are often pulled from database warehouses, streaming IoT input, or centralized data lakes. In many use cases, you can use Amazon S3 as a target endpoint for their training datasets. ETL processing services (Amazon Athena, AWS Glue, Amazon Redshift Spectrum) are functionally complementary and can be built to preprocess datasets stored in or targeted to Amazon S3. In addition to transforming data with services like Athena and Amazon Redshift Spectrum, you can use services like AWS Glue to provide metadata discovery and management features. The choice of ETL processing tool is also largely dictated by the type of data you have. For example, tabular data processing with Athena lets you manipulate your data files in Amazon S3 using SQL. If your datasets or computations are not optimally compatible with SQL, you can use AWS Glue to seamlessly run Spark jobs (Scala and Python support) on data stored in your Amazon S3 buckets.

You can store a single source of data in Amazon S3 and perform ad hoc analysis

This reference architecture shows how AWS services for big data and ML can help build a scalable analytical layer for healthcare data. Customers can store a single source of data in Amazon S3 and perform ad hoc analysis with Athena, integrate with a data warehouse on Amazon Redshift, build a visual dashboard for metrics using Amazon QuickSight, and build an ML model to predict readmissions using Amazon SageMaker. By not moving the data around and connecting to it using different services, customers avoid building redundant copies of the same data.

Topics related to this subdomain: Here are some topics you may want to study for more in-depth information related to this subdomain:

+ Apache Spark on Amazon EMR
+ Apache Spark and Amazon SageMaker
+ AWS Glue

### EDA

+ Sanitize and prepare data for modeling
+ Perform feature engineering
+ Analyze and visualize data for ML

Topics related to this subdomain: Here are some topics you may want to study for more in-depth information related to this subdomain

+ Dataset generation

   + Amazon SageMaker Ground Truth
   + Amazon Mechanical Turk 
   + Amazon Kinesis Data Analytics
   + Amazon Kinesis Video Streams

+ Data augmentation
+ Descriptive statistics
+ Informative statistics
+ Handling missing values and outliers

some points on feature eng.:
+ You may need to perform feature engineering because of the dimensionality of your dataset—particularly if there are too many features for your model to handle. To reduce the number of features, you need to deploy dimensionality reduction techniques like **principal component analysis (PCA)** or **t-distributed stochastic neighbor embedding.**
+ You will often handle categorical data that needs to be converted into numerical data before it can be read by your ML algorithm. Your approach will differ depending on whether your data is ordinal (the categories are ordered) or nominal (categories are not ordered).
+ An example: Home type and garden size represent categorical features. Garden size, more specifically, represents ordinal data, because Small, Medium, and Large can be represented in an order (note: N/A represents “no garden”). For ordinal variables like this one, you can use a map function in Pandas to convert the text into numerical values. For example, you can define the relative difference for those different categories in the ordinal variable. Often, the numerical value you provide in the mapping is derived from your business insight of the dataset and the business itself. For the garden size S, you can use 5; for M, use 10; for L, use 20; and for N, use 0. You can easily apply the map function from Pandas to replace the categorical variable with the numerical value. It is a one-to-one mapping.
+ By contrast, AWS does not recommend encoding nominal variables like home type to numerical data. If you encode this variable or feature into integers, it becomes one, two, and three. One, two, and three really implies that something has a numerical value. They have order difference, and there is also a magnitude to the difference between the numbers. These additional features are artifacts that do not belong to the original data. And these artifacts may give you the wrong or unexpected results. So how do you encode nominal variables? The one-hot encoding method is a good choice. Here’s how it works.
+ Common techniques for scaling
So how do we do it, exactly? How can we align different features into the same scale? Keep in mind that not all ML algorithms will be sensitive to different scales of inputted features. Here is a collection of commonly used scaling and normalizing transformations that we usually use for data science and ML projects:

   + Mean/variance standardization
   + MinMax scaling
   + Maxabs scaling
   + Robust scaling
   + Normalizer

Topics related to this subdomain: Here are some topics you may want to study for more in-depth information related to this subdomain:

+ Scaling
+ Normalizing
+ Dimensionality reduction
+ Date formatting
+ One-hot encoding


### Modeling 

+ Frame business problems as ML problems
+ Select the appropriate model(s) for an ML problem
+ Train ML models
+ Perform hyperparameter optimization
+ Evaluate ML models

#### Amazon SageMaker built-in algorithms for NLP and CV

Algorithms for natural language processing (NLP)

There are Amazon SageMaker built-in algorithms for natural language processing:

+ **BlazingText** algorithm provides highly optimized implementations of the Word2vec and text classification algorithms.
+ **Sequence2sequence** is a supervised learning algorithm where the input is a sequence of tokens (for example, text, audio) and the output generated is another sequence of tokens.
+ **Object2Vec** generalizes the well-known Word2Vec embedding technique for words that are optimized in the Amazon SageMaker BlazingText algorithm.

#### Algorithms for computer vision (CV)

There are Amazon SageMaker built-in algorithms for computer vision:

+ Image classification is a supervised learning algorithm used to classify images.
+ Object detection algorithm detects and classifies objects in images using a single deep neural network. It is a supervised learning algorithm that takes images as input and identifies all instances of objects within the image scene. The object is categorized into one of the classes in a specified collection with a confidence score that it belongs to the class. Its location and scale in the image are indicated by a rectangular bounding box.
+ Semantic segmentation algorithm tags every pixel in an image with a class label from a predefined set of classes.

## References:

AWS Certified Machine Learning Specialty 2023 - Hands On!

Process Model: CRISP-DM on the AWS Stack (https://explore.skillbuilder.aws/learn/course/external/view/elearning/14575/process-model-crisp-dm-on-the-aws-stack-thai) 

Exam Readiness: AWS Certified Machine Learning - Specialty

Exam Prep Official Practice Question Set: AWS Certified Machine Learning - Specialty (MLS-C01 - English)
