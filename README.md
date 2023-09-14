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

### FIRST DOMAIN: Data Engineering
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

### SECOND DOMAIN: EDA

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


### THIRD DOMAIN: Modeling 

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

#### Other options for training algorithms

Up to now, the focus has been exclusively on Amazon SageMaker built-in algorithms, but there are other options for training algorithms:

+ Use Apache Spark with Amazon SageMaker 
+ Submit custom code to train a model with a deep learning framework like TensorFlow or Apache MXNet
+ Use your own custom algorithm and put the code together as a Docker image
+ Subscribe to an algorithm from AWS Marketplace

### Train ML models

+ Use validation dataset to estimate model's performance while tunning its hyperparameters and/or to compare performance across different models you may be considering 
+ K-fold cross-validation is a common validation method

K-fold cross-validation is a common validation method. In k-fold cross-validation, you split the input data into k subsets of data (also known as folds). You train your models on all but one (k-1) of the subsets, and then evaluate them on the subset that was not used for training. This process is repeated k times, with a different subset reserved for evaluation (and excluded from training) each time. 

For instance, performing a 5-fold cross-validation generates four models, four datasets to train the models, four datasets to evaluate the models, and four evaluations, one for each model. In a 5-fold cross-validation for a binary classification problem, each of the evaluations reports an area under curve (AUC) metric. You can get the overall performance measure by computing the average of the four AUC metrics.

+ Other cross-validation methods

You can use a host of other cross-validation methods, depending on your requirements. If you have a **small dataset, for instance, consider Leave-one-out cross-validation** or, as mentioned above, K-fold cross-validation. Or you might use the **Stratified K-fold cross-validation** when you have **imbalanced data**. Just remember that these techniques increase the computational power needed during training.

+ **Cross-validation techniques increase the computational power needed for the training**

#### Creating a training job in Amazon SageMaker

With an algorithm chosen and your data split up, you can now run the actual training job. Creating a training job in Amazon SageMaker typically requires the following steps:

+ S3 bucket training data (we need to provide the URLs of the S3 buckets for the training data and also for the model output or artifacts where we want to store them 
+ We need to specify the compute resources that we want Amazon Sagemaker to use for training. compute resources are ML compute instances that are managed by SageMaker
+ We need to specify the Amazon Elastic Container Registry path where the training code is stored 

After you create the training job, Amazon SageMaker launches the compute instances and uses the training code and the training dataset to train the model. It saves the resulting model artifacts and other output in the Amazon S3 bucket you specified for that purpose.

Topics related to this subdomain: Here are some topics you may want to study for more in-depth information related to this subdomain:

+ Amazon SageMaker workflow for training jobs
+ Running a training job using containers
+ Build your own containers
+ P3 instances
+ Components of an ML training job for deep learning

P3 instances are part of the AWS EC2 (Elastic Compute Cloud) family and are designed for high-performance machine learning (ML), deep learning (DL), and artificial intelligence (AI) workloads. They are optimized for GPU-intensive tasks.

###  Perform hyperparameter optimization

There are different categories of hyperparameters: 

+ Model hyperparameters: they define the model itself - attributes of a NN architecture like filter size, pooling, stride, padding
+ Optimizer hyperparameters: are related to how the model learns the patterns based on data and are used for an NN model. These types of hyperparameters include optimizers like gradient descent and stochastic gradient descent or even optimizers using momentum like Adam or initializing the parameter weights using methods like Xavier initialization or He initialization
+ Data hyperparameters: are related to the attributes of the data, often used when you don't have enough data or enough variation in data - Data augmentation techniques like cropping, resizing 

Traditionally, this was done manually: someone who has domain experience related to that hyperparameter and the use case would manually select the hyperparameters based on their intuition and experience. Then they would train the model and score it on the validation data. This process would be repeated over and over again until satisfactory results are achieved. 

Needless to say, this is not always the most thorough and efficient way of tuning your hyperparameters. As a result, several other methods for hyperparameter tuning have been developed. Alternatively, Amazon SageMaker offers automated hyperparameter tuning

Then there’s automated hyperparameter tuning, which uses methods like gradient descent, Bayesian optimization, and evolutionary algorithms to conduct a guided search for the best hyperparameter settings.

Amazon SageMaker lets you perform automated hyperparameter tuning. Amazon SageMaker automatic model tuning, also known as hyperparameter tuning, finds the best version of a model by running many training jobs on your dataset using the algorithm and ranges of hyperparameters that you specify. It then chooses the hyperparameter values that result in a model that performs the best, as measured by a metric that you choose.

Topics related to this subdomain: Here are some topics you may want to study for more in-depth information related to this subdomain:


+ Amazon SageMaker hyperparameter tuning jobs
+ Common hyperparameters to tune:

   + Momentum
   + Optimizers
   + Activation functions
   + Dropout
   + Learning rate

+ Regularization:
   + Dropout
  +  L1/L2

### Evaluate ML models

Topics related to this subdomain: Here are some topics you may want to study for more in-depth information related to this subdomain:

+ Metrics for regression: sum of squared errors, RMSE
+ Sensitivity
+ Specificity
+ Neural network functions like Softmax for the last layer

**Amazon SageMaker DeepAR**:
The Amazon SageMaker DeepAR forecasting algorithm is a supervised learning algorithm for forecasting scalar (one-dimensional) **time series using recurrent neural networks (RNN).**

### FOURTH DOMAIN: ML Implementation and Operations

This domain covers four subdomains:

+ Building ML solutions for performance, availability, scalability, resiliency, and fault tolerance
+ Recommending and implementing the appropriate ML services and features for a given problem
+ Applying basic AWS security practices to ML solutions
+ Deploying and operationalizing ML solutions

#### Domain 4.1: Build ML solutions for performance, availability, scalability, resiliency, and fault tolerance

Part of this domain is focused on deploying your ML solution into production. But before you walk through the steps of model deployment, you have to ensure your solution is **designed to effectively deal with operational failure**. This subdomain focuses on best practices for how to do this in the context of machine learning.

##### High availability and fault tolerance

At the heart of designing for failure are two concepts known as high availability and fault tolerance.

In a highly available solution, the system will continue to function even when any component of the architecture stops working. A key aspect of high availability is fault tolerance, which, when built into an architecture, ensures that applications will continue to function without degradation in performance, despite the complete failure of any component of the architecture.

##### One method of achieving high availability and fault tolerance is loose coupling

With a loosely coupled, distributed system, the failure of one component can be managed in between your application tiers so that the faults do not spread beyond that single point. Loose coupling is often achieved by making sure application components are independent of each other. For example, you should always decouple your storage layer with your compute layer because a training job only requires minimal time, but storing data is permanent. Decoupling helps turn off the compute resources when they are not needed.

**Tightly coupled**:
   + More interdependency
  +  More coordination
   + More information

**Loosely coupled**:
   + Less interdependency
   + Less coordination
   + Less information

**Queues are used in loose coupling to pass messages between components**

In a general architecture, you can use a queue service like Amazon SQS or workflow service like AWS Step Functions to create a workflow between various components.

**Amazon CloudWatch helps you monitor your system**

Services like Amazon CloudWatch help you monitor your system while storing all the logs and operational metrics separately from the actual implementation and code for training and testing your ML models. In this example, Amazon CloudWatch is used to keep a history of the model metrics for a specific amount of time, visualize model performance metrics, and create a CloudWatch dashboard. Amazon SageMaker provides out-of-the-box integration with Amazon CloudWatch, which collects near-real-time utilization metrics for the training job instance, such as CPU, memory, and GPU utilization of the training job container.

**AWS CloudTrail captures API calls and related events**

AWS CloudTrail captures API calls and related events made by or on behalf of your AWS account and delivers the log files to an Amazon S3 bucket that you specify. You can identify which users and accounts called AWS, the source IP address from which the calls were made, and when the calls occurred.

**You can design for the failure of any individual component by leveraging key AWS services and features**
+ AWS Glue and Amazon EMR:

You should decouple your ETL process from the ML pipeline. The compute power needed for ML isn’t the same as what you’d need for an ETL process—they have very different requirements. 

   + An ETL process needs to read in files from multiple formats, transform them as needed, and then write them back to a persistent storage. Keep in mind that reading and writing takes a lot of memory and disk I/O, so when you decouple your ETL process, use a framework like Apache Spark, which can handle large amounts of data easily for ETL.
   + Training, on the other hand, may require GPUs which are much more suited to handle the training requirements than CPUs. However, GPUs are less cost-effective to keep running when a model is not being trained. So you can make use of this decoupled architecture by simply using an ETL service like AWS Glue or Amazon EMR, which use Apache Spark for your ETL jobs and Amazon SageMaker to train, test, and deploy your models.

+ Amazon SageMaker Endpoints

To ensure a highly available ML serving endpoint, deploy Amazon SageMaker endpoints backed by multiple instances across Availability Zones. 

+ Amazon SageMaker

Amazon SageMaker makes it easy to containerize ML models for both training and inference. In doing so, you can create ML models made up of loosely coupled, distributed services that can be placed on any number of platforms, or close to the data that the applications are analyzing.

+ AWS Auto Scaling

Use AWS Auto Scaling to build scalable solutions by configuring automatic scaling for the AWS resources such as Amazon SageMaker endpoints that are part of your application in response to the changes in traffic to your application.

With AWS Auto Scaling, you configure and manage scaling for your resources through a scaling plan. The scaling plan uses dynamic scaling and predictive scaling to automatically scale your application’s resources. 

This ensures that you add the required computing power to handle the load on your application, and then remove it when it's no longer required. The scaling plan lets you choose scaling strategies to define how to optimize your resource utilization. You can optimize for availability, for cost, or a balance of both. 

As you increase the number of concurrent prediction requests, at some point the endpoint responds more slowly and eventually errors out for some requests. Automatically scaling the endpoint avoids these problems and improves prediction throughput. When the endpoint scales out, Amazon SageMaker automatically spreads instances across multiple Availability Zones. This provides Availability Zone-level fault tolerance and protects from an individual instance failure.

If the endpoint has only a moderate load, you can run it on a single instance and still get good performance. Use automatic scaling to ensure high availability during traffic fluctuations without having to constantly provision for peak traffic. For production workloads, use at least two instances. Because Amazon SageMaker automatically spreads the endpoint instances across multiple Availability Zones, a minimum of two instances ensures high availability and provides individual fault tolerance.

To determine the scaling policy for automatic scaling in Amazon SageMaker, test for how much load (RPS) the endpoint can sustain. Then configure automatic scaling and observe how the model behaves when it scales out. Expected behavior is lower latency and fewer or no errors with automatic scaling.

**Designing highly available and fault-tolerant ML architectures**

The example below brings together these and some other best practices related to designing highly available and fault-tolerant ML architectures. 

As previously mentioned, you can deploy your Amazon SageMaker-built models to an Amazon SageMaker endpoint. Once created, you need to invoke the endpoint outside the Amazon SageMaker notebook instance with appropriate input (the model signature). These input parameters can be in a file format such as CSV and LIBSVM, as well as an audio, image, or video file. You can use AWS Lambda and Amazon API Gateway to format the input request and invoke the endpoint from the web. 

Topics related to this subdomain: Here are some topics you may want to study for more in-depth information related to this subdomain:

+ Amazon Deep Learning containers
+ AWS Deep Learning AMI (Amazon Machine Image)
+ AWS Auto Scaling
+ AWS GPU (P2 and P3) and CPU instances
+ Amazon CloudWatch
+ AWS CloudTrail

#### Domain 4.2: Recommend and implement the appropriate ML services and features for a given problem

Most of the exam domains have focused on at least one phase of the ML pipeline and, where applicable, the AWS services needed to perform related tasks and create relevant solutions. For instance, **AWS Glue, Amazon EMR, and Amazon Kinesis were discussed as services used for data ingestion and transformation, while Amazon SageMaker was discussed as the service to use for model building, training, tuning, and evaluation.**

This particular subdomain, however, pivots away from the ML pipeline and focuses more generally on the entire ecosystem of AWS ML services and their use cases.

The stack for Amazon machine learning has three tiers:

![](https://github.com/DanialArab/images/blob/main/AWS_ML_Specialty_Exam/AWS%20ML%20three%20tiers.PNG)

**ML frameworks + infrastructure**

The bottom tier of the stack is for expert ML practitioners who work at the framework level. To work with these frameworks, you are comfortable building, training, tuning, and deploying ML models on the metal, so to speak. ML frameworks are the foundation from which innovation in ML is designed. The focus here is on making it easier for you to connect more broadly to the AWS ecosystem, whether that’s about pulling in IoT data from AWS IOT Greengrass, accessing state-of-the art chips (P3), or leveraging elastic inference. 


**The vast majority of deep learning and ML in the cloud is done on P3 instances in AWS**. You can use whichever ML deep learning framework you like, but some popular options are TensorFlow, MXNet, and PyTorch, which are all supported on AWS.

**ML services**

While there is a lot of activity at this bottom layer, the reality is that there just aren't that many expert ML practitioners out there. 

That’s why the second tier on the stack, platform services, was created. At the heart of this tier is Amazon SageMaker, which we’ve discussed. While ML can provide tremendous business value, today the process for authoring, training, and deploying ML models has many challenges. Why? Because collecting, cleaning, and formatting the training data can be time-consuming, particularly if you’re not using the latest tools. 

Once the training data set is created, you need to ensure that your algorithms and compute environments can quickly handle the scale of the data needed. Simply figuring out how to train increasingly complex models on increasingly larger data sets can be a blocker, and companies that frequently train models often have dedicated teams just to manage the training environments. 

Once you’re ready to move to production, a new set of challenges can come up. Often, the person developing the model hands it off to another team to begin the tedious process of figuring out how to run the model at scale. This involves an entirely different set of computer science challenges related to efficiently operating high-scale distributed systems. If you don’t do this routinely, it can be a slow and cumbersome process. Amazon SageMaker was designed to address many of these fundamental challenges.

**AI services**

AWS services in the top tier are for customers who really don't want to deal with building and training their ML models. All of that has been abstracted away, leaving you with easy-to-use services designed to help you deal with common ML problems in various domains, like computer vision, NLP, and time series forecasting.  

Topics related to this subdomain: Here are some topics you may want to study for more in-depth information related to this subdomain:

+ Amazon SageMaker Spark containers
+ Amazon SageMaker build your own containers
+ Amazon AI services 

   + Amazon Translate
   + Amazon Lex
   + Amazon Polly
   + Amazon Transcribe
   + Amazon Rekognition
   + Amazon Comprehend

#### Domain 4.3: Apply Basic AWS security practices to ML solutions

For any solution you build, including an ML solution, you need to make sure it and your data are secure. Applying security practices to your ML solutions in the cloud is what this subdomain is centered around.

This section will provide you with an overview of those security practices by walking you through an example involving Amazon SageMaker. 

##### Security is intrinsically built into Amazon SageMaker

Security is intrinsically built in to Amazon SageMaker. With training data, Amazon SageMaker gets data from Amazon S3, passes that data to the training job environment, and then passes the generated model back to Amazon S3. This is done in the customer’s account and is not saved in the Amazon SageMaker managed account. If you want to deploy the model, it is loaded into instances that are serving the model so that you can call the endpoint for prediction. 

To keep this process secure, Amazon SageMaker supports IAM role-based access to secure your artifacts in Amazon S3, where you can set different roles for different parts of the process. For instance, a certain data scientist can have access to PII information in the raw data bucket, but the DevOps engineer only has access to the trained model itself. This approach helps you restrict access to the user(s) who need it. For the data scientist, you can use a notebook execution role for creating and deleting notebooks, and a training job execution role to run the training jobs. 

##### You can launch an Amazon SageMaker instance in a customer-managed VPC

When you create an Amazon SageMaker notebook instance, you can launch the instance with or without your Virtual Private Cloud (VPC) attached. When launched with your VPC attached, the notebook can either be configured with optional direct internet access.

You specify your private VPC configuration when you create a model by specifying subnets and security groups. When you specify the subnets and security groups, Amazon SageMaker creates elastic network interfaces that are associated with your security groups in one of the subnets. 

Network interfaces allow your model containers to connect to resources in your VPC. For instances with direct internet access, Amazon SageMaker provides a network interface that allows for the notebook to talk to the internet through a VPC managed by the service. If you disable direct internet access, the notebook instance won't be able to train or host models unless your VPC has an interface endpoint (PrivateLink) or a NAT gateway and your security groups allow outbound connections.

##### Amazon SageMaker also encrypts data at rest

Along with IAM roles to prevent unwanted access, Amazon SageMaker also encrypts data at rest with either AWS Key Management Service (AWS KMS) or a transient key if the key isn’t provided and in transit with TLS 1.2 encryption for the all other communication. Users can connect to the notebook instances using an AWS SigV4 authentication so that any connection remains secure. Any API call you make is executed over an SSL connection.

##### You can use encrypted Amazon S3 buckets for model artifacts and data

Similar to encrypting data in Amazon SageMaker, you can use encrypted Amazon S3 buckets for model artifacts and data, and pass an AWS KMS key to Amazon SageMaker notebooks, training jobs, hyperparameter tuning jobs, batch transform jobs, and endpoints, to encrypt the attached ML storage volume. If you do not specify an AWS KMS key, Amazon SageMaker encrypts storage volumes with a transient key. A transient key is discarded immediately after it is used to encrypt the storage volume.

AWS KMS gives you centralized control over the encryption keys used to protect your data. You can create, import, rotate, disable, delete, define usage policies for, and audit the use of encryption keys used to encrypt your data. You specify a KMS key ID when you create Amazon SageMaker notebook instances, training jobs, or endpoints. 

The attached ML storage volumes are encrypted with the specified key. You can specify an output Amazon S3 bucket for training jobs that is also encrypted with a key managed with AWS KMS, and pass in the KMS key ID for storing the model artifacts in that output S3 bucket.

##### There are two ways to use AWS KMS with Amazon S3

You can protect data at rest in Amazon S3 by using three different modes of server-side encryption:
+ SSE-KMS: requires that AWS manage the data key, but you manage the customer master key in AWS KMS 
+ SSE-C: requires that you manage the encryption key 
+ SSE-S3: requires Amazon S3 to manage the data and master encryption keys 

##### Below is a summary of security features integrated with Amazon SageMaker

+ Authentication
   + IAM federation

+ Gaining insight
   + Restrict access by IAM policy and condition keys

+ Audit
   + API logs to AWS CloudTrail - exception of InvokeEndpoint

+ Data protection at rest
   + AWS KMS-based encryption for:

      + Notebooks
      + Training jobs
      + Amazon S3 location to store modelsEndpoint

+ Data protection at motion
   + HTTPS for: 

      + API/console
      + Notebooks
      + VPC-enabled
      + Interface endpoint
      + Limit by IPTraining jobs/endpoints

+ Compliance programs
   + PCI DSS
   + HIPAA-eligible with BAA
   + ISO

Topics related to this subdomain: Here are some topics you may want to study for more in-depth information related to this subdomain:

+ Security on Amazon SageMaker
+ Infrastructure security on Amazon SageMaker
+ What is a:

   + VPC
   + Security group
   + NAT gateway
   + Internet gateway
   
+ AWS Key Management Service (AWS KMS)
+ AWS Identity and Access Management (IAM)

#### Domain 4.4: Deploy and operationalize ML solutions

The ML model you develop is one component in a larger software ecosystem. All the usual software engineering and management practices must still be applied, including security, logging and monitoring, task management, API versioning, and so on. This ecosystem must be managed using cloud and software engineering practices, including:

   + End-to-end and A/B testing
   + API versioning, if multiple versions of the model are used
   + Reliability and failover
   + Ongoing maintenance
   + Cloud infrastructure best practices, such as continuous integration/continuous deployment (CI/CD)

##### Deploying a model using Amazon SageMaker hosting services is a three-step process

###### Create a model in Amazon SageMaker
You need:

   + The Amazon S3 path where the model artifacts are stored 
   + The Docker registry path for the image that contains the inference code 
   + A name that you can use for subsequent deployment steps

###### Create an endpoint configuration for an HTTPS endpoint
You need:

   + The name of one or more models in production variants
   +The ML compute instances that you want Amazon SageMaker to launch to host each production variant. When hosting models in production, you can configure the endpoint to elastically scale the deployed ML compute instances. For each production variant, you specify the number of ML compute instances that you want to deploy. When you specify two or more instances, Amazon SageMaker launches them in multiple Availability Zones. This ensures continuous availability. Amazon SageMaker manages deploying the instances.


## References:

AWS Certified Machine Learning Specialty 2023 - Hands On!

Process Model: CRISP-DM on the AWS Stack (https://explore.skillbuilder.aws/learn/course/external/view/elearning/14575/process-model-crisp-dm-on-the-aws-stack-thai) 

Exam Readiness: AWS Certified Machine Learning - Specialty

Exam Prep Official Practice Question Set: AWS Certified Machine Learning - Specialty (MLS-C01 - English)
