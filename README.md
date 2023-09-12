# Exam date: ‎Sunday, November 26, 2023‎

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

HERE 


## References:

AWS Certified Machine Learning Specialty 2023 - Hands On!

Process Model: CRISP-DM on the AWS Stack (https://explore.skillbuilder.aws/learn/course/external/view/elearning/14575/process-model-crisp-dm-on-the-aws-stack-thai) 

Exam Readiness: AWS Certified Machine Learning - Specialty

Exam Prep Official Practice Question Set: AWS Certified Machine Learning - Specialty (MLS-C01 - English)
