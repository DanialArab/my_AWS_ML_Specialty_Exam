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

#### create data repositories for ML

+ You need a way to store your data in a **centralized repository**. **Data lake** is a key solution to this challenge. With a data lake, you can store structured and unstructured data.
+ AWS Lake Formation is your data lake solution, and Amazon S3 is the preferred storage option for data science processing on AWS
+ Use Amazon S3 storage classes to reduce the cost of data storage.
+ Amazon S3 is integrated with Amazon SageMaker to store your training data and model training output.
+ However, Amazon S3 isn't your only storage solution for model training. 
+ Amazon FSx for Lustre: When your training data is already in Amazon S3 and you plan to run training jobs several times using different algorithms and parameters, consider using Amazon FSx for Lustre, a file system service. FSx for Lustre speeds up your training jobs by serving your Amazon S3 data to Amazon SageMaker at high speeds. The first time you run a training job, FSx for Lustre automatically copies data from Amazon S3 and makes it available to Amazon SageMaker. You can use the same Amazon FSx file system for subsequent iterations of training jobs, preventing repeated downloads of common Amazon S3 objects.
+ Amazon S3 with Amazon EFS: Alternatively, if your training data is already in Amazon Elastic File System (Amazon EFS), we recommend using that as your training data source. Amazon EFS has the benefit of directly launching your training jobs from the service without the need for data movement, resulting in faster training start times. This is often the case in environments where data scientists have home directories in Amazon EFS and are quickly iterating on their models by bringing in new data, sharing data with colleagues, and experimenting with including different fields or labels in their dataset. For example, a data scientist can use a Jupyter notebook to do initial cleansing on a training set, launch a training job from Amazon SageMaker, then use their notebook to drop a column and re-launch the training job, comparing the resulting models to see which works better.
+ When choosing a file system, take into consideration the training load time: The table below shows an example of some different file systems and the relative rate that they can transfer images to a compute cluster. This performance is only a single measurement and is only a suggestion for which file systems you could use for a given workload. The specifics of a given workload might change these results.

![](https://github.com/DanialArab/images/blob/main/AWS_ML_Specialty_Exam/comparison%20of%20training%20load%20time.PNG)

## References:

AWS Certified Machine Learning Specialty 2023 - Hands On!

Process Model: CRISP-DM on the AWS Stack (https://explore.skillbuilder.aws/learn/course/external/view/elearning/14575/process-model-crisp-dm-on-the-aws-stack-thai) 

Exam Readiness: AWS Certified Machine Learning - Specialty

Exam Prep Official Practice Question Set: AWS Certified Machine Learning - Specialty (MLS-C01 - English)
