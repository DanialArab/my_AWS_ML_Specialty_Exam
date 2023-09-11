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

some points:
+ You need a way to store your data in a **centralized repository**. **Data lake** is a key solution to this challenge. With a data lake, you can store structured and unstructured data.
+ AWS Lake Formation is your data lake solution, and Amazon S3 is the preferred storage option for data science processing on AWS
+ Use Amazon S3 storage classes to reduce the cost of data storage.
+ 




## References:

AWS Certified Machine Learning Specialty 2023 - Hands On!

Process Model: CRISP-DM on the AWS Stack (https://explore.skillbuilder.aws/learn/course/external/view/elearning/14575/process-model-crisp-dm-on-the-aws-stack-thai) 

Exam Readiness: AWS Certified Machine Learning - Specialty

Exam Prep Official Practice Question Set: AWS Certified Machine Learning - Specialty (MLS-C01 - English)
