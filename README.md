# Facebook-Friend-Recommendation

## 1. Problem Statement:
### Imagine Users in social media as Nodes (dots)  and friendship/relationship between users as edges/Links

![image](https://user-images.githubusercontent.com/61958476/117765284-56979c80-b24b-11eb-9367-d317774c1e22.png)

### Here Each user is Node/Vertex & Line connected to it is edge/link   (Here Edges are divided in 2 Categories)
### Directed and Undirected 
#### Here in given image lets, User 0 follows User 3 , User 4, User 1 , but they dont follow back so such type of edge/link is Known as directed edge (mostly we are going to work on this type of data)

### 1.2 Data Overview
Taken data from facebook's recruting challenge on kaggle https://www.kaggle.com/c/FacebookRecruiting   
    - Data columns (total 2 columns):  
    - source_node         int64  
    - destination_node    int64  
 
## 1.3 Type of Machine Learning Problem:

### Case 1
    - If user i and User j  have edge between them 
    - ( 1. either ui is following uj)
    -  (2. or uj is following ui)
    -  (3. Both follow each other)
So if any of these cases staisfies  Output (Yi = 1)
### Case 2
else:
    Yi = 0
### This is Binary Classification problem

## 1.4 How to featurize this data:
![image](https://user-images.githubusercontent.com/61958476/117766973-fa824780-b24d-11eb-88dd-5ac30fd38739.png)

### Task predict if there is a edge between [User 14 and User 15]
U14 - {U16,U17,U18}  
U15 - {U17,U18,U19}  

#### Out of these set of vertices Common vertices are (u17,u18) so as there are higher number of common vertices between them chances of having edge between U14 and U15 can be high 
### Note : This is one of the featurization technique which can help us there might be more to explore as we go forward

## 2. Objective and Constraints
### A: No Low Latency is Required 
---> i.e. we should not be in hurry to find results between them as we have to update our results on regular basis, where is User i and User j might not be friends till now but after few days they might follow each other so due to this we have to update our results on regualr basis but no need to be in hurry for fast results

### B (imp) : Find Probability of links which might be usefull to recommend highest prob links to user
----> eg: User 1 --------->>> (User 10  Prob(0.8), User 17  Prob(0.9), User 56  Prob(0.65), .........)
So these probablity scores might help us to find wheater User i might follow them or not
#### Here's the small example of Instagram follow recommendation interface to get some clarity:

![image](https://user-images.githubusercontent.com/61958476/117768437-0242eb80-b250-11eb-9709-f02f448100d8.png)

## 3. Exploratory Data Analysis (Part 1) Basic Stats and Graph Visulaization
### Basic Overviw:
            --if not os.path.isfile('data/after_eda/train_woheader.csv'):   # If this file is not present then do this below steps
                    traincsv = pd.read_csv("data/tran.csv")
                    print(traincsv[traincsv.isna().any(1)])
                    print(traincsv.info())
                    print("No of duplicates",sum(traincsv.duplicated()))
                    traincsv.to_csv('data/after_eda/train_woheader.csv',header=False,index=False)
              else:
                    g = nx.read_edgelist('data/after_eda/train_woheader.csv',delimiter = ",",create_using = nx.DiGraph(),nodetype = int )
                    print(nx.info(g))




