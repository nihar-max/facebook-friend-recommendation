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
              if not os.path.isfile('data/after_eda/train_woheader.csv'):   # If this file is not present then do this below steps
                    traincsv = pd.read_csv("data/tran.csv")
                    print(traincsv[traincsv.isna().any(1)])
                    print(traincsv.info())
                    print("No of duplicates",sum(traincsv.duplicated()))
                    traincsv.to_csv('data/after_eda/train_woheader.csv',header=False,index=False)
              else:
                    g = nx.read_edgelist('data/after_eda/train_woheader.csv',delimiter = ",",create_using = nx.DiGraph(),nodetype = int )
                    print(nx.info(g))
                    
 ![image](https://user-images.githubusercontent.com/61958476/117770467-94e48a00-b252-11eb-90d4-6782fb3bc124.png)
 
### Basic Visualization using Subset 
                     if not os.path.isfile("train_woheader_sample.csv"):
                    # Taking only first 30 rows with no headers
                        pd.read_csv("data/train.csv",nrows = 30).to_csv('train_woheader_sample.csv',header = False,index = False)
    
                    subgraph = nx.read_edgelist('train_woheader_sample.csv',delimiter=',',create_using=nx.DiGraph(),nodetype=int)

                     pos=nx.spring_layout(subgraph)

                    nx.draw(subgraph,pos,node_color='#A0CBE2',edge_color='#00bb5e',width=1,edge_cmap=plt.cm.Blues,with_labels=True)
                    plt.savefig("graph_sample.pdf")
                     print(nx.info(subgraph))

![image](https://user-images.githubusercontent.com/61958476/117770753-f147a980-b252-11eb-857a-36fef4c516c6.png)

### Q. Total Number of Followers for each person? (In_Degree) ?
        indegree_dist = list(dict(g.in_degree()).values())
        indegree_dist.sort()
        plt.figure(figsize=(10,6))
        plt.plot(indegree_dist)
        plt.xlabel('Index No')
        plt.ylabel('No Of Followers')
        plt.show()
![image](https://user-images.githubusercontent.com/61958476/117931077-5d421480-b31c-11eb-99bf-f7114867c218.png)


##### We can see that From 0 to 1.6 million (Index) there are very less number of users less than 100 and then suddenly there is massive growth where one of the index has more than 500 followers

#### Calculate percentile 
     for i in range(10,110,10):
         print(99+(i/100),'percentile value is',np.percentile(indegree_dist,99+(i/100)))
         

![image](https://user-images.githubusercontent.com/61958476/117930444-9e85f480-b31b-11eb-9d86-248ce52570dd.png)

##### Conclusion: Over 99.9% of people are having less than 112 Followers


### Q. No of people each person is following (Out_Degree) ?
    outdegree_dist = list(dict(g.out_degree()).values())
    outdegree_dist.sort()
    plt.figure(figsize=(10,6))
    plt.plot(outdegree_dist)
    plt.xlabel('Index No')
    plt.ylabel('No Of Followers')
    plt.show()
    
![image](https://user-images.githubusercontent.com/61958476/117930914-2f5cd000-b31c-11eb-94dd-b64f3bc4076d.png)
#### Calculate percentile 
![image](https://user-images.githubusercontent.com/61958476/117930658-e573ea00-b31b-11eb-9a51-d7db7f34dcde.png)

##### Conclusion: Over 99.9% of people are following less than 123 peoples

#### Observations: So the key take away from all this analysis is that there are only few people who have max following + followers rest all have very few followers & following so in this case we have high Outliers

## 4. Featurizartion 
### Note: After doing all the basic analysis and Train Test split we have source node , destination node and output so from that we cant apply ML models 
### So we have to add some more relevent features

## 4.1 Similarity measure 
### 1. jaccard Distance 
![image](https://user-images.githubusercontent.com/61958476/118234178-9f03c400-b4b0-11eb-9aff-764ed5e5375d.png)

#### How Jaccard Distance works?
![image](https://user-images.githubusercontent.com/61958476/118234013-5cda8280-b4b0-11eb-817f-df8140e462ce.png)

#### Lets: Create 2 Sets X & Y
        X is set of followers User 0 has -- > {u2, u3, u4}
        Y is set of followers User 1 has -- > {u3, u4, u5}
    
Jaccard distance for X & y = 2/4 

       Note: Higher the Jaccard Distance b/w U0 and U1 chances of having edge b/w them is higher
       
### 2. Cosine Simmilarity
![image](https://user-images.githubusercontent.com/61958476/118234419-ed18c780-b4b0-11eb-85c8-5f103045f7ba.png)

### 3. Page Rank
![image](https://user-images.githubusercontent.com/61958476/118247711-1fcabc00-b4c1-11eb-91fa-e0a042309164.png)
##### How page rank works?
    Given a Directed graph--- page rank algorithm --- > gives each vertex/node (ui) ---> Score 
    That score represent how imp the vertex is

### 4. Shortest Path
![image](https://user-images.githubusercontent.com/61958476/118349495-a38eb200-b56e-11eb-9de4-f084b20d1f19.png)
##### Shortest path from Ui to Uj:
    From Ui to Uj ---> [2,Uj]     length = 2
    From Ui to Uj ---> [3,4,5,Uj] length = 4
##### So we will take shortest path from Ui to Uj with length = 2   

### 5 Adamic/ Adar Index
![image](https://user-images.githubusercontent.com/61958476/118351071-6844b100-b577-11eb-90de-ae66c68ca5f0.png)

##### Source: https://en.wikipedia.org/wiki/Adamic/Adar_in


## Apply Machine Learning Model for classification:
### Model used : Random forest
Train f1 score 0.9422858984622743

Test f1 score 0.9272375772768796
### Feature importance :
![image](https://user-images.githubusercontent.com/61958476/118778561-dbb42e80-b8a7-11eb-8857-deecc0683810.png)

# Conclusion:
So we the main task of this Case study was featurization more than modelling, we have added many graph based features some of them using inbilt libraries called networkx and finally applied Random forest classifier with 2 hyper parameters and also observed feature importance on it

# Future Scope:
In future we can add some more graph based features like Preferentail attachment with followers & followees (link: http://be.amazd.com/link-prediction/) and also use ensemble models like XGBoost. 







