# Facebook-Friend-Recommendation

## Problem Statement:
### Imagine Users in social media as Nodes (dots)  and friendship/relationship between users as edges/Links

![image](https://user-images.githubusercontent.com/61958476/117765284-56979c80-b24b-11eb-9367-d317774c1e22.png)

### Here Each user is Node/Vertex & Line connected to it is edge/link   (Here Edges are divided in 2 Categories)
### Directed and Undirected 
#### Here in given image lets, User 0 follows User 3 , User 4, User 1 , but they dont follow back so such type of edge/link is Known as directed edge (mostly we are going to work on this type of data)

### Data Overview
Taken data from facebook's recruting challenge on kaggle https://www.kaggle.com/c/FacebookRecruiting   
    - Data columns (total 2 columns):  
    - source_node         int64  
    - destination_node    int64  
 
## type of Machine Learning Problem:

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
