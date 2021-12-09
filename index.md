---
layout: home
title: "Home"
---


# FRCNN_trademark_similarity
*this is not being maintained now*

Using Faster RCNN and Mask RCNN, Implemented Multilabel classification and Image similarity.

Implemented by Custom trademark dataset & custom class for training

## Multi label classification
One of difficult trademark's feature is that it is figurative. It could be expressed like "the heart shaping cat", "the S shaping chicken", etc
And trademarks are classificated by ViennaCodes, but It couldn't explain the all image appearance.
So we need to make new classification table based on ViennaCodes.
We tried to contain the all feature but some are removed due to accuracy.
For example, the trademarks' viennacodes distribution is not flat because many trademark contain a Creature.
<img src="./images/dist.PNG" width="750px" height="250px" title="dist" alt="dist"></img>

So we define new rule for this and made a new classification table.

### result samples
<br><img src="./images/det1.jpg" width="250px" height="250px" title="det1" alt="det1"></img>
<img src="./images/det2.jpg" width="250px" height="250px" title="det2" alt="det2"></img>
<img src="./images/det3.jpg" width="250px" height="250px" title="det3" alt="det3"></img><br/>


## Image Similarity
We extracted the normal feature from efficientnet-b0 that pretrained by custom dataset and triplet loss.
And we mixed it to Multi label classification result, and calculated the Cosine Similarity from it's fully connected vectors.
So we can calculate image silmilarity score between 


### result samples
<br><img src="./images/sim1.PNG" width="400px" height="250px" title="sim1" alt="sim1"></img>
<img src="./images/sim2.PNG" width="400px" height="250px" title="sim2" alt="sim2"></img>
<img src="./images/sim3.PNG" width="400px" height="250px" title="sim3" alt="sim3"></img>
<img src="https://user-images.githubusercontent.com/32811724/142619840-5c51e3bb-9ec6-44ee-a3bc-63b35ebf0211.png" width="400px" height="250px"></img><br/>


## Similar tradeamark searching service
As an application of this repository, we made a Similar trademark seraching service.
![image](https://user-images.githubusercontent.com/32811724/142619590-b5fa63a2-6b15-4720-9d99-cbd15fddc3c7.png)
![image](https://user-images.githubusercontent.com/32811724/142619604-3c5aeaad-fa1d-4b18-b169-804314554c2e.png)

