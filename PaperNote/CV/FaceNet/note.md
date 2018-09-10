#FaceNet: AUniﬁedEmbeddingforFaceRecognitionandClustering

>The main contribution of this paper is** triplet loss**

# Abstraction

> present facenet directly learns a mapping from face image to a compact Euclidean space where distance directly correspond to a measure of face similarity.
> 
> method uses a deep convolutional network trained to directly optimize the embedding itself, rather than an intermediate bottleneck layer as in previous deep learning approaches.

# 1 Introduction

>Once embedding space has been produced, aforementioned tasks become straight-forward: 
>- face verification simply involves **thresholding** the distance between the two embeded vectors
>- recognition becomes a k-NN classification problem
>- clustering can be achieved using off-the-shelf techniques such as k-means or agglomerative clustering

