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

>previous way of face recognition based on deep networks use classifcation layers, trained over a set of known face identities and then take an intermediate bottleneck layer as a representation used to generalize recognition beyound the set of identities used intraing.
>However this is indirectness and ineffciency:
>one has to hope that the bottleneck representation generalizes well to new faces
>using a bottleneck layer the representation size per face is usually very large, some one use PCA, but this is linear transformation that can be easily learnt in one layer of network

>Choosing which **triplets** to use turns out to be very important for achieving good performance

## 2 Related work

This paper use 2 different deep network architectures:
- multiple interleaved layers of convolutions, non-linear activations, local response normalizations, and max pooling layers
- 