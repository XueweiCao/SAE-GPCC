# Stacked Autoencoder-based Compression of Point Cloud Geometry

Xuewei Cao, Wenbiao Zhou, Shuyu Yan, Genpei Liu

> **Abstract:** *Point clouds have gained widespread application in various fields, but their high resolution often results in large data volumes, posing challenges for storage, transmission, and processing.  Traditional 2D image or video compression methods are unsuitable due to the spatial irregularity and sparseness of point clouds.  Inspired by the effectiveness of autoencoders in visual analysis tasks and image compression, this paper proposes a novel stacked autoencoder-based geometry compression method for point clouds.  By transforming point clouds into Morton codes using a linear octree structure and further encoding them into integer sequences, the proposed method leverages stacked autoencoders to reduce the dimensions of these sequences, achieving both high reconstruction quality and high compression ratios.  Experimental results demonstrate that our method outperforms TMC13, the testing model proposed by MPEG, especially for small-size point clouds.  By increasing the coding depth of the linear octree, our approach can even achieve lossless compression results, showcasing its potential as an effective geometry compression technique for point clouds.* 
<hr />

## Compression Architecture
