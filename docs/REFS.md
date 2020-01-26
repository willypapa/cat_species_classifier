References - Computer Vision for Earth Observation
======================

# General

Below is a list of general resources for deep learning (possibly in remote
sensing), which do not fall into a specific category or pertain to a certain
problem domain.

## Papers

### [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks][86]

Describes a scaling strategy to maximally improve convnet performance at fixed computational resources.
Involves scaling filters, depth and input resolution simultaneously by empirical coefficients.

### [Deep Learning in Remote Sensing: A Review][17]

This covers a broad range of topics beginning with elementary machine learning,
but moving on to EO applications in:

- Hyperspectral imaging
- Synthetic aperture radar (SAR)
- High-resolution satellite images
  - Scene classification
  - Object detection
  - Change detection
- Multimodal data fusion
- 3D reconstruction

### [Deep learning in remote sensing applications: A meta-analysis and review][61]

*TODO*

### [Learning to Reweight Examples for Robust Deep Learning][78]

Seems like a promising method to reweight biased datasets that seriously
outperforms standard techniques like resampling and inverse frequency weighting.


### [Survey of Deep-Learning Approaches for Remote Sensing Observation Enhancement][81]

Comprehensive review of deep-learning methods for the enhancement of remote
sensing observations, focusing on critical tasks including single and
multi-band super-resolution, denoising, restoration, pan-sharpening, and
fusion, among others.

### [Learning to Zoom: a Saliency-Based Sampling Layer for Neural Networks][82]

Proposes a trainable saliency-based sampling layer that selectively upsamples
regions of low-resolution input images trained with access to higher resolution
originals.

### [PSI-CNN: A Pyramid-Based Scale-Invariant CNN Architecture for Face Recognition Robust to Various Image Resolutions][79]

Makes attempt to circumvent dependency on input image resolution on CNN's
performance by proposing PSI-CNN, a generic pyramid-based scale-invariant CNN
architecture which additionally extracts untrained feature maps across multiple
image resolutions, allowing the network to learn scale-independent
information and improving the recognition performance on low resolution images.

Experimental results on the LFW dataset and a CCTV database show PSI-CNN
consistently outperforming the widely-adopted VGG face model in terms of face
matching accuracy.

*looks like this applies during training, downsampling and fusing feature
information from different downsample factors to give tolerance to scale
variation*

### [Deep Residual Learning for Image Recognition][85]

Introduces residual blocks, which learn residual functions with respect to the
layer inputs rather than unreferenced functions. Leads to easier optimisation,
accuracy gains and possibility of much deeper networks.

## General Datasets

### [Zeebruges, or the Data Fusion Contest 2015 dataset][75]

Unlabelled, high quality fusion data including orthophotos and LiDAR point cloud.

LiDAR data at about 10-cm point spacing
Color orthophoto at 5-cm spatial resolution
7 separate tiles - each with a 10000 × 10000 px portion of color orthophoto
(GeoTIFF, RGB), a max 5000 × 5000 px portion of the Digital Surface Model
(GeoTIFF, floating point), and a text file listing the 3D points in XYZI format
[containing X (latitude), Y (longitude), Z (elevation), and I (LiDAR intensity)
information].

See also: [Extremely High Resolution LiDAR and Optical Data: A Data Fusion Challenge][49]

### [GRSS Data and Algorithm Standard Evaluation (DASE) Website][50]

A compendium of data sources and associated algorithm performance league tables
on certain challenges. Covers many of the specific datasets summarised below.

### [Net2Net: Accelerating Learning Via Knowledge Transfer][12]

Promising general transfer learning reference.

## Slides

### [An introduction to  Remote Sensing Analysis  with applications on land cover mapping][80]

A (very recent) blitz through various DL applications including time series
image analysis, Recurrent-CNNs etc.

## Articles

### [Deep learning for remote sensing - an introduction][11]

## Videos:

### [Detection and Segmentation][3]

-------------------------------------------------------------------------------

# Semantic Segmentation

Semantic Segmentation is the problem of assigning labels to every pixel in an
image. These labels correspond to a fixed number of classifications. There is
no notion of distinct objects.

Typical training data are pairs of matching images and masks. A mask is an array
of classification labels with exactly the dimensions of its corresponding image.

For more information on the machine-learning techniques used, see
[Algorithms](./ALGORITHMS.md).

## Datasets

### [ISPRS Vaihingen Dataset][57]

Semantic segmentation for 2D RGB orthophotos of buildings, trees, cars and other
classifications.

Sliding windows with overlap used to derive input images.

9 cm resolution.

**Determine size of dataset**

### [ISPRS Potsdam Dataset][73]

As with Vaihingen, 2D RGB orthophotos.

5 cm resolution.

**Determine size of dataset**

### [DeepGlobe Dataset][45]

Semantic segmentation for road, building and land cover classification.

**Details unclear - I have submitted request for dataset at:**
https://competitions.codalab.org/forums/15284/3198/.

### [Inria Aerial Image Labeling Dataset][33]

Semantic segmentation for building/not building classification.

810 km² coverage of aerial orthorectified color imagery with a spatial
resolution of 0.3 m

Dissimilar urban settlements.

### [Dstl Satellite Imagery Feature Detection][44]

Instance segmentation of buildings, roads and 8 other classifications.

Multispectral (3- and 16-band) GeoTiff input images (WorldView 3 sensor).

1km x 1km satellite images.

**resolution and sample size?**

Example notebook of mask prediction:
[Full pipeline demo: poly -> pixels -> ML -> poly][74]

### [Massachusetts Buildings Dataset][35]

Semantic segmentation of building/road classes.

51 aerial images of the Boston area, 1500×1500 pixels for an area of 2.25 km²
each, totalling 340 km².

### [The SARptical Dataset][77]

Semantic segmentation on SAR-derived point clouds using optical images.
10k pairs of matched SAR + optical images.

*Not especially relevant for our purposes at the moment, but super cool.*

See also:
[SARptical paper][51]


## Papers

### [Generalized Overlap Measures for Evaluation andValidation in Medical Image Analysis][90]

Measures  of  overlap  of  labelled  regions  of  images,such as the Dice and Tanimoto coefficients, have been extensivelyused to evaluate image registration and segmentation algorithms.Modern  studies  can  include  multiple  labels  defined  on  multipleimages  yet  most  evaluation  schemes  report  one  overlap  perlabelled  region,  simply  averaged  over  multiple  images.  In  thispaper,  common  overlap  measures  are  generalized  to  measurethe  total  overlap  of  ensembles  of  labels  defined  on  multiple  testimages  and  account  for  fractional  labels  using  fuzzy  set  theory.This framework allows a single “figure-of-merit” to be reportedwhich summarises the results of a complex experiment by imagepair, by label or overall. A complementary measure of error, theoverlap  distance,  is  defined  which  captures  the  spatial  extent  ofthe nonoverlapping part and is related to the Hausdorff distancecomputed on grey level images. The generalized overlap measuresare  validated  on  synthetic  images  for  which  the  overlap  can  becomputed analytically and used as similarity measures in nonrigidregistration  of  three-dimensional  magnetic  resonance  imaging(MRI)  brain  images.  Finally,  a  pragmatic  segmentation  groundtruth  is  constructed  by  registering  a  magnetic  resonance  atlasbrain to 20 individual scans, and used with the overlap measuresto evaluate publicly available brain segmentation algorithms.

### [Generalised Dice overlap as a deep learning lossfunction for highly unbalanced segmentations][89]

Deep-learning  has  proved  in  recent  years  to  be  a  powerfultool  for  image  analysis  and  is  now  widely  used  to  segment  both  2Dand 3D medical images. Deep-learning segmentation frameworks rely notonly on the choice of network architecture but also on the choice of lossfunction.  When  the  segmentation  process  targets  rare  observations,  asevere class imbalance is likely to occur between candidate labels, thusresulting  in  sub-optimal  performance.  In  order  to  mitigate  this  issue,strategies  such  as  the  weighted  cross-entropy  function,  the  sensitivityfunction or the Dice loss function, have been proposed. In this work, weinvestigate the behavior of these loss functions and their sensitivity tolearning rate tuning in the presence of different rates of label imbalanceacross 2D and 3D segmentation tasks. We also propose to use the classre-balancing properties of the Generalized Dice overlap, a known metricfor segmentation assessment, as a robust and accurate deep-learning lossfunction for unbalanced tasks

### [Automatic Building Extraction in Aerial Scenes Using Convolutional Networks][88]

Automatic building extraction from aerial and satel-lite  imagery  is  highly  challenging  due  to  extremely  large  vari-ations   of   building   appearances.   To   attack   this   problem,   wedesign a convolutional network with a final stage that integratesactivations  from  multiple  preceding  stages  for  pixel-wise  pre-diction,  and  introduce  the  signed  distance  function  of  buildingboundaries as the output representation, which has an enhancedrepresentation  power.  We  leverage  abundant  building  footprintdata  available  from  geographic  information  systems  (GIS)  tocompile  training  data.  The  trained  network  achieves  superiorperformance  on  datasets  that  are  significantly  larger  and  morecomplex  than  those  used  in  prior  work,  demonstrating  that  theproposed method provides a promising and scalable solution forautomating  this  labor-intensive  task.

### [Boundary Loss for Highly Unbalanced Segmentation][87]

Widely used loss functions for convolutional neural network (CNN) segmentation, e.g., Dice orcross-entropy, are based on integrals (summations) over the segmentation regions.  Unfortunately,for highly unbalanced segmentations, such regional losses have values that differ considerably –typically of several orders of magnitude – across segmentation classes, which may affect trainingperformance and stability. We propose aboundaryloss, which takes the form of a distance metricon the space of contours (or shapes),  not regions.   This can mitigate the difficulties of regionallosses in the context of highly unbalanced segmentation problems because it uses integrals over theboundary (interface) between regions instead of unbalanced integrals over regions. Furthermore, aboundary loss provides information that is complimentary to regional losses. Unfortunately, it is notstraightforward to represent the boundary points corresponding to the regional softmax outputs ofa CNN. Our boundary loss is inspired by discrete (graph-based) optimization techniques for com-puting gradient flows of curve evolution.  Following an integral approach for computing boundaryvariations, we express a non-symmetricL2distance on the space of shapes as a regional integral,which avoids completely local differential computations involving contour points.  This yields aboundary loss expressed with the regional softmax probability outputs of the network, which canbe easily combined with standard regional losses and implemented with any existing deep net-work architecture for N-D segmentation. We report comprehensive evaluations on two benchmarkdatasets corresponding to difficult, highly unbalanced problems: the ischemic stroke lesion (ISLES)and white matter hyperintensities (WMH). Used in conjunction with the region-based generalizedDice loss (GDL), our boundary loss improves performance significantly compared to GDL alone,reaching up to 8% improvement in Dice score and 10% improvement in Hausdorff score.  It alsoyielded a more stable learning process. Our code is publicly available.

[github repository][https://github.com/LIVIAETS/surface-loss]

### [Road Extraction by Deep Residual U-Net][84]

**Probably very useful!**

A semantic segmentation neural network which combines the strengths of residual
learning and U-Net is proposed for road area extraction. The network is built
with residual units and has similar architecture to that of U-Net. The benefits
of this model is two-fold: first, residual units ease training of deep networks.
Second, the rich skip connections within the network could facilitate
information propagation, allowing  us to design networks with fewer parameters
however better performance. We test our network on a public road dataset and
compare it with U-Net and other two state of the art deep learning based road
extraction methods. The proposed approach outperforms all the comparing methods,
which demonstrates its superiority over recently developed state of the arts.

*Code available in* [Algorithms](./ALGORITHMS.md)

### [Semantic Segmentation of Urban Buildings from VHR Remote Sensing Imagery Using a Deep CNN (2019)][34]

**Potentially useful!**

A deep convolutional network architecture 'DeepResNet' is presented based on
UNet, which can perform semantic segmentation of urban buildings from VHR
imagery with higher accuracy than other SotA models
(FC/Seg/Deconv/U/ResU/DeepU-nets).

#### Inputs:

The proposed DeepResUnet was tested with aerial images with a spatial
resolution of 0.075 m.

#### Model structure:

The method contains two sub-networks: One is a cascade down-sampling network
for extracting feature maps of buildings from the VHR image, and the other is
an up-sampling network for reconstructing those extracted feature maps back to
the same size of the input VHR image. The deep residual learning approach was
adopted to facilitate training in order to alleviate the degradation problem
that often occurred in the model training process.

#### Results:

**Compared with the U-Net, the F1 score, Kappa coefficient and overall
accuracy of DeepResUnet were improved by 3.52%, 4.67% and 1.72%,
respectively**.

Moreover, the proposed DeepResUnet required much fewer parameters than the
U-Net, highlighting its significant improvement among U-Net applications.

#### Additional observations:

The inference time of DeepResUnet is slightly longer than that of the U-Net.

### [A Relation-Augmented Fully Convolutional Network for Semantic Segmentation in Aerial Scenes (2019)][76]

**Potentially useful**

Abstract:

*Most current semantic segmentation approaches fall back on deep convolutional
neural networks (CNNs). However, their use of convolution operations with local
receptive fields causes failures in modeling contextual spatial relations.
Prior works have sought to address this issue by using graphical models or
spatial propagation modules in networks. But such models often fail to capture
long-range spatial relationships between entities, which leads to spatially
fragmented predictions. Moreover, recent works have demonstrated that
channel-wise information also acts a pivotal part in CNNs. In this work, we
introduce two simple yet effective network units, the spatial relation module
and the channel relation module, to learn and reason about global relationships
between any two spatial positions or feature maps, and then produce
relation-augmented feature representations. The spatial and channel relation
modules are general and extensible, and can be used in a plug-and-play fashion
with the existing fully convolutional network (FCN) framework. We evaluate
relation module-equipped networks on semantic segmentation tasks using two
aerial image datasets, which fundamentally depend on long-range spatial
relational reasoning. The networks achieve very competitive results,
bringing significant improvements over baselines.*

### [Semantic Segmentation of EO Data Using Multi-model and Multi-scale deep networks (2016)][30]

This work investigates the use of deep fully convolutional neural networks
(DFCNN) for pixel-wise scene labelling of EO images. A variant of the SegNet
architecture is trained on remote sensing data over an urban area and different
strategies studied for performing accurate semantic segmentation. Our
contributions are the following:

1. A DFCNN is transferred efficiently from generic everyday images to remote
sensing images
2. A multi-kernel convolutional layer is introduced for fast aggregation of
predictions at multiple scales
3. Data fusion is performed from heterogeneous sensors (optical and laser) using
residual correction. The framework improves state-of-the-art accuracy on the
ISPRS Vaihingen 2D Semantic Labeling dataset.

#### Inputs

Uses the [ISPRS Vaihingen dataset][57].

#### Model Structure

- Initially, SegNet is used (encoder-decoder with VGG-16 pre-trained weights in
  encoder).
- Last layer is parallel multi-kernel (3,5,7) convolutions to aggregate spatial
  information at different scales.


#### Results

*SOTA accuracy, F1 score per-class (2016)*

## Articles

### [An overview of semantic image segmentation][83]

Excellent summary of algorithms and performance improvements gained from
different architectural features.

### [Semantic Segmentation Part 1: DeepLab-V3+][15]

Summarises the DeepLab-V3+ architecture, inference with code examples.

### [Semantic Segmentation Part 2: Training U-Net][16]

Summarises the U-Net architecture, training and inference with code examples.

### [Semantic Segmentation Part 4: State-of-the-Art][14]

Summarises pros and cons of architectures. DeepLab-V3+ for quick feasibility
checks. U-Net for production/small datasets.

### [deepsense.ai Satellite imagery semantic segmentation][9]

Describes application of modified U-Net on
[Dstl Satellite Imagery Feature Detection][44].

--------------------------------------------------------------------------------

# Object Detection

Object detection is a supervised learning problem which aims to predict the
*positions* (specified by bounding boxes) of objects within an image, and the
*probabilities* of each object of belonging to some of a fixed number of
classifications. The number of objects present does not have to be known a priori.

Training data consists of images with object categories and bounding boxes for
each instance of that category.

For more information on the machine-learning techniques used, see
[Algorithms](./ALGORITHMS.md).

See also: [Tensorflow Object Detection API][46]

## Datasets

### [xView Dataset][47]

Object detection dataset of 60 classes including buildings.

Satellite imagery, 1m examples in 60 classes, 30cm resolutionm 1415 km^2 area

Comes with pre-trained model.

See also:
[Deep Learning and Satellite Imagery: DIUx Xview Challenge][48]

### [Vehicle Detection in Aerial (VEDAI) Imagery Dataset][56]

Object detection dataset on aerial photographs of vehicles.

1.2k images with 12.5cm/px resolution. RGB and IR channels (separately). Many
angles and illumination conditions present.

9 classes of vehicle labelled, with average of 5.5 per image.

## Papers

### [Segment-before-Detect: Vehicle Detection and Classification through Semantic Segmentation of Aerial Images][25]

This paper uses semantic segmentation followed by connected component analysis
to approximate individual object detection.

### [Speed/accuracy trade-offs for modern convolutional object detectors][1]

### [A Survey on Object Detection in Optical Remote Sensing Images][2]

### [A modified faster R-CNN based on CFAR algorithm for SAR ship detection][59]

### [Learning Rotation-Invariant Convolutional Neural Networks for Object Detection in VHR Optical Remote Sensing Images][60]

### [Automatic Ship Detection of Remote Sensing Images from Google Earth in Complex Scenes Based on Multi-Scale Rotation Dense Feature Pyramid Networks][26]

### As yet unsorted, but potentially useful

[Spectral-spatial classification of hyperspectral imagery with a 3D CNN][27]
[Deep feature extraction and classification of hyperspectral images based on CNNs][28]
[Beyond RGB: Very High Resolution Urban Remote Sensing With Multimodal Deep Networks][29]
[TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation][42]
[Squeeze-and-Excitation Networks][43]
[Residual Hyperspectral Conv-Deconv Network in TensorFlow][52]
[Code for hyperspectral conv-deconv paper][55]
[Building instance classification using street view images][54]
[Technische Universität München Signal Processing in Earth Observation][53]
this looks promising for code examples
[Deep Learning for Spatio temporal Prediction][62]

## Articles

### [Object Detection with Deep Learning on Aerial Imagery][7]
### [Review: SSD — Single Shot Detector (Object Detection)][5]
### [Review: DSSD — Deconvolutional Single Shot Detector (Object Detection)][6]

-------------------------------------------------------------------------------

# Instance Segmentation

Instance Segmentation is a supervised learning problem which goes a step further
by aiming to predict - instead of bounding boxes - *pixel-wise masks* for each
object, along with the associated *classification probabilities*. Hybrid of
semantic segmentation and object detection. The number of objects present does
not have to be known a priori.

*Training data consists of images with object categories and pixel-wise masks
for each instance of that category.*

## Datasets

### [The SpaceNet Off-Nadir Dataset][36]

Instance segmentation of buildings.

Dataset has 120k building footprints over 665 km^2 in Atlanta from 27
associated WV-2 images. Multiple off-nadir angles supplied.

More information can be found at the following links:

[Introducing the SpaceNet Off-Nadir Imagery Dataset][37]
[Challenge results for SpaceNet Off-Nadir Imagery Dataset][38]
[The good and the bad in the SpaceNet Off-Nadir Building Footprint Extraction Challenge][39]
[Winning algorithms for the SpaceNet 4: Off-Nadir Building Footprint Detection Challenge][40]
[The SpaceNet Challenge Off-Nadir Buildings: Introducing the winners][41]

## Papers

## Articles

### [Semantic Segmentation Part 3: Transfer Learning with Mask R-CNN][4]

*TODO*

--------------------------------------------------------------------------------

# Change detection

Change detection is the problem of, given a series of co-registered images
displaced in time, identifying areas that have changed. This more complex than
analysing a simple difference of images, as some variable processes such as
illumination and weather effects are not considered true changes and algorithms
must learn to ignore these.

Supervised change detection algorithms typically use pairs of before and after
images, along with a mask identifying which pixels are considered to have changed.
More complex approaches might additionally distinguish the semantics of changes.

Current methods typically factorise into two kinds:

- *Post-classification comparison*: First classify the content of two temporally
different images of the same scene then compare to identify differences.
Inaccuracies arise due to errors in classification in either of the two images,
so an accurate classifier is required.

- *Difference image analysis*: Construct a DI to highlight differences. Further
analysis is then performed to determine nature of changes. CD results depend on
quality of produced DI. Atmospheric effects on reflectance values necessitate
techniques like radiometric correction, spectral differencing, rationing and
texture rationing.

For more information on the machine-learning techniques used, see
[Algorithms](./ALGORITHMS.md). **Actually, look below at papers atm.**

## Datasets

### [Onera Satellite Change Detection Dataset][32]
Pairs of Sentinel-2 multispectral images of urban areas, with pixelwise
*artificialisation, ie urban only* changes annotated as binary(no-)change).
10-60m resolution with 30 bands.

## Papers

### Common themes

A common theme in the literature is that deep-learning-based change detection
algorithms are not yet efficient and are limited by their training datasets.

Another is that object-based approaches to change detection (those which
segment the image's objects then compare them) are preferable to pixel- or
kernel-based methods, since the latter compare an insufficient amount of
information and cause irregular boundaries.

Yet another is that transfer-learning with networks pre-trained on large datasets
like ImageNet works successfully for CD architectures and is useful due to a
lack of remote-sensing training data.

### [Urban Change Detection for Multispectral Earth Observation Using CNNs][31]

Presents the [Onera Satellite Change Detection Dataset][32] (Sentinel-2 pre/post
change image pairs).

Studies performance of two network architectures on dataset, determining the
presence of urban changes pixel-wise.

#### Inputs:

Uses pairs of before/after 15x15xC image patches as input, predicting central
pixel binary probability for (urban) change/no-change. Idea is for network to
learn from context how to ignore changes due to natural processes.

For tractability full change maps generated by larger strides than each pixel,
then a 2D-Gaussian assumption about change of surrounding pixels.

Different resolution channels (10-60m) are handled by upsampling lower res.

#### Model structure:

Two networks are compared:

- **Early Fusion (EF)**: Concatenate the two image pairs as the first
step of the network.  The input of the network can then be seen as a single
patch of 15x15x2C, which is then processed by a series of seven convolutional
layers and two fully connected layers, where the last layer is a softmax layer
with two outputs associated with the classes of change and no change.

- **Siamese (Siam) network**: Process each of the patches in parallel by two
branches of four convolutional layers with shared weights,concatenating the
outputs and using two fully connected layers to obtain two output values as
before.

#### Results:
- **Early Fusion network generally better** than Siamese network
- **70-90% Accuracy**  (per class)
- **3->13 channels => <~5% accuracy improvement**

#### Additional observations:
- Transfer learning approaches limited by fact that pretrained models are trained
using RGB input, while sentinel-3 images include 13 bands.

- OSM unreliable for change detection since addition date != building date, older
maps unavailable.

### [High-resolution optical remote sensing imagery change detection through deep transfer learning (2018)][65]

Proposes **unsupervised** change-detection approach for optical images based on
CNNs which learn transferable features which are invariant to
contrast/illumination changes between tasks.

Uses pre-trained AlexNet, PlacesNet, Network in Network and VGG-16 models.

Describes two frameworks, a fast one and an accurate one.

#### Inputs:

Pairs of co-registered images separated in time.

#### Model structure:

In the preprocessing phase, a geometric registration and radiometric correction
are done.

A naive pixel-wise change map is used in an intermediate step. This is defined
by forming a 6-band images stacking the channel-wise difference and log-ratio of
the pair of temporally displaced images (which are insensitive to sun angle etc).

This image is PCA'd, selecting the most important components (linear
combinations of pixels) which are classified into two classes by using the
K-means algorithm to get a naive binary change map, CM0.

##### Fast Framework

Given the pair of co-registered, temporally displaced images:

- Construct two sets of hierarchical hyperfeatures, by passing the pair
through a pre-trained FCN.

- Each set is extracted, upsampled if necessary to the input image size,
and concatenated forming a very high dimensional set of hyperfeatures

- Dimensionality reduction must be applied to remove unrepresentative features
and make computation more tractable. This is accomplished by convolving the
hyperfeatures with CMO to get a vector of (hyperfeature_depth x n_layers)
features, which can be pruned down to K x n_layers, where K is a constant value
for all layers.

- K-means with two classes is used to separate changed from unchanged regions,
using the Euclidean distance between hyperfeatures as a metric.

- *fast* but *imprecise boundary delineation* due to unpooling operation used in  
upsampling

##### Accurate Framework

Given the same inputs:

*similar to above, but with 2D gaussian windowing to get multiple ROIs*

#### Results:

Both frameworks **outperform previous state-of-the-art methods** at identifying
changed regions when presented with an identified ground truth.

**NOTE: The previous "state-of-the-art" methods referred to are <= 2015 and
don't use CNNs.**

Pre-trained VGG-16 performs best, quoting kappa = 0.7 and 0.9 respectively for
fast and accurate frameworks respectively.

Overall error tends to be more evenly spread between FP and FN than previous
algorithms.

#### Additional observations:

Choice of data representation is key to success of CD algorithms.

### [A Deep Convolutional Coupling Network for Change Detection Based on Heterogeneous Optical and Radar Images (2016)][69]

Proposes an unsupervised deep convolutional coupling network for change
detection based on pairs of heterogeneous images acquired by optical sensors
and SAR images on different dates.

Heterogeneous images are captured by different types of sensors with different
characteristics, e.g., optical and radar sensors.
Homogeneous images are acquired by homogeneous sensors between which the
intensity of unchanged pixels are assumed to be linearly correlated.

#### Inputs:

Pairs of co-registered, *heterogeneous*, denoised images of the same scene
temporally displaced.

#### Model structure:

*Will fill this in if using heterogeneous images becomes important.*

#### Results:

*Will fill this in if using heterogeneous images becomes important.*

#### Additional observations:

Different sensors may capture distinct statistical properties of the same ground
 object, and thus, inconsistent representations of heterogeneous images may make
 change detection more difficult than using homogeneous images.

### [Unsupervised Change Detection in Satellite Images Using CNNs (Late 2018)][19]

**Promising!**

Proposes **semi-unsupervised** method for detecting changes between pairs of
temporally displaced images.

Comprised of CNN trained for *semantic segmentation* to extract compressed image
features, and generates an effective difference image from the feature map
information without explicitly training on difference images. Uses **U-Net**.

Uses [ISPRS Vaihingen dataset][57].

Aims to classify nature of change automatically with semantic segmentation,
while being noise resistant.

#### Inputs:

1. *Training phase*: Images with classification masks. Omitted if pre-trained
U-net available. Only buildings, immutable surfaces and background classes used.
2. *Inference phase*: Pairs of images of the same locations, temporally
displaced. 320x320 px x 3 channels.

#### Model structure:

- Feature maps are generated by the U-Net encoder for each image.

- A difference image is created for each feature map using a fixed algorithm
with a fixed threshold cutoff which determines whether the DI value is zero
or the value of the activation in the second (most recent) image.

- The five DIs are used by the decoder in the copy-and-concatenate operation

- The model outputs a semantically segmented visual rendering of the DIs.

- Threshold values for the difference-zeroing were determined by empirical
testing.

#### Results:

The success of the proposed change detection method  heavily relies on
the ability of the trained model to perform semantic segmentation.

*Test images were constructed manually by cutting and pasting over images,
and don't reflect change from natural illumination and weather processes.*

- **Semantic Segmentation**: Average classification **accuracy was 89.2%**

- **Change Detection**: The model was able to detect the location of the change
and classify it to the correct semantic class for the **91.2%** of test pairs.
*Accuracy declined as total number of pixels changed increased, owing to the
consequently different feature map activations => threshold value in
differencing includes unwanted changes*. Robust to gaussian noise.

Performance can potentially be improved by cutting the high-dimensional
satellite images into overlapping subsets, and combining change detection
signals generated by overlapping areas.

#### Additional observations:

A source of variance may arise from the translation and rotation of two images,
or from a difference in the angleat which the images were captured. This can
have a significanteffect on the accuracy of a DI unless accounted for.
[Orthorectification methods][70] can be used to compensate for sensor
orientation, and consist of geometric transformations to a mutual coordinate
system.

Testing the model on real, as opposed to simulated change, is a necessary step
to confirm that the results of this pilot study hold in more realistic
environments.

Additional experimentation should be done to test resistance to angle,
translation, and rotation differences between two images.  

### [Land Cover Change Detection via Semantic Segmentation][72]

#### Inputs:
#### Model structure:
#### Results:
#### Additional observations:

### [Change Detection between Multimodal Remote Sensing Data Using Siamese CNN][20]

#### Inputs:
#### Model structure:
#### Results:
#### Additional observations:

### [Convolutional Neural Network Features Based Change Detection in Satellite Images][68]

**This paper appears to be an older article from the same author as [65][65],
sharing many characteristics.**

### [Zoom out CNNs Features for Optical Remote Sensing Change Detection][58]

**This paper is unclear.**

Presents unsupervised change detection network based on an ImageNet-pretrained
CNN and superpixel (SLIC) segmentation technique. Uses QuickBird and Google
Earth images.

[Superpixels and SLIC][66]

#### Inputs:

Bi-temporal images.

#### Model structure:

Images are first segmented into superpixels using SLIC. PCA is applied to
extract three uncorrelated channels. Median filter applied to smooth image and
eliminate noise, followed by a bilateral filter to preserve the edges.

Each region subjected to three levels of zoom-out which are separately passed
through a pre-trained CNN, starting from the superpixel itself to regions around.

Each zoom passed through stack of convolutional layers.
Features extracted from each zoom level belonging to the same superpixel and
concatenated.

Concatenated features compared to get final change map according to some
dissimilarity measure

#### Results:

*will fill this in if I get round to decrypting the paper*

#### Additional observations:

Change detection divided into three broad classes depending on analysis unit:
pixel, kernel and object-based approaches. Latter more popular due to more
sophisticated feature extraction.

Object-based change detection split naturally into three categories:

  1. *Object overlay*: Segment one of the multi-temporal images and superimpose
     boundaries on second image. Compare each object region. Disadvantage is
     that gemoetry of objects imitates only one of the multi-temporal images.
  2. *Image-object direct comparison*: Segment each image separately, then
     compare objects from the same location. Introduces problem of 'silver'
     objects under inconsistent segmentations.
  3. *Multi-temporal image objects*: Images stacked and co-segmented in one
      step.


### [Change Detection in Synthetic Aperture Radar Images Based on DNNs (2015)][71]

According to [Unsupervised Change Detection in Satellite Images Using CNNs (Late 2018)][19]:

*"Make use of unsupervised feature learning performed by a CNN to learn the
representation of the relationship between two images. The CNN is then
fine-tuned withsupervised learning to learn the concepts of the changed and
the  unchanged  pixels. During supervised learning, a change detection map,
created by other means, is used to represent differences and similarities
between two images on a per-pixel level. Once the network is fully trained, it
is able to produce a change map directly from two given images without having
to generate a DI."

"While this approach achieves good results, it requires the creation of accurate
change maps by other means for each image pair prior to the learning process.
This makes the training of the network an expensive and time-consuming  process.
Change detection is also formulated as a binary classification problem, as it
only classifies pixels aschanged or not changed, and does not classify the
nature of the change, as the present study sets out to do."*


### [Automatic Change Detection in Synthetic Aperture Radar Images Based on PCANet][67]

*TODO*


--------------------------------------------------------------------------------

# Time-series/sequence (spatiotemporal) prediction

*TODO*

## Datasets

*TODO*

## Papers

### [Deep-STEP: A Deep Learning Approach for Spatiotemporal Prediction of Remote Sensing Data][63]

### [A high-performance and in-season classification system of field-level crop types using time-series Landsat data and a machine learning approach][64]

### [Multi-Temporal Land Cover Classification with Sequential Recurrent Encoders][18]

--------------------------------------------------------------------------------
# Super-resolution and pansharpening

## Papers

### [Learned Spectral Super-Resolution][21]
### [Pansharpening by CNN][22]
### [Pansharpening via Detail Injection Based Convolutional Neural Networks][23]

--------------------------------------------------------------------------------

# Generative models for data synthesis

## Papers

### [GANs for Realistic Synthesis of hyperspectral samples][24]

--------------------------------------------------------------------------------
[1]: https://www.researchgate.net/publication/311223153_Speedaccuracy_trade-offs_for_modern_convolutional_object_detectors
[2]: https://arxiv.org/pdf/1603.06201.pdf
[3]: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=2ahUKEwiDlo67jY3mAhWFsKQKHcFGBxMQwqsBMAF6BAgLEAQ&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DnDPWywWRIRo&usg=AOvVaw251YCv68Wl_c-eUBdnE5h-
[4]: https://www.novatec-gmbh.de/blog/semantic-segmentation-part-3-transfer-learning/
[5]: https://towardsdatascience.com/review-ssd-single-shot-detector-object-detection-851a94607d11
[6]: https://towardsdatascience.com/review-dssd-deconvolutional-single-shot-detector-object-detection-d4821a2bbeb5
[7]: https://medium.com/data-from-the-trenches/object-detection-with-deep-learning-on-aerial-imagery-2465078db8a9
[9]: https://deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/
[10]: https://arxiv.org/pdf/1703.06870.pdf
[11]: https://dlt18.sciencesconf.org/data/Audebert.pdf
[12]: https://arxiv.org/pdf/1511.05641.pdf
[13]: https://www.researchgate.net/publication/311223153_Speedaccuracy_trade-offs_for_modern_convolutional_object_detectors
[14]: https://www.novatec-gmbh.de/blog/semantic-segmentation-part-4-state-of-the-art/
[15]: https://www.novatec-gmbh.de/blog/semantic-segmentation-part-1-deeplab-v3/
[16]: https://www.novatec-gmbh.de/blog/semantic-segmentation-part-2-training-u-net/
[17]: https://www.researchgate.net/publication/319955230_Deep_Learning_in_Remote_Sensing_A_Review
[18]: https://arxiv.org/abs/1802.02080
[19]: https://arxiv.org/pdf/1812.05815.pdf
[20]: https://arxiv.org/pdf/1807.09562.pdf
[21]: https://arxiv.org/abs/1703.09470
[22]: https://www.researchgate.net/publication/305338139_Pansharpening_by_Convolutional_Neural_Networks
[23]: https://ieeexplore.ieee.org/document/8667040
[24]: https://arxiv.org/abs/1806.02583
[25]: https://www.mdpi.com/2072-4292/9/4/368/htm
[26]: https://arxiv.org/pdf/1806.04331.pdf
[27]: https://www.mdpi.com/2072-4292/9/1/67/htm
[28]: https://elib.dlr.de/106352/2/CNN.pdf
[29]: https://arxiv.org/pdf/1711.08681.pdf
[30]: https://arxiv.org/abs/1609.06846
[31]: http://rcdaudt.github.io/files/2018igarss-change-detection.pdf
[32]: https://rcdaudt.github.io/oscd/
[33]: https://project.inria.fr/aerialimagelabeling/
[34]: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=2ahUKEwj9nLaY0o_mAhUBIVAKHTAsAhcQFjAAegQIARAC&url=https%3A%2F%2Fwww.mdpi.com%2F2072-4292%2F11%2F15%2F1774%2Fpdf&usg=AOvVaw3BMBIIkxymYokfm6HOQSJU
[35]: https://www.cs.toronto.edu/~vmnih/data/
[36]: https://spacenetchallenge.github.io/datasets/spacenet-OffNadir-summary.html
[37]: https://medium.com/the-downlinq/introducing-the-spacenet-off-nadir-imagery-and-buildings-dataset-e4a3c1cb4ce3
[38]: https://community.topcoder.com/longcontest/stats/?module=ViewOverview&rd=17313
[39]: https://medium.com/the-downlinq/the-good-and-the-bad-in-the-spacenet-off-nadir-building-footprint-extraction-challenge-4c3a96ee9c72
[40]: https://github.com/SpaceNetChallenge/SpaceNet_Off_Nadir_Solutions
[41]: https://medium.com/the-downlinq/the-spacenet-challenge-off-nadir-buildings-introducing-the-winners-b60f2b700266
[42]: https://arxiv.org/abs/1801.05746
[43]: https://arxiv.org/abs/1709.01507
[44]: https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data
[45]: http://deepglobe.org/
[46]: https://github.com/tensorflow/models/tree/master/research/object_detection
[47]: http://xviewdataset.org/
[48]: https://insights.sei.cmu.edu/sei_blog/2019/01/deep-learning-and-satellite-imagery-diux-xview-challenge.html
[49]: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7536139
[50]: http://dase.grss-ieee.org/
[51]: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7587405
[52]: https://core.ac.uk/download/pdf/147323889.pdf
[53]: https://www.sipeo.bgu.tum.de/downloads
[54]: https://reader.elsevier.com/reader/sd/pii/S0924271618300352?token=433A52C26A0BD7DE1E20DB752317DD7CDBE7A4CAFEFD688AFFFA429BF50362587FF40736B2A13D42FE938AFD616B9F62
[55]: https://drive.google.com/file/d/14WxHQBiFiHMH9_Xzv2BpS_w6urgA3scb/view
[56]: https://downloads.greyc.fr/vedai/
[57]: http://www2.isprs.org/commissions/comm2/wg4/vaihingen-2d-semantic-labeling-contest.html
[58]: https://www.researchgate.net/publication/318574881_Zoom_out_CNNs_features_for_optical_remote_sensing_change_detection
[59]: https://sci-hub.se/10.1109/rsip.2017.7958815
[60]: https://sci-hub.se/10.1109/tgrs.2016.2601622
[61]: https://iges.or.jp/en/publication_documents/pub/peer/en/6898/Ma+et+al+2019.pdf
[62]: https://github.com/geoslegend/Deep-Learning-for-Spatio-temporal-Prediction
[63]: https://ieeexplore.ieee.org/document/7752890
[64]: https://sci-hub.se/10.1016/j.rse.2018.02.045
[65]: https://www.researchgate.net/publication/337146658_High-Resolution_Optical_Remote_Sensing_Imagery_Change_Detection_Through_Deep_Transfer_Learning
[66]: https://medium.com/@darshita1405/superpixels-and-slic-6b2d8a6e4f08
[67]: https://sci-hub.se/10.1109/lgrs.2016.2611001
[68]: https://sci-hub.se/10.1117/12.2243798
[69]: https://sci-hub.se/10.1109/tnnls.2016.2636227
[70]: https://www.researchgate.net/publication/224999140_Comparison_of_orthorectification_methods_suitable_for_rapid_mapping_using_direct_georeferencing_and_RPC_for_optical_satellite_data
[71]: https://sci-hub.se/10.1109/tnnls.2015.2435783
[72]: https://arxiv.org/abs/1911.12903
[73]: http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html
[74]: https://www.kaggle.com/lopuhin/full-pipeline-demo-poly-pixels-ml-poly
[75]: http://www.grss-ieee.org/community/technical-committees/data-fusion/2015-ieee-grss-data-fusion-contest/
[76]: https://arxiv.org/abs/1904.05730
[77]: https://www.sipeo.bgu.tum.de/downloads
[78]: https://arxiv.org/pdf/1803.09050.pdf
[79]: https://www.researchgate.net/publication/327470305_PSI-CNN_A_Pyramid-Based_Scale-Invariant_CNN_Architecture_for_Face_Recognition_Robust_to_Various_Image_Resolutions
[80]: http://www.lirmm.fr/ModuleImage/Ienco.pdf
[81]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6767260/
[82]: http://cfg.mit.edu/sites/cfg.mit.edu/files/learning_to_zoom.pdf
[83]: https://www.jeremyjordan.me/semantic-segmentation/
[84]: https://arxiv.org/pdf/1711.10684.pdf
[85]: http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
[86]: https://arxiv.org/abs/1905.11946v1
[87]: http://proceedings.mlr.press/v102/kervadec19a/kervadec19a.pdf
[88]: https://arxiv.org/abs/1602.06564
[89]: https://arxiv.org/pdf/1707.03237.pdf
[90]: https://sci-hub.se/10.1109/tmi.2006.880587