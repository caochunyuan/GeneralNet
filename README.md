# GeneralNet

# 一般流程

1. 从[caffe/models/](https://github.com/BVLC/caffe/tree/master/models)下载`deploy.prototxt`和`net.caffemodel`，从[caffe/data/](https://github.com/BVLC/caffe/tree/master/data)下载数据集对应的标签`.txt`。

2. 运行`convert_prototxt.py`，输出**描述网络结构的JSON文件（CPU、GPU通用）**。从终端运行的时候需要给三个参数：`.prototxt`的路径、`.txt`的路径和希望输出的JSON文件名（不包括`.json`）。比如：`python convert_prototxt.py deploy.prototxt synset_words.txt alexnet`。输出就是`alexnet.json`。

3. 运行`convert_caffemodel.py`，输出**存有卷积层和全连接层的权重和偏置的dat文件**。从终端运行的时候需要给三个参数：`.prototxt`的路径、`.caffemodel`的路径和希望输出的dat文件名（不包括`.dat`）。比如：`python convert_caffemodel.py deploy.prototxt alexnet.caffemodel alexnet`。输出就是`alexnet.dat`。这一步可能出现的细节问题比较多，可能需要手动改`convert_caffemodel.py`的代码。

4. 把`.json`文件和`.dat`文件放进Xcode工程里，然后用`-initWithDescriptionFile:dataFile:`**初始化**一个`GeneralNet`，`-forwardWithImage:completion:`**输入一个UIImage并且运行网络**，`-labelsOfTopProbs`**取网络计算的结果**。例如：

```objc
UIImage *testImage = [UIImage imageNamed:@"test.jpg"];
GeneralNet *anyNet = [[GeneralNet alloc] initWithDescriptionFile:[[NSBundle mainBundle] pathForResource:@"anynet" ofType:@"json"]
                                                        dataFile:[[NSBundle mainBundle] pathForResource:@"anynet" ofType:@"dat"]];
[anyNet forwardWithImage:testImage completion:^{
    NSString *result = [anyNet labelsOfTopProbs];
}];
```

# 各部分详解

## convert_prototxt.py

这个脚本主要是把`.prototxt`转换成`.json`。它引用了`caffe_pb2.py`，他俩应该放在同一个文件夹里。

`.prototxt`里存有构建网络的所有必需的参数，但仅仅是“必需”的。比如说它只存有输入图像的长和宽，而中间每一层的计算结果的长和宽都是没有的，但是可以从输入层往下一层一层推断出来。获取这样需要推断的信息，就是这个脚本主要的任务之一。

55行到111行都是从`.prototxt`里直接取出已经存有的、不需要推断的信息，从123行到183行都是用来根据前后层来推断信息（主要是输入、输出的长、宽和通道数），191行以后是把必要的信息存到JSON文件里。

需要额外注意的是转换GoogLeNet的时候，`weight_offset`和`bias_offset`的设置需要额外处理（如果我们有一个卷积层或全连接层，就需要到`.dat`文件里去找这一层的权重和偏置，文件的头指针加上`weight_offset`／`bias_offset`，就得到指向权重／偏置的指针）。`weight_offset`和`bias_offset`在`convert_prototxt.py`里是通过一层一层地累加权重和偏置的数据长度的到的，与众不同的是，GoogLeNet的`.caffemodel`文件里有一个叫`loss1/classifier`和一个叫`loss2/classifier`的全连接层（[在此](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/train_val.prototxt#L904)可以看到），它们是**训练**神经网络的时候用到的参数，`deploy.txt`里面没有他们，我们转换出的JSON文件里也没有它们，iOS上也用不着。于是我们通过累加来得到`weight_offset`和`bias_offset`的时候，就漏加了它们的权重和偏置的数据长度，于是就乱了。

所以这是**caffemodel和prototxt不一致**引起的混乱，必需手动修改其中一个。一种解决方案是在`convert_caffemodel.py`里，让它遇到`loss1/classifier`和`loss2/classifier`的参数时直接跳过，不保存；另一个解决方案是用Convert文件夹里的`get_size.py`，它只需要一个参数，即`.caffemodel`的路径，比如：

```
python get_size.py googlenet.caffemodel
```

它会读取`.caffemodel`，记录所有卷积层和全连接层的权重和偏置的数据长度，存到一个JSON文件里；然后需要修改`convert_prototxt.py`，直接从JSON文件读取正确的`weight_offset`和`bias_offset`：

首先把JSON文件读进来

```python
f = open('offset_googlenet.json','r')
offsetDict = json.load(f)
```

然后把原来的第229、230行改成：

```python
'weight_offset': offsetDict[layer.name]['weight'],
'bias_offset': offsetDict[layer.name]['bias'],
```

再用同样的方法修改第251、252行即可。

## convert_caffemodel.py

这个脚本主要是把`.caffemodel`转换成`.dat`，存的是卷积层和全连接层的权重和偏置数据。这里也有一些很琐碎的注意事项。

首先`.caffemodel`是把权重按[outputChannels][inputChannels][kernelHeight][kernelWidth]的顺序存成四维数组，但GPU版`Metal`需要的顺序是[outputChannels][kernelHeight][kernelWidth][inputChannels]，也就是需要转置一下。那么首先需要把outputChannels、inputChannels、inputChannels和kernelWidth这四个参数读出来，见34到37行。麻烦的是各个net的`.caffemodel`存的可能不太一样，对于alexnet和googlenet，这几行是：

```python
c_o = blob.num
c_i = blob.channels
h   = blob.height
w   = blob.width

arr = np.array(blob.data, dtype=np.float32)
data = arr.reshape(c_o, c_i, h, w)
```

但是对于squeezenet，应该写成：

```python
if idx == 1:
    data = np.array(blob.data, dtype=np.float32)
elif idx == 0:
    c_o = blob.shape.dim[0]
    c_i = blob.shape.dim[1]
    h   = blob.shape.dim[2]
    w   = blob.shape.dim[3]
    
    arr = np.array(blob.data, dtype=np.float32)
    data = arr.reshape(c_o, c_i, h, w)
```

另外alexnet的全连接层需要特殊照顾，`data = arr.reshape(c_o, c_i, h, w)`得换成这一段：

```python
if layer.name == "fc6" and idx == 0:
    data = arr.reshape(4096, 256, 6, 6)
elif layer.name == "fc7" and idx == 0:
    data = arr.reshape(4096, 4096, 1, 1)
elif layer.name == "fc8" and idx == 0:
    data = arr.reshape(1000, 4096, 1, 1)
else:
    data = arr.reshape(c_o, c_i, h, w)
```

## iOS的通用网络和层

神经网络里每一层的任务都是取上一层的输出，做一些操作（卷积、全连接、池化…），然后放到自己的输出里。一个神经网络运行一遍，实际上就是按顺序把网络中每一个操作都做一遍。重点在于：

1. 每一步操作本身的参数，比如卷积层就得有卷积核大小、步长、分组数等，而这些已经在JSON文件的`layer_info`里写好了；
2. 这些操作按什么顺序执行，每一步操作应该从哪里取数据，算完以后存到哪里，而这些已经在JSON文件的`encode_seq`里写好了；
3. 所有操作做完后，根据网络最终输出的特征向量，找到可能性最大的标签（熊，火山，斑马…），而所有的标签已经保存在JSON文件的`labels`里了。

所以iOS上的核心任务就是：

1. 根据`layer_info`来构建每一步操作，并分配储存结果的空间(constructLayersWithInfo:)；
2. 按照`encode_seq`的顺序来执行操作(forwardWithImage:completion:)；
3. 在`labels`里找可能性最大的标签(labelsOfTopProbs)。

GPU版和CPU版的`GeneralNet`都有`NSMutableDictionary *layersDict`、`NSMutableArray *encodeSequence`和`NSArray *labels`这三样东西，就是拿来存JSON文件里的对应信息的。另外JSON文件里还有一个字典`inout_info`，这里储存的是有关网络输入输出层的信息，它们不参与神经网络的构建，但有其他用途。

## GPU版

GPU版网络目前都是参照苹果的[Demo](https://developer.apple.com/library/content/samplecode/MetalImageRecognition/Introduction/Intro.html)，调用`MPSCNN`的类实现的。

### 预处理

每次输入一个图像，首先要按照网络要求调整图像大小。这一步用`MPSImageLanczosScale *lanczos`实现，不需要设置缩放率之类的参数，它会根据输入输出图像的大小自己判断。

然后需要对每个像素点减去神经网络训练所用图像的RGB均值，并且互换R通道和B通道，变成GBR图像。这一步由`id <MTLComputePipelineState> pipelineRGB`来实现，具体操作写在`Shaders.metal`里面。

### MPSLayer

这个类是网络里所有层的基类，它有这些属性：

```objc
@property (strong, nonatomic) NSString *name;
@property (strong, nonatomic) MPSImageDescriptor *imageDescriptor;
@property (strong, nonatomic) MPSImage *outputImage;
@property (assign, nonatomic) NSUInteger readCount;
@property (strong, nonatomic) MPSCNNKernel *kernel;
```

分成三组来看：

1. 操作：`kernel`
2. 存储结果：`imageDescriptor`, `outputImage`, `readCount`
3. 附加信息：`name`

按照苹果的设计，他们把卷积神经网络里每一层的**操作**（卷积、全连接、池化…）封装在`MPSCNNKernel`里面，每一层的**输出**都储存在一个`MPSImage`里面。对于大多数的层来说它们兼具这两样东西，但是像在GoogLeNet和SqueezeNet里的Concat层，它们并不需要有操作，而只需要有放输出的地方；在Concat之前的层则只有操作，没有输出，因为它们的输出可以直接储存到Concat层的输出里，不需要自己再单独存储。

### MPSCNNKernel相关

`kernel`大部分时候就是按照JSON文件里面的参数来初始化，要注意的就是两个参数：`destinationFeatureChannelOffset`和`padding`。前者在处理Concat层的时候经常用到，在Concat之前的层都要设置这个参数，它的输出才能被放到正确的地方。另一个参数`padding`是很多种层都可能碰到的，这时候需要重写`encodeToCommandBuffer:sourceImage:destinationImage:`方法，改变`MPSCNNKernel`的`offset`属性即可。

另外苹果原有的`MPSCNNPoolingAverage`类没有`global`与否的选项，所以需要做一个特殊处理：改`offset`，让它只以中心像素为中心，计算平均值。如果将来poolingmax也有global的，也就是同样的改法。

### MPSImage相关

整个网络的第一个MPSImage存的是输入图像，中间的一大堆MPSImage是中间结果，最后一个MPSImage存的是输出的特征向量，即识别的结果。实际上只有最后一个是需要一直保持的，以使我们能在任何时候都能读取识别的结果；其他的中间结果我们**永远不会通过CPU去访问他们**，因此可以不给CPU提供访问权限，而且它们占用的空间可以在使用后立即销毁，不需要继续占用内存。针对后一种情况，苹果的Demo里使用的是`MPSTemporaryImage`，继承自`MPSImage`，并且把运行网络的代码放在autoreleasepool里面，这样这些中间结果所占的内存就会在用完之后被立即回收，而且任何时候CPU试图访问都会出错。

`MPSImage`需要通过一个`MPSImageDescriptor`来初始化，后者描述了这个图像的长宽和通道数。因为每次运行网络都要重新初始化一遍所有的`MPSTemporaryImage`，所以`MPSImageDescriptor`也作为`MPSLayer`的一个属性一直保存着。我们用`NSMutableArray *tempImageList`来记录有哪些层用的是临时图像。另外苹果建议在初始化之前先用`prefetchStorageWithCommandBuffer:`方法，输入所有将会用到的`MPSImageDescriptor`的列表，系统会自己计算一共要需要多少空间，提前分配好。所以我们用`NSMutableArray *prefetchList`来存储所有临时图像的`MPSImageDescriptor`。

但要注意，`MPSTemporaryImage`有时必须设置一个属性`readCount`。系统判断一个中间结果有没有被“使用完毕”、是否可以销毁，就是根据这个属性。`readCount`默认是1，也就是说它只能被接下来的一个层作为“上一层”，用完这次就被销毁，直到再次初始化。但是像GoogLeNet里的每一个Concat层都会被4次当作“上一层”，因此必须设置`tempImg.readCount = 4`。“被当作上一层”的次数已经在`convert_prototxt.py`被统计，即`as_bottom`，然后在JSON文件里被存为`read_count`，也是`MPSLayer`的属性之一。

另外需注意，如果debug的时候要打印每一层的输出，就不能再用`MPSTemporaryImage`，只能是`MPSImage`，否则CPU访问不到。所以在宏定义`ALLOW_PRINT`为1的时候，用的全都是`MPSImage`。还有必须等到执行了`[commandBuffer waitUntilCompleted];`以后才能去打印每层结果，否则网络可能还没有计算完，取到的是上一次的结果。

## CPU版

CPU版结构比较简单，每一层的操作都写在`-forwardWithInput:output:`里面。目前卷积层用的是caffe2的`ìm2col`+accelerate的`cblas_sgemm`，pooling层用的是NNPACK的代码，其他层是自己写的代码。若要更改某种层的算法，只改这一个方法即可。

苹果有

目前的问题是pooling层比较诡异，虽然用BNNS或者NNPACK算出来结果差不多，但就是和GPU版算出来的差很多，所以pooling层越多结果越难看。现在CPU版的alexnet和squeezenet还算准确，googlenet的结果就很难看。另外NNPACK的pooling算法不太好，用了一大堆的for循环，squeezenet用它比用BNNS慢了10多毫秒，这里是有改进空间的。

### 预处理

`-forwardWithImage:completion:`方法的第一步也是调整图像大小、减掉训练集的RGB均值、RGB转GBR，这些在代码里面都有标注。需要注意的是，`UIImage`取出来的数据是按照RGBARBGA...的顺序排列的，也就是一个一个像素地存储；但在神经网络中我们需要它按照RRR...GGG...BBB...的方式来存，也就是一个一个通道地存储，这一步预处理是GPU版不需要的。

### 准备权重和偏置

权重和偏置同样需要从`.dat`里面读取，但要注意**CPU版和GPU版不共用`.dat`文件**，因为CPU版需要的权重和偏执的存储顺序是与caffe一致的，也就是说运行`convert_caffemodel.py`时**不需要那一步转置**。要生成CPU版的dat文件，应该把`convert_caffemodel.py`第51行改成：

```python
pass
```

### Gemm方法

卷积层是用caffe2的`ìm2col`加上一个`Gemm`实现的。后者可以是Accelerate的`cblas_sgemm`，也可以是自己实现的`nnpack_gemm`或者`eigen_gemm`。`.pch`的宏定义有`USE_NNPACK_FOR_GEMM`和`USE_EIGEN_FOR_GEMM`，选一个定义为1即可；都为0则默认用Accelerate。注意如果用的是Eigen，必需把`CPULayer.m`的后缀改成`.mm`。如果用的是NNPACK，`CPULayer.m`后缀必须是`.m`；也可以把`nnpackGemm.c`后缀改成`.cpp`，`CPULayer.m`改成`.mm`。

Eigen的使用和SDK里面、caffe2里面都是一样的，只是要注意，已经发现用Debug版时Eigen极其慢，跑一张图片用了5秒多，用Release版的时候才比较正常，原因未知。GitHub的工程里我没有上传Eigen的源文件；搜索最新版本的Eigen，把其中的`Eigen`文件夹放进工程即可。

`nnpackGemm.c`是从NNPACK的源代码里面抽出来修改而成的，里面有一些静态定义，如：

```c
static const size_t cache_l1_size = 16 * 1024;
static const size_t cache_l2_size = 128 * 1024;
static const size_t cache_l3_size = 2 * 1024 * 1024;

static const size_t row_subblock_max = 4;
static const size_t col_subblock_max = 12;
```

这些参数都是从NNPACK的`init.c`文件里面抽出来的，如有疑问可以到`init.c`核查。
按原文件的宏定义来看，应该同样适用于Android的CPU。他的本意仿佛是按照CPU的L1、L2、L3级缓存的大小来安排每次运算的数据量，但又并没有真的去读取硬件信息；然后每次计算一个4x12的小块，这个大小的选择也没有提供理由。这些参数或许有问题，或许可以优化，目前只能确保在iPhone6P上没问题。

总的来说，NNPACK算`C = A * B`就是每次取A矩阵的4行，取B矩阵的12列，用`nnp_sgemm_only_4x12__neon`算出C矩阵的一个4x12的块（原来的算法可以在`NNPACK\src\neon\blas\sgemm.c`里面找到），然后对A矩阵最后不足4行、B矩阵不足12列的用`nnp_sgemm_upto_4x12__neon`来计算，也存到C矩阵里。原来的算法因为是和im2col紧密结合的，直接分离出来是用不了的，所以我对算法做了小的改动，目前是要求先对A矩阵转置，再调用`nnp_sgemm_only_4x12__neon`和`nnp_sgemm_upto_4x12__neon`。转置这一步时间开销也不小，如果能结合到`im2col`里面去，肯定还能更快。