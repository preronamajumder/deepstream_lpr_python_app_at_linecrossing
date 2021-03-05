# Sample For Car License Recognization

 - [Description](#description)
 - [Prerequisition](#prerequisition)
 - [Download](#Prepare-Models-and-TensorRT-engine)
 - [Build Parser](#Build-custom-parser)
 - [NVAnalytics Module](#NVAnalytics-Module)
 - [Run](#Run-app)

---

## Description

This sample is to show how to use graded models for detection and classification with DeepStream SDK version not less than 5.0.1. The models in this sample are all TLT3.0 models. Image is written to a folder when the license plate crosses a virtual line. The line crossing is achieved by using the NVAnalytics module of Deepstream python (https://github.com/NVIDIA-AI-IOT/deepstream_python_apps).

`PGIE(car license plate detection) -> SGIE(car license plate recognization)`


This pipeline is based on two TLT models below

* LPD (car license plate detection) model https://ngc.nvidia.com/catalog/models/nvidia:tlt_lpdnet
* LPR (car license plate recognization/text extraction) model https://ngc.nvidia.com/catalog/models/nvidia:tlt_lprnet

More details for TLT3.0 LPD and LPR models and TLT training, please refer to [TLT document](https://docs.nvidia.com/metropolis/TLT/tlt-getting-started-guide/).

## Prerequisition

* [DeepStream SDK 5.1](https://developer.nvidia.com/deepstream-getting-started)

  Make sure deepstream-test1 sample can run successful to verify your DeepStream installation


## Prepare Models and TensorRT engine

Download repo:
```
    git clone https://github.com/preronamajumder/deepstream_apps.git
```
Download models:
```
    cd deepstream_lpr_python_app
    ./download_us.sh
```
Download tlt-converter:

Download x86 or Jetson tlt-converter which is compatible to your platform from the following links inside the folder.

| Platform   |  Compute                       |        Link                                              |
|------------|--------------------------------|----------------------------------------------------------|
|x86 + GPU   |CUDA 10.2/cuDNN 8.0/TensorRT 7.1|[link](https://developer.nvidia.com/cuda102-cudnn80-trt71)|
|x86 + GPU   |CUDA 10.2/cuDNN 8.0/TensorRT 7.2|[link](https://developer.nvidia.com/cuda102-cudnn80-trt72)|
|x86 + GPU   |CUDA 11.0/cuDNN 8.0/TensorRT 7.1|[link](https://developer.nvidia.com/cuda110-cudnn80-trt71)|
|x86 + GPU   |CUDA 11.0/cuDNN 8.0/TensorRT 7.2|[link](https://developer.nvidia.com/cuda110-cudnn80-trt72)|
|Jetson      |JetPack 4.4                     |[link](https://developer.nvidia.com/cuda102-trt71-jp44)   |
|Jetson      |JetPack 4.5                     |[link](https://developer.nvidia.com/cuda102-trt71-jp45)   |

```
    wget <url>
    unzip <filename>
```
Convert LPR Model:  
DS5.0.1 gst-nvinfer cannot generate TRT engine for LPR model, so generate it with tlt-converter

```
    cd <foldername>
    chmod +x tlt-converter
    ./tlt-converter -k nvidia_tlt -p image_input,1x3x48x96,4x3x48x96,16x3x48x96 \
           ../models/LP/LPR/us_lprnet_baseline18_deployable.etlt -t fp16 -e ../models/LP/LPR/lpr_us_onnx_b16.engine
    cd ..
```

if you get the following:  
[WARNING] Half2 support requested on hardware without native FP16 support, performance will be negatively affected.  
Then use fp32 instead of fp16:

```
    ./tlt-converter -k nvidia_tlt -p image_input,1x3x48x96,4x3x48x96,16x3x48x96 \
           ../models/LP/LPR/us_lprnet_baseline18_deployable.etlt -t fp32 -e ../models/LP/LPR/lpr_us_onnx_b32.engine
    cd ..
```

## Build Parser

```
    cd nvinfer_custom_lpr_parser
    make
    cd ..
```
## NVAnalytics Module

More details on the nvanalytics module can be learnt from https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/app-deepstream-nvanalytics  
Change the following parameters in config_nvdsanalytics.txt:  
    [i] roi-RF: provide your own ROI where you want your LP to be detected  
    [ii] line-crossing-Entry or line-crossing-Exit: provide your own virtual line and direction as well as entry or exit.

## Run app

Run the application:
```
    python3 deepstream_lpr_app.py file://<file location>
    python3 deepstream_lpr_app.py <uri1> <uri2> ..... <uriN>
```

