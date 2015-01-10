VideoAnalysis
=============

This is a stand-alone OpenCV-based video analysis program designed to annotate and navigate through video files, perform lens correction from camera calibration data, and to perform automated animal tracking.

Currently implemented:
<ul>
	<li>Fast keyboard-based video navigation</li>
	<li>Background segmentation using Mixtures of Gaussians (MoG)</li>
	<li>ROI-based processing</li>
	<li>tracking coming soon!</li>
</ul>

The code depends on features of [OpenCV 3.0.0 beta](http://opencv.org/downloads.html), Qt 5+, and OpenCV-contrib (tracking-api). It has been tested in Ubuntu 14.04 LTS.

Installation instructions for OpenCV on Ubuntu 14.04
----------------------------------------------------

<ol>
	<li>download latest version opencv-3.0.0-beta from http://opencv.org/downloads.html</li>
	<li> <code>sudo apt-get install build-essential</code>
		<code>sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev</code>
		<code>sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev</code></li>
	<li>In your terminal, make sure you are within the OpenCV directory and run the following commands:
	<code>git clone https://github.com/Itseez/opencv_contrib.git</code>
	<code>mkdir build</code>
	<code>cd build</code>
	<code>cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local -DFORCE_VTK=ON -DWITH_GDAL=ON -DWITH_XINE=ON -DWITH_CUBLAS=ON -DWITH_CUFFT=ON -DWITH_OPENGL=ON -DWITH_QT=ON -DWITH_TBB=ON -DBUILD_DOCS=ON -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON -D CUDA_ARCH_BIN="3.0" -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules .. -DBUILD_opencv_cvv=OFF</code></li>
	<li>Then build and install:
	<code>make -j7 # runs 7 jobs in parallel</code>
	<code>sudo make install</code></li>
	<li>finally, configure OpenCV: <code>sudo ldconfig</code></li>
</ol>

