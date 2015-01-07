VideoAnalysis
=============

This is a stand-alone OpenCV-based video analysis program designed to annotate and navigate through video files, perform lens correction from camera calibration data, and to perform automated animal tracking.

Currently implemented:
(1) Fast keyboard-based video navigation
(2) Faux video generation
(3) Video acquisition using PointGrey camera drivers (only working in linux)
(4) Video compression and saving using the LibAV library

The code depends on features of OpenCV 3.0.0 beta, Qt 5+, and OpenCV-contrib (tracking-api). It has been tested in Ubuntu 14.04 LTS.
