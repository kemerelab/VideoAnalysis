//#include "opencv2/opencv.hpp"	// add all openCV functionality
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/tracking/tracker.hpp>
#include "opencv2/highgui.hpp"
//C++ includes
#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <stdlib.h>     /* div, div_t */
#include <string>

//TODO: for multicamera implementation, and better performance control, use grab & retrieve methods instead of cap >> img0...
//TODO: add toggle for smoothing
//TODO: add GUI to change algorithms, load files, etc.
//TODO: add toggle button for performance mode---don't compute any unneccessary information and/or images.

using namespace std;
using namespace cv;

static double currframe, maxframes, maxkeyframes, keyframe;	// TODO: move this into SystemState or elsewhere local in the main program
static vector<int> framelist;	// list of bookmarked frames // TODO: load and save bookmarks from and to file

static VideoCapture cap;
static Mat ROImask;			// TODO: move this into SystemState or elsewhere local in the main program

static Rect2d ROIrect;		// TODO: move this into SystemState or elsewhere local in the main program

//TODO: consider changing structs to classes, for cleaner and more powerful C++ implenentations
struct SystemState
{
	bool update_bg_model;
	bool compute_bg_image;
	bool paused;
	int morph_size;
};

struct Windows
{
	string main, fgmask, fgimg, bgmodel, mmask, mimg;
};

struct MouseParams
{
    Mat img;
    string window_title;
};

static void help()
{
	printf("Learns the background at the start and then segments.\n"
	"Learning is toggled by the space key. Will read from file or camera\n"
	"Usage: \n"
	"	./vidanalysis [--camera]=<use camera, if this key is present>, [--file_name]=<path to movie file> \n\n");
}

const char* keys =
{
	"{c camera   |         | use camera or not}"
	"{m methodBG |mog2     | method for background subtraction (knn or mog2) }"
	"{s smooth   |         | smooth the BG mask }"
	"{fn file_name|../data/tree.avi | video file }"
	"{t methodTracker |MIL | method for tracking (MIL, BOOSTING, MEDIANFLOW, TLD,...)}"
};

struct timeStruct {
    int hour;
    int minute;
    int second;
    int millisecond;
};

timeStruct getTime()
{
	timeStruct myTime;
	double currtime = cap.get(CAP_PROP_POS_MSEC);
	div_t qr = div(currtime, 1000);
	myTime.millisecond = qr.rem;
	qr = div(qr.quot, 60);
	myTime.second = qr.rem;
	qr = div(qr.quot, 60);
	myTime.minute = qr.rem;
	myTime.hour = qr.quot;
	return myTime;
}

void updateDisplay(string window_title, const Mat& in)
{
	//display image in window:	
	imshow(window_title, in);
}


void updateMainDisplay(string window_title, const Mat& in)
{
	double alpha = 0.3;
	int beta = 0;
	Mat tmp = Mat::zeros(in.size(), in.type());

	// darken image:
	in.convertTo(tmp, -1, alpha, beta); 
	// apply bounding box to main image:
	in.copyTo(tmp,ROImask);  
	//draw ROI bbox:
	rectangle(tmp, ROIrect, Scalar(255,170,0),2);
	//display image in main window:	
	imshow(window_title, tmp);
}

void onMouseClick(int event, int x, int y, int flags, void* param )
{
	if (event == EVENT_LBUTTONDOWN && flags == EVENT_FLAG_CTRLKEY + EVENT_FLAG_LBUTTON) 
	{
		cout << "BBox selection started!" << endl;
		ROIrect.x = x;
		ROIrect.y = y;
	}
	else if (event != EVENT_MOUSEMOVE || flags != EVENT_FLAG_CTRLKEY + EVENT_FLAG_LBUTTON) return;

	MouseParams* mp = (MouseParams*)param;
    Mat in = mp->img;
	
	// update roi rectangle:
	//TODO: make it robust: check for limits, allow for right-to-left specification
	ROIrect.width = std::abs(x - ROIrect.x);
	ROIrect.height = std::abs(y - ROIrect.y);
	
	/*
	cout << "ROIrect.x = " << ROIrect.x << endl;
	cout << "ROIrect.y = " << ROIrect.y << endl;
	cout << "ROIrect.width = " << ROIrect.width << endl;
	cout << "ROIrect.height = " << ROIrect.height << endl;
	*/

	// update mask
	ROImask = Mat::zeros(in.size(), in.type());
	ROImask(ROIrect) = 1;

	// display image in main window:
	updateMainDisplay(mp->window_title, mp->img);	// this call to updateMainDisplay() gives CORRECT coordinates, if used on its own
}

int handleKeys(string window_title, SystemState& state, int timeout)
{
	//TODO: add toggle for persistent frame and/or time display
	//TODO: add time display when navigating video
	//TODO: add stats key, with video length, framerate, resolution, and number of frames, etc.
	int msgtimeout = 1500;	// time in milliseconds that message is displayed
	char overlaytext[255];
	char k = (char)waitKey(timeout);

	if (k==27) // ESC key exits the program
		return -1;
	if (k==' ') // space bar toggles background model learning
	{
		state.update_bg_model = !state.update_bg_model;
		if(state.update_bg_model)
		{
			printf("Background update is on\n");
			displayOverlay(window_title,"Background update is on",msgtimeout);
		}
		else
		{
		    printf("Background update is off\n");
			displayOverlay(window_title,"Background update is off",msgtimeout);
		}
		if (!state.paused)
			return 1;
		else
			return 0;
	}
	if (k==43) // + increase morph_size
	{
	//TODO:updateWindow(const string& winname);
		state.morph_size = ++state.morph_size;
		sprintf(overlaytext, "morph_size = %d", state.morph_size);
		displayOverlay(window_title,overlaytext, msgtimeout);
		if (!state.paused)
			return 1;
		else
			return -2; // update morphological fitering in paused state
	}
	if (k==45) // - decrease morph_size
	{
	//TODO:updateWindow(const string& winname);
		if (state.morph_size>1)
			state.morph_size = --state.morph_size;
		sprintf(overlaytext, "morph_size = %d", state.morph_size);
		displayOverlay(window_title,overlaytext, msgtimeout);
		if (!state.paused)
			return 1;
		else
			return -2; // update morphological fitering in paused state
	}

	if (k==48) // go to most recent interesting frame
	{
		if (maxkeyframes==0)
		{
			displayOverlay(window_title,"No keyframes available",msgtimeout);
			return 0; // don't update display
		}
		else
		{
			currframe = framelist[keyframe];
			cap.set(CAP_PROP_POS_FRAMES,currframe);
			sprintf(overlaytext, "keyframe #%g at frame #%g", keyframe+1, currframe+1);
			displayOverlay(window_title,overlaytext,msgtimeout);
		}
		return 1;
	}
	if ((k==']')||(k==56)) // go to next interesting frame, wrapping around at end
	{
		if (maxkeyframes==0)
		{
			displayOverlay(window_title,"No keyframes available",msgtimeout);
			return 0; // don't update display
		}
		else
		{
			if (keyframe < maxkeyframes-1)
				keyframe = ++keyframe;
			else
				keyframe = 0;
			currframe = framelist[keyframe];
			cap.set(CAP_PROP_POS_FRAMES,currframe);
			sprintf(overlaytext, "keyframe #%g at frame #%g", keyframe+1, currframe+1);
			displayOverlay(window_title,overlaytext,msgtimeout);
		}
		return 1;
	}
	if ((k=='[')||(k==50)) // go to previous interesting frame, wrapping around in beginning
	{
		if (maxkeyframes==0)
		{
			displayOverlay(window_title,"No keyframes available",msgtimeout);
			return 0; // don't update display
		}
		else
		{
			if (keyframe > 0)
				keyframe = --keyframe;
			else
				keyframe = maxkeyframes-1;
			currframe = framelist[keyframe];
			cap.set(CAP_PROP_POS_FRAMES,currframe);
			sprintf(overlaytext, "keyframe #%g at frame #%g", keyframe+1, currframe+1);
			displayOverlay(window_title,overlaytext,msgtimeout);
		}
		return 1;
	}
	if (k==51) // go forward 1 frame
	{
		currframe = cap.get(CAP_PROP_POS_FRAMES);
		//check if end of video has been reached
		if (currframe < maxframes)
		{
//			currframe = currframe+1;
			cap.set(CAP_PROP_POS_FRAMES,currframe);
			sprintf(overlaytext, "frame #%g", currframe+1);
			displayOverlay(window_title,overlaytext,msgtimeout);
			return 1;	
		}
		else
		{
			sprintf(overlaytext, "Cannot go forward---reached end of video");
			displayOverlay(window_title,overlaytext,msgtimeout);
			return 0;			
		}
	}
	if (k==49) // go back 1 frame
	{
		currframe = cap.get(CAP_PROP_POS_FRAMES);
		//check if start of video has been reached
		if (currframe > 1)
		{
			currframe = currframe-2;
			cap.set(CAP_PROP_POS_FRAMES,currframe);
			sprintf(overlaytext, "frame #%g", currframe+1);
			displayOverlay(window_title,overlaytext,msgtimeout);
			return 1;
		}
		else
		{
			sprintf(overlaytext, "Cannot go backward---reached start of video");
			displayOverlay(window_title,overlaytext,msgtimeout);
			return 0;			
		}
	}
	if ((k=='.')||(k==54)) // go forward 30 frames
	{
		currframe = cap.get(CAP_PROP_POS_FRAMES);
		//check if end of video has been reached
		if (currframe < maxframes-29)
		{
			currframe = currframe+29;
			cap.set(CAP_PROP_POS_FRAMES,currframe);
			sprintf(overlaytext, "frame #%g", currframe+1);
			displayOverlay(window_title,overlaytext,msgtimeout);
			return 1;	
		}
		else
		{
			sprintf(overlaytext, "Cannot go forward---reached end of video");
			displayOverlay(window_title,overlaytext,msgtimeout);
			return 0;			
		}
	}
	if ((k==',')||(k==52)) // go back 30 frames
	{
		currframe = cap.get(CAP_PROP_POS_FRAMES);
		//check if start of video has been reached
		if (currframe > 30)
		{
			currframe = currframe-31;
			cap.set(CAP_PROP_POS_FRAMES,currframe);
			sprintf(overlaytext, "frame #%g", currframe+1);
			displayOverlay(window_title,overlaytext,msgtimeout);
			return 1;
		}
		else
		{
			sprintf(overlaytext, "Cannot go backward---reached start of video");
			displayOverlay(window_title,overlaytext,msgtimeout);
			return 0;			
		}
	}
	if ((k==62)||(k==57)) // go 150 frames forward, with < (SHIFT + ,)
	{
		currframe = cap.get(CAP_PROP_POS_FRAMES);
		//check if end of video has been reached
		if (currframe < maxframes-149)
		{
			currframe = currframe+149;
			cap.set(CAP_PROP_POS_FRAMES,currframe);
			sprintf(overlaytext, "frame #%g", currframe+1);
			displayOverlay(window_title,overlaytext,msgtimeout);
			return 1;		
		}
		else
		{
			sprintf(overlaytext, "Cannot go forward---reached end of video");
			displayOverlay(window_title,overlaytext,msgtimeout);
			return 0;	
		}
	}
	if ((k==60)||(k==55)) // go 150 frames back, with > (SHIFT + .)
	{
		currframe = cap.get(CAP_PROP_POS_FRAMES);
		//check if start of video has been reached
		if (currframe > 150)
		{
			currframe = currframe-151;
			cap.set(CAP_PROP_POS_FRAMES,currframe);
			sprintf(overlaytext, "frame #%g", currframe+1);
			displayOverlay(window_title,overlaytext,msgtimeout);
			return 1;
		}
		else
		{
			sprintf(overlaytext, "Cannot go backward---reached start of video");
			displayOverlay(window_title,overlaytext,msgtimeout);
			return 0;	
		}
	}

	if ((k=='m')||(k==53)) // mark current frame as interesting frame
	{
		currframe = cap.get(CAP_PROP_POS_FRAMES);
		maxkeyframes = ++maxkeyframes;
		framelist.push_back(currframe-1); // add currently displayed frame
		//std::for_each(framelist.begin(), framelist.end(), displayValue);
		sprintf(overlaytext, "Added keyframe #%g at frame #%g", maxkeyframes, currframe);
		displayOverlay(window_title,overlaytext,msgtimeout);
		if (!state.paused)
			return 1;
		else
			return 0;
	}
	if (k==80) // go to start of video (keypress: HOME)
	{
		currframe = 0;
		cap.set(CAP_PROP_POS_FRAMES,currframe);
		sprintf(overlaytext, "start of video; frame #%g", currframe+1);
		displayOverlay(window_title,overlaytext,msgtimeout);
		return 1;
	}
	if (k==87) // go to end of video (keypress: END)
	{
		currframe = maxframes-1;
		cap.set(CAP_PROP_POS_FRAMES,currframe);
		sprintf(overlaytext, "end of video; frame #%g", currframe+1);
		displayOverlay(window_title,overlaytext,msgtimeout);
		return 1;
	}
	if (k=='i') // print info about current frame and position
	{
		timeStruct myTime = getTime();	
		currframe = cap.get(CAP_PROP_POS_FRAMES);
		if (myTime.hour>0)
			sprintf(overlaytext, "current frame #%g\n current time: %d:%d:%d.%d", currframe, myTime.hour, myTime.minute, myTime.second, myTime.millisecond);
		else if (myTime.minute>0)
			sprintf(overlaytext, "current frame #%g\n current time: %d:%d.%d", currframe, myTime.minute, myTime.second, myTime.millisecond);
	else
			sprintf(overlaytext, "current frame #%g\n current time: %d.%d", currframe, myTime.second, myTime.millisecond);
		displayOverlay(window_title,overlaytext,msgtimeout);
		if (!state.paused)
			return 1;
		else
			return 0;
	}
	if (k=='p') // pause or unpause
	{
		if (!state.paused)
		{
			state.paused = true;
			displayOverlay(window_title,"Video playback paused. Press 'p' to resume.",0);
			return 0;
		}
		else
		{
			state.paused = false;
			displayOverlay(window_title,"Video playback resumed",msgtimeout);
			return 1;
		}
	}
	if (k=='b') // toggle compute background image (not to be confused with computing background model!)
	{
		state.compute_bg_image = !state.compute_bg_image;
		if (state.compute_bg_image==true)
			displayOverlay(window_title,"compute_bg_image = ON",msgtimeout);
		else
			displayOverlay(window_title,"compute_bg_image = OFF",msgtimeout);
		return 1;
	}
	if (k==63) // display keyboard shortcuts "?"
	{
		sprintf(overlaytext, "'p' pause or resume playback\n '[' go to previous keyframe\n']'go to next keyframe\n'?' display keyboard shortcuts\n','\n\'.'\nSHIFT + ','\nSHIFT + '.'\n'i'\nSPACE\n<ESC> quits the program\nHOME\nEND");
		displayOverlay(window_title,overlaytext,3000);
		if (!state.paused)
			return 1;
		else
			return 0;
	}
	/*
	if (k>0)
	{
		sprintf(overlaytext, "keycode: %d", (int)k);
		displayOverlay(window_title,overlaytext,1500);
		return 1;
	}
	*/

	// default behavior:
	if (!state.paused)
		return 1;	// default is to update display
	else
		return 0;
}

RotatedRect computeFgBBox(Mat& fgmask)
{
	vector<Point> points;
	Mat_<uchar>::iterator it = fgmask.begin<uchar>();
	Mat_<uchar>::iterator end = fgmask.end<uchar>();
  	for (; it != end; ++it)
		if (*it)
			points.push_back(it.pos());
	return minAreaRect(Mat(points));
}

void drawFgBBox(Mat& fgimg, RotatedRect bbox)
{
	Point2f vertices[4];
	bbox.points(vertices);
	int lineWidth=2;
	for(int i = 0; i < 4; ++i)
		line(fgimg, vertices[i], vertices[(i + 1) % 4], Scalar(255, 170, 0), lineWidth); 
}	

SystemState initializeSystemState()
{
	SystemState state;
	state.compute_bg_image = true;
	state.update_bg_model = true;
	state.paused = false;	//TODO: fix bug: if initialized with paused=true, then program crashes...
	state.morph_size = 12;
	//TODO: incorporate more variables and/or objects into system state:
	//useCamera
	//smoothMask
	//file
	//method
	//
	return state;
}

void applyMorphology(const Mat& fgmask, Mat& mmask, int morph_size)
{
	int morpho_type = MORPH_ELLIPSE;	//MORPH_RECT -OR- MORPH_CROSS -OR- MORPH_ELLIPSE;
	int closing_size = morph_size;
	int opening_size = morph_size;

	Mat elementOpen = getStructuringElement(morpho_type, Size( 2*opening_size + 1, 2*opening_size+1 ), Point( opening_size, opening_size ) );
	Mat elementClose = getStructuringElement(morpho_type, Size( 2*closing_size + 1, 2*closing_size+1 ), Point( closing_size, closing_size ) );

	erode(fgmask, mmask, elementOpen);
	dilate(fgmask, mmask, elementOpen);
	dilate(fgmask, mmask, elementClose);
	erode(fgmask, mmask, elementClose);
}

Windows initializeWindows()
{
	Windows windows;
	// define titles for all named windows
	windows.main="video feed";
	windows.fgmask="foreground mask";
	windows.fgimg="foreground image";
	windows.bgmodel="mean background image";
	windows.mmask="morphological mask";
	windows.mimg="morphological image";
	
	return windows;
}

void positionWindows(Windows windows)
{
	int xstart = 100;
    int ystart = 300;
    int xsep = 410;
    int ysep = 350;
	moveWindow(windows.main, xstart+2*xsep, ystart);
	moveWindow(windows.bgmodel, xstart+2*xsep, ystart+ysep);
	moveWindow(windows.mimg, xstart+xsep, ystart+ysep);
	moveWindow(windows.fgimg, xstart+xsep, ystart);
	moveWindow(windows.mmask, xstart, ystart+ysep);
	moveWindow(windows.fgmask, xstart, ystart);
}


int main(int argc, const char** argv)
{
	//TODO: filename arguments "~/..." does not work. Have to specify "/home/etienne/..."
		
	SystemState state = initializeSystemState();	// initialize system state (paused, compute_bg_model, etc.)
	Windows windows = initializeWindows();			// initialize window names and positions
	//namedWindow(windows.main, WINDOW_AUTOSIZE);
	namedWindow(windows.main, WINDOW_NORMAL|WINDOW_KEEPRATIO);
    namedWindow(windows.fgmask, WINDOW_NORMAL|WINDOW_KEEPRATIO);
    namedWindow(windows.fgimg, WINDOW_NORMAL|WINDOW_KEEPRATIO);
    namedWindow(windows.bgmodel, WINDOW_NORMAL|WINDOW_KEEPRATIO);
    namedWindow(windows.mmask, WINDOW_NORMAL|WINDOW_KEEPRATIO);
    namedWindow(windows.mimg, WINDOW_NORMAL|WINDOW_KEEPRATIO);
	positionWindows(windows);
	
    Mat img0, img, fgmask, fgimg, mmask, mimg;
    
    help();

    CommandLineParser parser(argc, argv, keys);
    
    bool useCamera = parser.has("camera");
    bool smoothMask = parser.has("smooth");
    string file = parser.get<string>("file_name");
    printf("file: %s\n",file.c_str());
    string method = parser.get<string>("method");
    
    if (useCamera)
        cap.open(0);	// open default connected camera}
    else
	{
		cap.open(file.c_str());	// open video file
		//cap.open(file);	// open video file
		maxframes = cap.get(CAP_PROP_FRAME_COUNT);	// determine number of frames in video file
		cout << "Total number of frames: " << maxframes << endl;
	}

    parser.printMessage();

    if( !cap.isOpened() )
    {
        printf("can not open camera or video file\n");
        return -1;
    }

	double fps = cap.get(5); //get the frames per seconds of the video
	cout << "Frame per seconds : " << fps << endl;
	cout << "Source width: " << cap.get(CAP_PROP_FRAME_WIDTH) << endl;
	cout << "Source height: " << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
	
	// define default ROI mask (entire image)
	cap >> img0;	// get first frame from camera or video file
	cvtColor(img0, img0, COLOR_BGR2GRAY);	// convert to monochrome
	if (img0.empty())
		cout << "Unable to read from source!" << endl;
	resize(img0, img, Size(640, 640*img0.rows/img0.cols), INTER_LINEAR);

	// defailt ROIrect is entire image:
	ROIrect.x = 0;
	ROIrect.y = 0;
	ROIrect.width = img.cols;	//TODO: this will only work for monocrhome images! Consider generalizing...
	ROIrect.height = img.rows;

	// update mask
	ROImask = Mat::zeros(img.size(), img.type());
	ROImask(ROIrect) = 1;
    
    MouseParams mp;
    mp.img = img;
    mp.window_title = windows.main;
    
	setMouseCallback(windows.main, onMouseClick, (void*)(&mp) );

    Ptr<BackgroundSubtractor> bg_model = method == "knn" ?
            createBackgroundSubtractorKNN().dynamicCast<BackgroundSubtractor>() :
            createBackgroundSubtractorMOG2().dynamicCast<BackgroundSubtractor>();

    for(;;)
    {
		Mat bgimg;
		//TODO: fix delay for close to realtime playback:
		int k = handleKeys(windows.main, state, 15); // check for key presses, and handle accordingly
        if (k==1) 
        {
			// get next frame from file:
        	cap >> img0;	// get next frame from camera or video file
        	//TODO: fix methods to work in color or monochrome mode
        	cvtColor(img0, img0, COLOR_BGR2GRAY);	// convert to monochrome
			//TODO: fix end of video termination issue
			if (img0.empty())
				displayOverlay(windows.main,"Unable to get next frame/end of video",1500);
			
			resize(img0, img, Size(640, 640*img0.rows/img0.cols), INTER_LINEAR);	// why do I want to resize this? computational efficiency?
	
	        if (fgimg.empty())
				fgimg.create(img.size(), img.type());

		    //update the background model (learning, if active, otherwise simply compute new background image)
			bg_model->apply(img, fgmask, state.update_bg_model ? -1 : 0);
			if (smoothMask)
				{
				    GaussianBlur(fgmask, fgmask, Size(21, 21), 3.5, 3.5);
				    threshold(fgmask, fgmask, 10, 255, THRESH_BINARY);
				}
			
			// ###################### MORPHOLOGICAL FILTERING #####################################			
			// apply morphological filtering:
			applyMorphology(fgmask, mmask, state.morph_size);

			// compute foreground image BEFORE morphological filtering TODO:remove this
			fgimg = Scalar::all(0);
			img.copyTo(fgimg, fgmask);	//TODO: remove fgimg, and only use mimg...

			// compute foreground image after morphological filtering:
			mimg = Scalar::all(0);
			img.copyTo(mimg, mmask);
			// ####################################################################################
		
			// compute and draw BBOx on foreground image: TODO: toggle or remove this; unneccesary computation
			RotatedRect fgbox = computeFgBBox(mmask);
			drawFgBBox(mimg, fgbox);

			// compute background model image: TODO: toggle or remove this; unneccesary computation
			if (state.compute_bg_image == true)
				bg_model->getBackgroundImage(bgimg);
        }
        else if (k==-2) // update morphological filtering while paused
        {
  			// ###################### MORPHOLOGICAL FILTERING #####################################			
			// apply morphological filtering:
			applyMorphology(fgmask, mmask, state.morph_size);

			// compute foreground image after morphological filtering:
			mimg = Scalar::all(0);
			img.copyTo(mimg, mmask);
			// ####################################################################################
			
			// compute and draw BBOx on foreground image: TODO: toggle or remove this; unneccesary computation
			RotatedRect fgbox = computeFgBBox(mmask);
			drawFgBBox(mimg, fgbox);
        }
        else if (k==-1)
        	break;
        		
        //Mat tmp = Mat::zeros(img0.size(), img0.type());
        //img0.copyTo(tmp);
        // update video displays:
        //TODO: why are my coordinates messed up? The y-coords...
		//updateMainDisplay(windows.main, img);
   		//updateDisplay(windows.main, tmp);
	    updateDisplay(windows.fgmask, fgmask);
		updateDisplay(windows.fgimg, fgimg);
		updateDisplay(windows.mmask, mmask);
		updateDisplay(windows.mimg, mimg);
		//TODO: consolidate background update and background update display for better performance...
		if(!bgimg.empty() && state.compute_bg_image==true)
	    	updateDisplay(windows.bgmodel, bgimg );
	}
    return 0;
}

