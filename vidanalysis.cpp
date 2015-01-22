//#include "opencv2/opencv.hpp"	// add all openCV functionality
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/tracking/tracker.hpp>		// from opencv-contrib tracking api
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
static vector<Point> traj;

static VideoCapture cap;
static Mat ROImask;			// TODO: move this into SystemState or elsewhere local in the main program
static Rect2d ROIrect;
static Rect2d trk_boundingBox; // TODO: move into SystemState or elsewhere

//TODO: consider changing structs to classes, for cleaner and more powerful C++ implenentations
struct SystemState
{
	bool update_bg_model;
	bool compute_bg_image;
	bool paused;
	bool tracking;
	bool tracker_initialized;
	int morph_size;
	bool tracker_init_in_progress;
	bool tracker_init_started;
	//Rect2d ROIrect;
};

struct Windows
{
	string main, fgmask, fgimg, bgmodel, mmask, mimg;
};

struct MouseParams
{
    Mat img;
    string window_title;
    Point topleft,topright,bottomleft,bottomright,origin;
    bool* tracker_init_in_progress;
	bool* tracker_initialized;
	bool* tracker_init_started;
	bool tracker_region_ready;
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
	"{t methodTracker |BOOSTING | method for tracking (MIL, BOOSTING, MEDIANFLOW, TLD,...)}"
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

void updateDisplay(string window_title, const Mat& in, Rect2d rect, cv::Size size)
{
	int lineWidth = 1;
	Mat tmp = Mat::zeros(size, in.type());	// create empty matrix with size of the video frame
	// copy content into ROI:
	in.copyTo(tmp(rect));
	// draw ROI bbox:
	rectangle(tmp, ROIrect, Scalar(255,170,0),lineWidth);
	// display image in window:	
	imshow(window_title, tmp);
}

void updateMainDisplay(string window_title, const Mat& in)
{
	int lineWidth = 1;
	double alpha = 0.3;
	int beta = 0;
	Mat tmp = Mat::zeros(in.size(), in.type());

	// darken image:
	in.convertTo(tmp, -1, alpha, beta); 
	// apply bounding box to main image:
	in.copyTo(tmp,ROImask);  
	// draw ROI bbox:
	rectangle(tmp, ROIrect, Scalar(255,170,0),lineWidth);
	// display image in main window:	
	imshow(window_title, tmp);
}

void onMouse(int event, int x, int y, int flags, void* param )
{
	MouseParams* mp = (MouseParams*)param;
	Mat in = mp->img;
	
	if (*(mp->tracker_init_in_progress)==true)
	{
		//TODO:do tracker_init code here... INIT on mouse_up
		switch ( event )
		{
			case EVENT_LBUTTONDOWN:
				*(mp->tracker_init_started)=true;
				//set origin of the bounding box
				trk_boundingBox.x = x;
				trk_boundingBox.y = y;
				cout << "boundingBox.x = " << x << endl;
				cout << "boundingBox.y = " << y << endl;
				break;
			case EVENT_LBUTTONUP:
				//TODO: shouldn't we set the x and y origin again? (if different from BBox.origin)?
				//set width and height of the bounding box
				trk_boundingBox.width = std::abs( x - trk_boundingBox.x );
				trk_boundingBox.height = std::abs( y - trk_boundingBox.y );
				*(mp->tracker_init_in_progress) = false;
				mp->tracker_region_ready = true;
				*(mp->tracker_initialized) = false;
				break;
			case EVENT_MOUSEMOVE:
				if (*(mp->tracker_init_started)==true)
				{
					//draw the bounding box
				  	Mat currentFrame;
					in.copyTo( currentFrame );
			  		rectangle( currentFrame, Point( trk_boundingBox.x, trk_boundingBox.y ), Point( x, y ), Scalar( 255, 170, 0 ), 2, 1 );
			  		imshow(mp->window_title, currentFrame );
			  	}  
		  		break;
			}
			
		//cout << "mp->tracker_init_in_progress==true" << endl;
		return;
	}
	/*
	Point topleft = mp->topleft;
	Point topright = mp->topright;
	Point bottomleft = mp->bottomleft;
	Point bottomright= mp->bottomright;
*/
	
	if (event == EVENT_LBUTTONDOWN && flags == EVENT_FLAG_CTRLKEY + EVENT_FLAG_LBUTTON) 
	{
		cout << "BBox selection started!" << endl;
		mp->origin.x = x;
		mp->origin.y = y;
		mp->topleft.x = x;
		mp->topleft.y = y;
		mp->topright.x = x+1;
		mp->topright.y = y;
		mp->bottomleft.x = x;
		mp->bottomleft.y = y+1;
		mp->bottomright.x = x+1;
		mp->bottomright.y = y+1;
	}
	else if (event != EVENT_MOUSEMOVE || flags != EVENT_FLAG_CTRLKEY + EVENT_FLAG_LBUTTON) return;

    
	
	// update roi rectangle:
	//TODO: make it robust: check for limits, allow for right-to-left specification

	if ((x<=mp->origin.x)&&(y<=mp->origin.y))
	{
		mp->topleft.x = x;
		mp->topleft.y = y;
		mp->topright.y = y;
		mp->bottomleft.x = x;
	}
	else if (x<=mp->origin.x)
	{
		mp->bottomleft.x = x;
		mp->bottomleft.y = y;
		mp->bottomright.y = y;
		mp->topleft.x = x;
	}
	else if ((x>=mp->origin.x)&&(y<=mp->origin.y))
	{
		mp->topright.x = x;
		mp->topright.y = y;
		mp->topleft.y = y;
		mp->bottomright.x = x;
	}
	else
	{
		mp->bottomright.x = x;
		mp->bottomright.y = y;
		mp->topright.x = x;
		mp->bottomleft.y = y;
	}
	
	// constrain ROI rectangle:
	mp->topleft.x = max(mp->topleft.x,1);
	mp->topleft.x = min(mp->topleft.x,mp->img.cols);
	mp->topleft.y = max(mp->topleft.y,1);
	mp->topleft.y = min(mp->topleft.y,mp->img.rows);

	mp->topright.x = max(mp->topright.x,1);	
	mp->topright.x = min(mp->topright.x,mp->img.cols);
	mp->topright.y = max(mp->topright.y,1);
	mp->topright.y = min(mp->topright.y,mp->img.rows);
	
	mp->bottomright.x = max(mp->bottomright.x,1);	
	mp->bottomright.x = min(mp->bottomright.x,mp->img.cols);
	mp->bottomright.y = max(mp->bottomright.y,1);
	mp->bottomright.y = min(mp->bottomright.y,mp->img.rows);
	
	mp->bottomleft.x = max(mp->bottomleft.x,1);	
	mp->bottomleft.x = min(mp->bottomleft.x,mp->img.cols);
	mp->bottomleft.y = max(mp->bottomleft.y,1);
	mp->bottomleft.y = min(mp->bottomleft.y,mp->img.rows);
	
	ROIrect.x = mp->topleft.x;
	ROIrect.y = mp->topleft.y;
	ROIrect.width = std::max((mp->topright.x - mp->topleft.x),1);
	ROIrect.height = std::max((mp->bottomleft.y - mp->topleft.y),1);
	
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
	updateMainDisplay(mp->window_title, mp->img);	// this call to updateMainDisplay() gives CORRECT coordinates, if used on its own, even with WINDOW_NORMAL
}

int handleKeys(string window_title, SystemState& state, int timeout)
{
	//TODO: add toggle for persistent frame and/or time display
	//TODO: add time display when navigating video
	//TODO: add stats key, with video length, framerate, resolution, and number of frames, etc.
	int msgtimeout = 1500;	// time in milliseconds that message is displayed
	char overlaytext[255];
	char k = (char)waitKey(timeout);
	
	if ((state.tracker_init_in_progress==true)&&(k!=27))
		return 0;
	else if ((k==27)&&(state.tracker_init_in_progress==false)) // ESC key exits the program
		return -1;
	else if ((k==27)&&(state.tracker_init_in_progress==true))
		{
			state.tracker_init_in_progress = false;
			displayOverlay(window_title,"",1);
			//TODO: what else should happen here? overwrite display messages?
			if (!state.paused)
				return 1;
			else
				return 0;
		}
	if (k=='g')
	{
		state.tracker_init_started = false;
		state.tracker_init_in_progress = true;
		printf("Initializing tracker...\n");
		displayOverlay(window_title,"Initializing tracker: define region, or ESC to cancel",0);
		return 0;
	}
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
	if (k=='t')	// toggle tracking
	{
		if (state.tracker_initialized)
			if (state.tracking==true)
			{
				state.tracking = false;
				displayOverlay(window_title,"Tracking disabled",msgtimeout);
			}
			else
			{
				state.tracking = true;
				displayOverlay(window_title,"Tracking enabled",msgtimeout);
			}
		else
			displayOverlay(window_title,"Tracker not initialized",msgtimeout);
		if (!state.paused)
			return 1;
		else
			return 0;
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
	int lineWidth=2;
	Point2f vertices[4];
	bbox.points(vertices);
	
	for(int i = 0; i < 4; ++i)
		line(fgimg, vertices[i], vertices[(i + 1) % 4], Scalar(255, 170, 0), lineWidth); 
}	

SystemState initializeSystemState()
{
	SystemState state;
	state.compute_bg_image = true;
	state.update_bg_model = true;
	state.paused = false;
	state.morph_size = 12;
	state.tracking = false;						// tracker enabled (TRUE) or disabled (FALSE)
	state.tracker_initialized = false;
	state.tracker_init_in_progress = false;
	state.tracker_init_started = false;
	//state.ROIrect;
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
		
	const int newwidth = 320;
	SystemState state = initializeSystemState();	// initialize system state (paused, compute_bg_model, etc.)
	Windows windows = initializeWindows();			// initialize window names and positions
	//namedWindow(windows.main, WINDOW_NORMAL|WINDOW_KEEPRATIO);
	namedWindow(windows.main, WINDOW_AUTOSIZE);
    namedWindow(windows.fgmask, WINDOW_AUTOSIZE);
    namedWindow(windows.fgimg, WINDOW_AUTOSIZE);
    namedWindow(windows.bgmodel, WINDOW_AUTOSIZE);
    namedWindow(windows.mmask, WINDOW_AUTOSIZE);
    namedWindow(windows.mimg, WINDOW_AUTOSIZE);
	positionWindows(windows);	// does not work with WINDOW_AUTOSIZE
	
    Mat img0, img, fgmask, fgimg, mmask, mimg, roimat;
	//Rect2d trk_boundingBox; //TODO: fix, use, or remove this... move into state???

    help();

    CommandLineParser parser(argc, argv, keys);
    
    bool useCamera = parser.has("camera");
    bool smoothMask = parser.has("smooth");
    string file = parser.get<string>("file_name");
    printf("file: %s\n",file.c_str());
    string methodBG = parser.get<string>("methodBG");
    String methodTracker = parser.get<string>("methodTracker");

	cout << "methodBG = " << methodBG << endl;    
	cout << "methodTracker = " << methodTracker << endl;
	
	if (methodTracker.empty()){
		cout << "No tracking algorithm specified! Assume BOOSTING" << endl;
		methodTracker = "BOOSTING";
	}
	
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
    
    // instantiates the specific Tracker
	Ptr<Tracker> tracker = Tracker::create(methodTracker);
	if (tracker==NULL)
	{
		cout << "***Error in the instantiation of the tracker...***\n";
		return -1;
	}

	/*
	double fps = cap.get(5); //get the frames per seconds of the video
	cout << "Frame per seconds : " << fps << endl;
	cout << "Source width: " << cap.get(CAP_PROP_FRAME_WIDTH) << endl;
	cout << "Source height: " << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
	*/
	
	// define default ROI mask (entire image)
	cap >> img0;	// get first frame from camera or video file
	cvtColor(img0, img0, COLOR_BGR2GRAY);	// convert to monochrome	//TODO: uncomment this!!! Only commented to test tracking...
	if (img0.empty())
		cout << "Unable to read from source!" << endl;
	resize(img0, img, Size(newwidth, newwidth*img0.rows/img0.cols), INTER_LINEAR);

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
	mp.tracker_init_in_progress = &state.tracker_init_in_progress;
	mp.tracker_initialized = &state.tracker_initialized;
	mp.tracker_init_started = &state.tracker_init_started;
	mp.tracker_region_ready = false;
	
	//cout << "*(mp.tracker_init_in_progress) = " << *(mp.tracker_init_in_progress) << endl;
    
	setMouseCallback(windows.main, onMouse, (void*)(&mp) );

    Ptr<BackgroundSubtractor> bg_model = methodBG == "knn" ?
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
			cvtColor(img0, img0, COLOR_BGR2GRAY);	// convert to monochrome //TODO: BUG: tracking seems to work only with color videos...
			//TODO: fix end of video termination issue
			if (img0.empty())
				displayOverlay(windows.main,"Unable to get next frame/end of video",1500);
			
			resize(img0, img, Size(newwidth, newwidth*img0.rows/img0.cols), INTER_LINEAR);	// why do I want to resize this? computational efficiency?
			
			// obtain ROI slice from img:
			roimat = img(ROIrect);
	
			// check if ROI size has changed, and re-initialize fgimg, if necessary:
	        if ((fgimg.empty()||fgimg.size()!=roimat.size())&&(!roimat.empty())){
	        	//cout << "size changed!" << endl;
				fgimg.create(roimat.size(), roimat.type());}

		    //update the background model (learning, if active, otherwise simply compute new background image)
			bg_model->apply(roimat, fgmask, state.update_bg_model ? -1 : 0);
			if (smoothMask)
				{
				    //GaussianBlur(fgmask, fgmask, Size(21, 21), 3.5, 3.5);
				    GaussianBlur(fgmask, fgmask, Size(9, 9), 3.5, 3.5);	//TODO: allow for smoothing size to be dynamically changed
				    threshold(fgmask, fgmask, 10, 255, THRESH_BINARY);
				}
			
			// ###################### MORPHOLOGICAL FILTERING #####################################			
			// apply morphological filtering:
			applyMorphology(fgmask, mmask, state.morph_size);

			// compute foreground image BEFORE morphological filtering TODO:remove this
			fgimg = Scalar::all(0);
			roimat.copyTo(fgimg, fgmask);	//TODO: remove fgimg, and only use mimg...

			// compute foreground image after morphological filtering:
			mimg = Scalar::all(0);
			roimat.copyTo(mimg, mmask);
			// ####################################################################################
		
			// compute and draw BBOx on foreground image: TODO: toggle or remove this; unneccesary computation
			RotatedRect fgbox = computeFgBBox(mmask);
			drawFgBBox(mimg, fgbox);

			// compute background model image: TODO: toggle or remove this; unneccesary computation
			if (state.compute_bg_image == true)
				bg_model->getBackgroundImage(bgimg);
				
			// ### perform tracking on filtered foreground image ###
			// TODO: the tracking should be applied whether or not the state is PAUSED---makes sense for some algorithms (detection-based) and not for others
			if( !state.tracker_initialized && mp.tracker_region_ready )
			{
				cout << "***Attempting to intialize tracker...***\n";
				//initializes the tracker TODO: trk_boundingBox should be shifted relative to frame
				if( !tracker->init(img, trk_boundingBox ) )
				{
					cout << "***Could not initialize tracker...***\n";
					return -1;
				}
				state.tracker_initialized = true;
				mp.tracker_region_ready = false;
				cout << "***Tracker initialized successfully...***\n";
			}
			else if (state.tracker_initialized && state.tracking)
			{
				//updates the tracker
				if( tracker->update(img, trk_boundingBox ) )
				{
					rectangle( img, trk_boundingBox, Scalar( 255, 170, 0 ), 2, 1 ); //TODO: not fgimg...
				}
			}			
			// #####################################################
        }
        else if (k==-2) // update morphological filtering while paused
        {
  			// ###################### MORPHOLOGICAL FILTERING #####################################			
			// apply morphological filtering:
			applyMorphology(fgmask, mmask, state.morph_size);

			// compute foreground image after morphological filtering:
			mimg = Scalar::all(0);
			roimat.copyTo(mimg, mmask);
			// ####################################################################################
			
			// compute and draw BBOx on foreground image: TODO: toggle or remove this; unneccesary computation
			RotatedRect fgbox = computeFgBBox(mmask);
			drawFgBBox(mimg, fgbox);
        }
        else if (k==-1)
        	break;
        else if (k==0)
        {
	        // obtain ROI slice from img:
			roimat = img(ROIrect);
        	// if the ROI size changed while the system was PAUSED, then we need to update the foreground images:	
		    if ((fgimg.size()!=roimat.size())&&(!roimat.empty())){
				//cout << "ROI size changed while PAUSED" << endl;
		    	fgimg.create(roimat.size(), roimat.type());
		    	//update the background model (learning, if active, otherwise simply compute new background image)
				bg_model->apply(roimat, fgmask, state.update_bg_model ? -1 : 0);
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
				roimat.copyTo(fgimg, fgmask);	//TODO: remove fgimg, and only use mimg...

				// compute foreground image after morphological filtering:
				mimg = Scalar::all(0);
				roimat.copyTo(mimg, mmask);
				// ####################################################################################
		
				// compute and draw BBOx on foreground image: TODO: toggle or remove this; unneccesary computation
				RotatedRect fgbox = computeFgBBox(mmask);
				drawFgBBox(mimg, fgbox);

				// compute background model image: TODO: toggle or remove this; unneccesary computation
				if (state.compute_bg_image == true)
					bg_model->getBackgroundImage(bgimg);			
		    }
		}
		if (!state.tracker_init_in_progress){
		    // update video displays:
		    //TODO: why are my coordinates messed up? The y-coords...
			updateMainDisplay(windows.main, img);
	   		//updateDisplay(windows.main, tmp, ROIrect, img.size());
			updateDisplay(windows.fgmask, fgmask, ROIrect, img.size());
			updateDisplay(windows.fgimg, fgimg, ROIrect, img.size());
			updateDisplay(windows.mmask, mmask, ROIrect, img.size());
			updateDisplay(windows.mimg, mimg, ROIrect, img.size());
			//TODO: consolidate background update and background update display for better performance...
			if(!bgimg.empty() && state.compute_bg_image==true)
				updateDisplay(windows.bgmodel, bgimg, ROIrect, img.size());
		}
		}
    return 0;
}

