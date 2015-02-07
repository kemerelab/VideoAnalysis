
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
#include <iterator>
#include <algorithm>
#include <stdlib.h>     /* div, div_t */
#include <string>
#include <exception>

//#include <fcntl.h>   // F_OK
//#include <linux/fs.h> // file ACCESS
//#include <stdint.h>
#include <unistd.h> // for access()

#define HEADER_SIZE 256
#define MAX_FNAME_LENGTH 1000

//TODO: for multicamera implementation, and better performance control, use grab & retrieve methods instead of cap >> img0...
//TODO: add toggle for smoothing
//TODO: add GUI to change algorithms, load files, etc.
//TODO: add toggle button for performance mode---don't compute any unneccessary information and/or images.

using namespace std;
using namespace cv;

// numframes
// framenum 	x 	y 	type (m_anual,n_orat,a_uto)
// bookmarklist
// ratlist
// ratflist
// ROImask
// BGmodel
// system state: 
/*
class MyData //my video analysis project data//
{
public:
    MyData()
    {
    	cout << "MyData::MyData()" << endl;
    	//bookmarks = NULL;
    	ratflist.insert(ratflist.begin(),0);
		ratlist.insert(ratlist.begin(),false);
    }
    //explicit MyData(int) : A(97), X(CV_PI), id("mydata1234") // explicit to avoid implicit conversion
    //{}
    void write(FileStorage& fs) const                        //Write serialization for this class
    {
        fs << "bookmarks" << "[" << bookmarks << "]"; // << "ratflist" << ratflist << "ratlist" << ratlist << "}";
    }
    void read(const FileNode& node)                          //Read serialization for this class
    {
        bookmarks = (unsigned long)node["bookmarks"];
        X = (unsigned long)node["ratflist"];
        ratlist = (bool)node["ratlist"];
    }
public:   // Data Members
    vector<unsigned long> bookmarks;	// list of bookmarked frames // TODO: load and save bookmarks from and to file TODO: should this be double? we could have many frames...
	vector<unsigned long> ratflist;	// list of frame number where rat/no_rat signals are pinned
	vector<bool> ratlist;
};

//These write and read functions must be defined for the serialization in FileStorage to work
static void write(FileStorage& fs, const std::string&, const MyData& x)
{
    x.write(fs);
}
static void read(const FileNode& node, MyData& x, const MyData& default_value = MyData()){
    if(node.empty())
        x = default_value;
    else
        x.read(node);
}

// This function will print our custom class to the console
static ostream& operator<<(ostream& out, const MyData& m)
{
    out << "{ id = " << m.id << ", ";
    out << "X = " << m.X << ", ";
    out << "A = " << m.A << "}";
    return out;
}

*/

static int32_t currframe, maxframes, maxkeyframes, keyframe;	// TODO: move this into SystemState or elsewhere local in the main program
static vector<int32_t> bookmarks;	// list of bookmarked frames // TODO: load and save bookmarks from and to file TODO: should this be double? we could have many frames...
static vector<int32_t> ratflist;	// list of frame number where rat/no_rat signals are pinned
static vector<int32_t> ratlist;	// list of rat/no_rat pins 	// TODO: only need bool, but bool is not supported by YAML writer...
static std::vector<int32_t>::iterator prevratpin,nextratpin;
static vector<Point> traj;
static Point* trajArray;
static char* trajType;

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
	string filename;
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

bool ratInFrame(int framenumber){

//std::cout << "entering ratInFrame()" << endl;
/*
  low=std::lower_bound (v.begin(), v.end(), 20); //          ^
  up= std::upper_bound (v.begin(), v.end(), 20); //                   ^

  std::cout << "lower_bound at position " << (low- v.begin()) << '\n';
  std::cout << "upper_bound at position " << (up - v.begin()) << '\n';
*/
  	if (*prevratpin > framenumber){
  		prevratpin = std::lower_bound(ratflist.begin(), ratflist.end()-1, framenumber);
  	}
  	else
  	{
  		/*std::cout << "entering else clause" << endl;
  		std::cout << "framenumber: " << framenumber << endl;
  		std::cout << "begin(): " << *prevratpin << endl;
  		std::cout << "end(): " << *ratflist.end() << endl;*/
  		prevratpin = std::lower_bound(prevratpin, ratflist.end()-1, framenumber);
  	}
  	//std::cout << "ratInFrame::prevratpinA: " << *prevratpin << endl;
  	if ((*prevratpin > framenumber)&&(*prevratpin!=0)){
  		prevratpin=prevratpin-1;
	}
		//prevratpin = nextratpin--;
	//}

//	std::cout << "ratInFrame::prevratpinB: " << *prevratpin << endl;
	//std::cout << "ratInFrame::nextratpin: " << *nextratpin << endl;
//	std::cout << "leaving ratInFrame()" << endl;
	return (bool)ratlist[prevratpin - ratflist.begin()];

	//bookmarks.push_back(currframe-1);
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
	int lineWidth = 2;
	double alpha = 0.3;
	int beta = 0;
	Mat tmp = Mat::zeros(in.size(), in.type());

	// darken image:
	in.convertTo(tmp, -1, alpha, beta); 
	// apply bounding box to main image:
	in.copyTo(tmp,ROImask);  
	// draw ROI bbox:
	if (!ratInFrame(cap.get(CAP_PROP_POS_FRAMES)))
		rectangle(tmp, ROIrect, Scalar(170,0,255),lineWidth); // magenta border ==> no rat in frame
	else
		rectangle(tmp, ROIrect, Scalar(255,170,0),lineWidth); // cyan border ==> rat in frame
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
	ROImask = Mat::zeros(in.size(), CV_8UC1);
	ROImask(ROIrect) = 1;

	// display image in main window:
	updateMainDisplay(mp->window_title, mp->img);	// this call to updateMainDisplay() gives CORRECT coordinates, if used on its own, even with WINDOW_NORMAL
}

void replaceExt(string& s, const string& newExt) {

   string::size_type i = s.rfind('.', s.length());

   if (i != string::npos) {
      s.replace(i+1, newExt.length(), newExt);
   }
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
	if (k=='s'){

		int count = 0;
		char output_fname[MAX_FNAME_LENGTH];
		string new_output_fname;
		new_output_fname = state.filename;
		replaceExt(new_output_fname, "traj");
		cout << "output traj file:" << new_output_fname << endl;
		strncpy(output_fname, new_output_fname.c_str(), MAX_FNAME_LENGTH);

		FILE *output_fp;

		//TODO: give option to overwrite .traj file
		if ( access(output_fname, F_OK ) == 0 ) {
		     fprintf(stderr, "Output file already exists.\n");
		     displayOverlay(window_title,"Output .traj file already exists. Overwriting...",msgtimeout);
		     //return -2;
		}

		output_fp = fopen(output_fname, "w");
		if (output_fp == NULL) {
		    fprintf(stderr, "Error opening output file.\n");
		    displayOverlay(window_title,"Error opening output .traj file.",msgtimeout);
		    return -2;
		}

		count = fprintf(output_fp, "VideoAnalysis trajectory file v0.01\n");
		count += fprintf(output_fp, "%d Frames written in format: [int32_t frame number][int32_t pixel_x][int32_t pixel_y][char type]\n", maxframes);

		for (int i = 0; i < (HEADER_SIZE - count - 1); i++)
		      fprintf(output_fp, " ");
		fprintf(output_fp, "\n");

		/* WRITE OUT ACTUAL TRAJECTORY INFORMATION */

		if (maxframes>0){
			int res;
			double x,y;
			cout << "writing out stored trajectory..." << endl;
			for (int32_t ff=1;ff<maxframes;++ff){
				x = trajArray[ff].x;
				y = trajArray[ff].y;
				//cout << "trajArray[" << ff << "].x = " << x << endl;
				//cout << "trajArray[" << ff << "].y = " << y << endl;
				if (res = fwrite(&ff, sizeof(int32_t), 1, output_fp) != 1) {
      				fprintf(stderr, "Error in writing currframe to file.\n");
      				return -2;
    			}
				if (res = fwrite(&x, sizeof(double), 1, output_fp) != 1) {
      				fprintf(stderr, "Error in writing trajArray.x to file.\n");
      				return -2;
    			}
    			if (res = fwrite(&y, sizeof(double), 1, output_fp) != 1) {
      				fprintf(stderr, "Error in writing trajArray.y to file.\n");
      				return -2;
    			}
    			if ((trajType[ff]=='a')||(trajType[ff]=='n')||(trajType[ff]=='m')){
					if (res = fwrite(trajType+ff, sizeof(char), 1, output_fp) != 1) {
	      				fprintf(stderr, "Error in writing trajType to file.\n");
	      				return -2;
	    			}
    			}
    			else // untracked 
    				if (res = fwrite("u", sizeof(char), 1, output_fp) != 1) {
	      				fprintf(stderr, "Error in writing trajType to file.\n");
	      				return -2;
	    			}	
			}
	
		}

		
		fclose(output_fp);


		FileStorage fs("bookmarks.yml", FileStorage::WRITE);
		write(fs,"bookmarks",bookmarks);
		write(fs,"ratlist",ratlist);
		write(fs,"ratflist",ratflist);
		fs.release();

		/*
		vector<int> keypoints;
		keypoints.push_back(1);
		keypoints.push_back(2);

		FileStorage fs("keypoint1.yml", FileStorage::WRITE);
		write(fs , "keypoint", keypoints);
		fs.release();

		vector<int> newKeypoints;
		FileStorage fs2("keypoint1.yml", FileStorage::READ);
		FileNode kptFileNode = fs2["keypoint"];
		read(kptFileNode, newKeypoints);
		fs2.release();
		*/
		
		/*
		vector<int> keypoints;
		keypoints.push_back(3);
		keypoints.push_back(5);
		keypoints.push_back(2);
		FileStorage fs("keypoint1.yml", FileStorage::WRITE);
		write( fs , "keypoint", keypoints );
		fs.release();
		*/
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
	if (k=='r') // toggle rat in frame
	{
		try{
			currframe = cap.get(CAP_PROP_POS_FRAMES);
		//	cout << "rat toggled" << endl;
		//	std::cout << "prevratpin: " << *prevratpin << endl;
			if (*prevratpin < 0){
				cout << "invalid prevratpin detected!" << endl;
				throw 1;
			}
			//std::cout << "current frame: " << currframe << endl;
		
			if (ratInFrame(currframe)==true){
				int32_t offset = prevratpin - ratflist.begin();
				if (offset < 0){
					cout << "invalid offset detected!" << endl;
					throw 2;
				}
				if (currframe==*prevratpin){
		//			cout << "modifying rat boolean in position begin() + " << prevratpin - ratflist.begin()  << " to 0 (FALSE)" << endl;
					ratlist[offset] = 0;
				}
				else{
		//			cout << "adding new rat boolean in position begin() + " << prevratpin - ratflist.begin() + 1 << endl;
					ratlist.insert(ratlist.begin()+ (offset + 1),0);
					ratflist.insert(prevratpin+1,currframe);
				}
			}
			else{
				int32_t offset = prevratpin - ratflist.begin();
				if (offset < 0){
				//	cout << "invalid offset detected!" << endl;
					throw 2;
				}
				if (currframe==*prevratpin){
				//	cout << "modifying rat boolean in position begin() + " << prevratpin - ratflist.begin() << " to 1 (TRUE)" << endl;
					ratlist[offset] = 1;
				}
				else{
				//	cout << "adding new rat boolean in position begin() + " << prevratpin - ratflist.begin() + 1 << endl;
					ratlist.insert(ratlist.begin()+ (offset + 1),1);
					ratflist.insert(prevratpin+1,currframe);
				}
			}
			
			sprintf(overlaytext, "rat toggle");
			displayOverlay(window_title,overlaytext, msgtimeout);
		}
		catch(exception& e){
			cout << "unexpected error occured when toggling rat presence..." << e.what() << endl;
		}
		catch(...){
			cout << "unexpected error occured when toggling rat presence..." << endl;
		}
	
		 		
		//std::cout << "prevratpin: " << *prevratpin << endl;
		
		/*
		std::vector<int32_t>::iterator itrl;
		std::cout << "ratlist contains:";
  		for (itrl=ratlist.begin(); itrl<ratlist.end(); itrl++)
    		std::cout << ' ' << *itrl;
  		std::cout << '\n';

		std::vector<int32_t>::iterator it;
  		std::cout << "ratflist contains:";
  		for (it=ratflist.begin(); it<ratflist.end(); it++)
    		std::cout << ' ' << *it;
  		std::cout << '\n';
		*/

		if (!state.paused)
			return 1;
		else
			return -2; // update morphological fitering in paused state
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
			currframe = bookmarks[keyframe];
			cap.set(CAP_PROP_POS_FRAMES,currframe);
			sprintf(overlaytext, "keyframe #%d at frame #%d", keyframe+1, currframe+1);
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
			currframe = bookmarks[keyframe];
			cap.set(CAP_PROP_POS_FRAMES,currframe);
			sprintf(overlaytext, "keyframe #%d at frame #%d", keyframe+1, currframe+1);
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
			currframe = bookmarks[keyframe];
			cap.set(CAP_PROP_POS_FRAMES,currframe);
			sprintf(overlaytext, "keyframe #%d at frame #%d", keyframe+1, currframe+1);
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
			sprintf(overlaytext, "frame #%d", currframe+1);
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
			sprintf(overlaytext, "frame #%d", currframe+1);
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
			sprintf(overlaytext, "frame #%d", currframe+1);
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
			sprintf(overlaytext, "frame #%d", currframe+1);
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
			sprintf(overlaytext, "frame #%d", currframe+1);
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
			sprintf(overlaytext, "frame #%d", currframe+1);
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
		bookmarks.push_back(currframe-1); // add currently displayed frame
		//std::for_each(bookmarks.begin(), bookmarks.end(), displayValue);
		sprintf(overlaytext, "Added keyframe #%d at frame #%d", maxkeyframes, currframe);
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
		sprintf(overlaytext, "start of video; frame #%d", currframe+1);
		displayOverlay(window_title,overlaytext,msgtimeout);
		return 1;
	}
	if (k==87) // go to end of video (keypress: END)
	{
		currframe = maxframes-1;
		cap.set(CAP_PROP_POS_FRAMES,currframe);
		sprintf(overlaytext, "end of video; frame #%d", currframe+1);
		displayOverlay(window_title,overlaytext,msgtimeout);
		return 1;
	}
	if (k=='i') // print info about current frame and position
	{
		timeStruct myTime = getTime();	
		currframe = cap.get(CAP_PROP_POS_FRAMES);
		if (myTime.hour>0)
			sprintf(overlaytext, "current frame #%d\n current time: %d:%d:%d.%d", currframe, myTime.hour, myTime.minute, myTime.second, myTime.millisecond);
		else if (myTime.minute>0)
			sprintf(overlaytext, "current frame #%d\n current time: %d:%d.%d", currframe, myTime.minute, myTime.second, myTime.millisecond);
	else
			sprintf(overlaytext, "current frame #%d\n current time: %d.%d", currframe, myTime.second, myTime.millisecond);
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
	state.filename = "";
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
    state.filename = parser.get<string>("file_name");
    printf("file: %s\n",state.filename.c_str());
    string methodBG = parser.get<string>("methodBG");
    String methodTracker = parser.get<string>("methodTracker");

	cout << "methodBG = " << methodBG << endl;    
	cout << "methodTracker = " << methodTracker << endl;
	
	if (methodTracker.empty()){
		cout << "No tracking algorithm specified! Assume BOOSTING" << endl;
		methodTracker = "BOOSTING";
	}
	
    if (useCamera){
        cap.open(0);	// open default connected camera}
    	maxframes = -1;
    }
    else
	{
		cap.open(state.filename.c_str());	// open video file
		//cap.open(state.file);	// open video file
		maxframes = cap.get(CAP_PROP_FRAME_COUNT);	// determine number of frames in video file
		cout << "Total number of frames: " << maxframes << endl;
		try{
			trajArray = new Point[maxframes];
			trajType = new char[maxframes];
		}
  		catch (exception& e){
  			cout << "Standard exception when allocating memory for trajArray and trajType: " << e.what() << endl;
  		}	
	}
	
	parser.printMessage();

    if( !cap.isOpened() )
    {
        printf("can not open camera or video file\n");
        return -1;
    }
    
    Ptr<Tracker> tracker;

    /* load bookmarks.yml if it exists... */
    FileStorage fs("bookmarks.yml", FileStorage::READ);
    FileNode myBookmarksFileNode = fs["bookmarks"];
    FileNode myRatlistFileNode = fs["ratlist"];
    FileNode myRatflistFileNode = fs["ratflist"];
    read(myBookmarksFileNode, bookmarks);
    read(myRatlistFileNode, ratlist);
    read(myRatflistFileNode, ratflist);
	fs.release();

	maxkeyframes = bookmarks.size();
	cout << "bookmarks.size():" << maxkeyframes << endl;
	cout << "ratlist.size():" << ratlist.size() << endl;

	if (ratlist.size()==0){
		cout << "empty ratlist; assuming defaults..." << endl;
		//ratflist.insert(ratflist.begin(),2);
		//ratflist.insert(ratflist.begin(),1);
		ratflist.insert(ratflist.begin(),0);
		//ratflist.insert(ratflist.begin(),5);
		std::cout << "main::ratflist.begin(): " << *ratflist.begin() << endl;
		std::cout << "main::ratflist.end()-1: " << *(ratflist.end()-1) << endl;

		ratlist.insert(ratlist.begin(),0);
		//ratlist.insert(ratlist.begin(),false);
		//ratlist.insert(ratlist.begin(),false);
		prevratpin = ratflist.begin();
		//nextratpin = ratflist.end();
	}
	else
	{
		prevratpin = ratflist.begin();
		std::cout << "prevratpin: " << *prevratpin << endl;
		//std::cout << "nextratpin: " << *nextratpin << endl;
		std::vector<int32_t>::iterator itrl;
		std::cout << "ratlist contains:";
  		for (itrl=ratlist.begin(); itrl<ratlist.end(); itrl++)
    		std::cout << ' ' << *itrl;
  		std::cout << '\n';

		std::vector<int32_t>::iterator it;
  		std::cout << "ratflist contains:";
  		for (it=ratflist.begin(); it<ratflist.end(); it++)
    		std::cout << ' ' << *it;
  		std::cout << '\n';
	}



	/*
	double fps = cap.get(5); //get the frames per seconds of the video
	cout << "Frame per seconds : " << fps << endl;
	cout << "Source width: " << cap.get(CAP_PROP_FRAME_WIDTH) << endl;
	cout << "Source height: " << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
	*/
	
	// define default ROI mask (entire image)
	cap >> img0;	// get first frame from camera or video file
	//if (!img0.empty())
	//	cvtColor(img0, img0, COLOR_BGR2GRAY);	// convert to monochrome	//TODO: uncomment this!!! Only commented to test tracking...
	if (img0.empty())
		cout << "Unable to read from source!" << endl;
	resize(img0, img, Size(newwidth, newwidth*img0.rows/img0.cols), INTER_LINEAR);

	// defailt ROIrect is entire image:
	ROIrect.x = 0;
	ROIrect.y = 0;
	ROIrect.width = img.cols;	//TODO: this will only work for monocrhome images! Consider generalizing...
	ROIrect.height = img.rows;

	// update mask
	ROImask = Mat::zeros(img.size(), CV_8UC1);
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
        	if (img0.empty())
        		printf("img0 is empty!\n");
			//if (!img0.empty())	
			//	cvtColor(img0, img0, COLOR_BGR2GRAY);	// convert to monochrome //TODO: BUG: tracking seems to work only with color videos...
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
				displayOverlay(windows.main,"",1);
				cout << "***Attempting to intialize tracker...***\n";
				cout << img.size() << endl;

				// instantiates the specific Tracker
				tracker = Tracker::create(methodTracker);
				if (tracker==NULL)
				{
					cout << "***Error in the instantiation of the tracker...***\n";
					return -1;
				}
				
				//initializes the tracker TODO: trk_boundingBox should be shifted relative to frame
				if( !tracker->init(img, trk_boundingBox ) ) // DOES NOT WORK ON GRAYSCALE!!! TODO: investigate
				{
					cout << "***Could not initialize tracker...***\n";
					return -1;
				}
				currframe = cap.get(CAP_PROP_POS_FRAMES);
				state.tracker_initialized = true;
				mp.tracker_region_ready = false;
				cout << "***Tracker initialized successfully...***\n";
				trajArray[currframe].x = trk_boundingBox.x + (float)trk_boundingBox.width/2;
				trajArray[currframe].y = trk_boundingBox.y + (float)trk_boundingBox.height/2;
				trajType[currframe]='m';
			}
			else if (state.tracker_initialized && state.tracking)
			{
				currframe = cap.get(CAP_PROP_POS_FRAMES);	// TODO: this only works for video files, not live feeds...
				// check if rat is expected in frame:
				if (ratInFrame(currframe)){
					// check if current frame has been manually set or not
					if (trajType[currframe]=='m'){
						// re-initializes the tracker with manual coords:
						trk_boundingBox.x = trajArray[currframe].x - (float)trk_boundingBox.width/2;
						trk_boundingBox.y = trajArray[currframe].y - (float)trk_boundingBox.height/2;
						// instantiates the specific Tracker
						tracker = Tracker::create(methodTracker);
						if (tracker==NULL)
						{
							cout << "***Error in the instantiation of the tracker...***\n";
							return -1;
						}
						
						//initializes the tracker TODO: trk_boundingBox should be shifted relative to frame
						if( !tracker->init(img, trk_boundingBox ) ) // DOES NOT WORK ON GRAYSCALE!!! TODO: investigate
						{
							cout << "***Could not initialize tracker...***\n";
							return -1;
						}
						rectangle(img, trk_boundingBox, Scalar( 170, 255, 0 ), 2, 1 ); // green for manual				
					}
					else
					//updates the tracker
					if( tracker->update(img, trk_boundingBox ) )
					{
						double x,y;
						x = trk_boundingBox.x + (float)trk_boundingBox.width/2;
						y = trk_boundingBox.y + (float)trk_boundingBox.height/2;
						//cout << "(x,y) = (" << x << "," <<  y << ")" << endl;
						trajArray[currframe].x = x;
						trajArray[currframe].y = y;
						trajType[currframe] = 'a';	// automatically tracked
						rectangle(img, trk_boundingBox, Scalar( 0, 170, 255 ), 2, 1 ); // orange for auto
					}
				}
				else{
					trajType[currframe] = 'n';	// no rat in frame
					//rectangle(img, trk_boundingBox, Scalar( 170, 0, 255 ), 2, 1 ); // magenta for no_rat
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
	cout << "releasing memory for trajArray and trajType..." << endl;
	delete [] trajArray;
	delete [] trajType;
    return 0;
}

