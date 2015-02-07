%% readTraj.m
% (C) Copyright Etienne Ackermann 2015 (with Eric Lewis)
% reads VideoAnalysis .traj files and displays the trajectory

clear
clc
clf
close all

OUTPUT_HEADER_SIZE = 256;
myfile = '/home/etienne/kemerelab-stuff/Code/VidAnalysis/example_data/snooker.traj';

fid = fopen(myfile);

% reaf and display file header:
hdr = transpose(fread(fid, OUTPUT_HEADER_SIZE, 'char*1=>char'))

% extract number of frames from file header:
C = strsplit(hdr,'\n');
C2 = strsplit(C{2});
Nt = str2num(C2{1})-1;   % number of frames

clear hdr OUTPUT_HEADER_SIZE C C2 myfile

%% allocate memory
fnum = zeros(Nt,1);
px = zeros(Nt,1);
py = zeros(Nt,1);
ttype = char(zeros(Nt,1));


%% read data from file (parse)
for nn=1:Nt
    fnum(nn) = fread(fid, 1, 'int32');
    px(nn) = fread(fid, 1, 'double');
    py(nn) = fread(fid, 1, 'double');
    ttype(nn) = fread(fid, 1, 'char*1=>char');
end % for nn

fclose(fid);

clear nn fid ans

%% modify trajectory

% replace traj points witht NaNs wherever frames are untracked:
idxu = (ttype=='u');
px(idxu) = NaN;
py(idxu) = NaN;

% replace traj points witht NaNs wherever frames do not contain a rat:
idxnr = (ttype=='n');
px(idxnr) = NaN;
py(idxnr) = NaN;

clear idxnr idxu ttype

%% Display trajectory
load('example_data/snookershot'); % frame 185
imshow(snookershot); hold on;
plot(px,py,'--w', 'LineWidth',1); hold off
%axis image; axis ij; xlim([0 320]); ylim([0 240]);

%% animate trajectory
k = 2;
kstart=20;
for kk=kstart:numel(px)
    tempx = px(kk-k+1:kk);
    tempy = py(kk-k+1:kk);
    imshow(snookershot); hold on;
    plot(px,py,'--w');
    plot(tempx,tempy,'-o', 'Color','w'); hold off;
    axis image; axis ij; ylim([0 240]); xlim([0 320]);
    %title(sprintf('frame %g',kk))
    drawnow;
    %pause(0.03);
end %for

