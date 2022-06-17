f=imread('out90.png');
% Change sigma for different standard deviations e.g. 0.25, 0.5, 0.75 etc.
sigma = 0.5

w2 = fspecial('log',[3 3], sigma); 

% % Returns a rotationally symmetric Laplacian of Gaussian filter of size hsize with standard deviation sigma

filtered_img2=imfilter(f,w2,'replicate'); 
figure
imshow(filtered_img2);


% Detect scale invariant feature transform (SIFT) features and return SIFTPoints object
f2= rgb2gray(f)
filtered_img2 = rgb2gray(filtered_img2)
points = detectSIFTFeatures(f2)
points_2 = detectSIFTFeatures(filtered_img2)

% Overlay most salient features on 
figure
imshow(f)
hold on;
plot(points.selectStrongest(200))

% Overlay 200 most salient features on filtered image
figure
imshow(filtered_img2)
hold on;
plot(points_2.selectStrongest(200))

% Feature Matching Diagram Example

out93 = imread("out93.png");
out94 = imread("out94.png");
out93gray = rgb2gray(out93);
out94gray = rgb2gray(out94);

load stereoPointPairs
points93 = detectSIFTFeatures(out93gray)
points93 = points93.selectStrongest(400)
points94 = detectSIFTFeatures(out94gray)
points94 = points94.selectStrongest(400)

% Compute fundamental matrix
fRANSAC = estimateFundamentalMatrix(points93, ...
    points94,Method="RANSAC", ...
    NumTrials=2000,DistanceThreshold=1e-4)

figure;
showMatchedFeatures(out93,out94,points93, points94,'montage','PlotOptions',{'ro','go','b--'});
title('Putative Point Matches');