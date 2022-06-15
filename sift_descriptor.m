f=imread('out90.png')
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