clear all;
close all;


caxis = [-3 7];


%%  1e8

%------------
% image 1
%------------
%---
%noisy input
%---
load('../osa_data/1e+08/1/osa_phn1e+08_test1_img1.mat');
img_noisy = currentImage;
figure,imagesc(log10(img_noisy),caxis);

colorbar
xlabel('z axis'),ylabel('x axis')
title('image 1 (10^8) : noisy')     % check the image id


%---
% model input for 1e8 image1
%---

load('1e6model-1e8-log_test1_img1.mat');

% revert log(x + 1) = y  => x = exp(y) - 1
x = exp(output_clean) - 1;

max(max(x))
min(min(x))

pos = x < 0.0;
x(pos) = 1e-8;

figure,imagesc(log10(x),caxis);
colorbar
xlabel('z axis'),ylabel('x axis')
title('image 1 (10^8) with log : 1e6model')

%---
% clean 
%---
load('../osa_data/1e+09/osa_1e9_img1.mat');
img_clean = currentImage;

figure,imagesc(log10(img_clean),caxis);
colorbar
xlabel('z axis'),ylabel('x axis')
title('image 1 (10^9) : clean')



%------------
% image 50 
%------------

%---
%noisy input
%---
load('../osa_data/1e+08/1/osa_phn1e+08_test1_img50.mat');
img_noisy = currentImage;
figure,imagesc(log10(img_noisy),caxis);

colorbar
xlabel('z axis'),ylabel('x axis')
title('image 50 (10^8) : noisy')     % check the image id


%---
% model input
%---

load('1e6model-1e8-log_test1_img50.mat');

% revert log(x + 1) = y  => x = exp(y) - 1
x = exp(output_clean) - 1;

max(max(x))
min(min(x))

pos = x < 0.0;
x(pos) = 1e-8;

figure,imagesc(log10(x),caxis);
colorbar
xlabel('z axis'),ylabel('x axis')
title('image 50 (10^8) with log : 1e6model')

%---
% clean 
%---
load('../osa_data/1e+09/osa_1e9_img50.mat');
img_clean = currentImage;

figure,imagesc(log10(img_clean),caxis);
colorbar
xlabel('z axis'),ylabel('x axis')
title('image 50 (10^9) : clean')


%%  1e7

%------------
% image 1
%------------
%---
%noisy input
%---
load('../osa_data/1e+07/1/osa_phn1e+07_test1_img1.mat');
img_noisy = currentImage;
figure,imagesc(log10(img_noisy),caxis);

colorbar
xlabel('z axis'),ylabel('x axis')
title('image 1 (10^7) : noisy')     % check the image id


%---
% model input
%---

load('1e6model-1e7-log_test1_img1.mat');

% revert log(x + 1) = y  => x = exp(y) - 1
x = exp(output_clean) - 1;

max(max(x))
min(min(x))

pos = x < 0.0;
x(pos) = 1e-8;

figure,imagesc(log10(x),caxis);
colorbar
xlabel('z axis'),ylabel('x axis')
title('image 1 (10^7) with log : 1e6model ')

%---
% clean 
%---
load('../osa_data/1e+09/osa_1e9_img1.mat');
img_clean = currentImage;

figure,imagesc(log10(img_clean),caxis);
colorbar
xlabel('z axis'),ylabel('x axis')
title('image 1 (10^9) : clean')



%------------
% image 50 
%------------

%---
%noisy input
%---
load('../osa_data/1e+07/1/osa_phn1e+07_test1_img50.mat');
img_noisy = currentImage;
figure,imagesc(log10(img_noisy),caxis);

colorbar
xlabel('z axis'),ylabel('x axis')
title('image 50 (10^7) : noisy')     % check the image id


%---
% model input
%---

load('1e6model-1e7-log_test1_img50.mat');

% revert log(x + 1) = y  => x = exp(y) - 1
x = exp(output_clean) - 1;

max(max(x))
min(min(x))

pos = x < 0.0;
x(pos) = 1e-8;

figure,imagesc(log10(x),caxis);
colorbar
xlabel('z axis'),ylabel('x axis')
title('image 50 (10^7) with log : 1e6model')

%---
% clean 
%---
load('../osa_data/1e+09/osa_1e9_img50.mat');
img_clean = currentImage;

figure,imagesc(log10(img_clean),caxis);
colorbar
xlabel('z axis'),ylabel('x axis')
title('image 50 (10^9) : clean')


%%  1e6

%------------
% image 1
%------------
%---
%noisy input
%---
load('../osa_data/1e+06/1/osa_phn1e+06_test1_img1.mat');
img_noisy = currentImage;
figure,imagesc(log10(img_noisy),caxis);

colorbar
xlabel('z axis'),ylabel('x axis')
title('image 1 (10^6) : noisy')     % check the image id


%---
% model input
%---

load('1e6model-1e6-log_test1_img1.mat');

% revert log(x + 1) = y  => x = exp(y) - 1
x = exp(output_clean) - 1;

max(max(x))
min(min(x))

pos = x < 0.0;
x(pos) = 1e-8;

figure,imagesc(log10(x),caxis);
colorbar
xlabel('z axis'),ylabel('x axis')
title('image 1 (10^6) with log : 1e6model')

%---
% clean 
%---
load('../osa_data/1e+09/osa_1e9_img1.mat');
img_clean = currentImage;

figure,imagesc(log10(img_clean),caxis);
colorbar
xlabel('z axis'),ylabel('x axis')
title('image 1 (10^9) : clean')



%------------
% image 50 
%------------

%---
%noisy input
%---
load('../osa_data/1e+06/1/osa_phn1e+06_test1_img50.mat');
img_noisy = currentImage;
figure,imagesc(log10(img_noisy),caxis);

colorbar
xlabel('z axis'),ylabel('x axis')
title('image 50 (10^6) : noisy')     % check the image id


%---
% model input
%---

load('1e6model-1e6-log_test1_img50.mat');

% revert log(x + 1) = y  => x = exp(y) - 1
x = exp(output_clean) - 1;

max(max(x))
min(min(x))

pos = x < 0.0;
x(pos) = 1e-8;

figure,imagesc(log10(x),caxis);
colorbar
xlabel('z axis'),ylabel('x axis')
title('image 50 (10^6) with log : 1e6model')

%---
% clean 
%---
load('../osa_data/1e+09/osa_1e9_img50.mat');
img_clean = currentImage;

figure,imagesc(log10(img_clean),caxis);
colorbar
xlabel('z axis'),ylabel('x axis')
title('image 50 (10^9) : clean')


%%  1e5

%------------
% image 1
%------------
%---
%noisy input
%---
load('../osa_data/1e+05/1/osa_phn1e+05_test1_img1.mat');
img_noisy = currentImage;
figure,imagesc(log10(img_noisy),caxis);

colorbar
xlabel('z axis'),ylabel('x axis')
title('image 1 (10^5) : noisy')     % check the image id


%---
% model input
%---

load('1e6model-1e5-log_test1_img1.mat');

% revert log(x + 1) = y  => x = exp(y) - 1
x = exp(output_clean) - 1;

max(max(x))
min(min(x))

pos = x < 0.0;
x(pos) = 1e-8;

figure,imagesc(log10(x),caxis);
colorbar
xlabel('z axis'),ylabel('x axis')
title('image 1 (10^5) with log : 1e6model ')

%---
% clean 
%---
load('../osa_data/1e+09/osa_1e9_img1.mat');
img_clean = currentImage;

figure,imagesc(log10(img_clean),caxis);
colorbar
xlabel('z axis'),ylabel('x axis')
title('image 1 (10^9) : clean')



%------------
% image 50 
%------------

%---
%noisy input
%---
load('../osa_data/1e+05/1/osa_phn1e+05_test1_img50.mat');
img_noisy = currentImage;
figure,imagesc(log10(img_noisy),caxis);

colorbar
xlabel('z axis'),ylabel('x axis')
title('image 50 (10^5) : noisy')     % check the image id


%---
% model input
%---

load('1e6model-1e5-log_test1_img50.mat');

% revert log(x + 1) = y  => x = exp(y) - 1
x = exp(output_clean) - 1;

max(max(x))
min(min(x))

pos = x < 0.0;
x(pos) = 1e-8;

figure,imagesc(log10(x),caxis);
colorbar
xlabel('z axis'),ylabel('x axis')
title('image 50 (10^5) with log : 1e6model ')

%---
% clean 
%---
load('../osa_data/1e+09/osa_1e9_img50.mat');
img_clean = currentImage;

figure,imagesc(log10(img_clean),caxis);
colorbar
xlabel('z axis'),ylabel('x axis')
title('image 50 (10^9) : clean')



