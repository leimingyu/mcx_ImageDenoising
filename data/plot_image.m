caxis = [-3 7];

%figure,imagesc(squeeze(log10(image(:,50,:))),caxis);


load('./osa_data/1e+05/50/osa_phn1e+05_test50_img50.mat');
img_noisy = currentImage;
figure,imagesc(log10(img_noisy),caxis);

colorbar
xlabel('z axis'),ylabel('x axis')
title('image 50 (10^5) : noisy')     % check the image id

%%

load('./osa_data/1e+09/osa_1e9_img50.mat');
img_clean = currentImage;

figure,imagesc(log10(img_clean),caxis);
colorbar
xlabel('z axis'),ylabel('x axis')
title('image 50 (10^9) : clean')


%%

load('test_model_output.mat');
max(max(output_clean))
min(min(output_clean))

output_clean1 = output_clean;
pos = output_clean1 < 0.0;
output_clean1(pos) = 1e-8;


% output_clean_back = output_clean * 19029108.0;
output_clean_back = output_clean1 * 19029108.0;

figure,imagesc(log10(output_clean_back),caxis);
colorbar
xlabel('z axis'),ylabel('x axis')
title('image 50 (10^5) : model output')