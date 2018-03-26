% rotate image

clc
close all
clear all

topFolderName='osa_data';
pho_cnt = [1e5, 1e6, 1e7, 1e8];

N = 100; % run N times simulation for each test
x = 100;
y = 100;
z = 100;

% plot config
caxis = [-3 7]; 

%% Test

for k=1:4
    dir_phn = sprintf('./%s/%1.0e', topFolderName, pho_cnt(k));
    for tid =1:N
        dir_phn_test = sprintf('%s/%d', dir_phn, tid);
        for imageID=1:y
            fname = sprintf('%s/osa_phn%1.0e_test%d_img%d.mat', dir_phn_test, pho_cnt(k), tid, imageID);
			disp(fname)
            result = load(fname);
            currentImage = result.currentImage;
            %size(currentImage)
            
            % plot image
            figure,imagesc(log10(currentImage),caxis);
            
            % 90 degree
            fname = sprintf('%s/osa_phn%1.0e_test%d_img%d_r90.mat', dir_phn_test, pho_cnt(k), tid, imageID);
            disp(fname)
            currentImage = rot90(result.currentImage, 1);
            figure,imagesc(log10(currentImage),caxis);
            feval('save', fname, 'currentImage');
            
            
            % NOTE: add the clean image too
            
            
            
            % 180 degree
            fname = sprintf('%s/osa_phn%1.0e_test%d_img%d_r180.mat', dir_phn_test, pho_cnt(k), tid, imageID);
            disp(fname)
            currentImage = rot90(result.currentImage, 2);
            figure,imagesc(log10(currentImage),caxis);
            feval('save', fname, 'currentImage');
            
            % 270 degree
            fname = sprintf('%s/osa_phn%1.0e_test%d_img%d_r270.mat', dir_phn_test, pho_cnt(k), tid, imageID);
            disp(fname)
            currentImage = rot90(result.currentImage, 3);
            figure,imagesc(log10(currentImage),caxis);
            feval('save', fname, 'currentImage');
            
            %fprintf('Generating %s\n',fname);
			%currentImage = squeeze(image3D(:,imageID,:));
			%feval('save', fname, 'currentImage');
			%%currentImage = squeeze(log10(image3D(:,imageID,:)));
			%%export_mcx_fig(fname, currentImage);

			break
        end
       
        break
    end
    
    break
end





%% Run

% for k=1:4
%     dir_phn = sprintf('./%s/%1.0e', topFolderName, pho_cnt(k));
%     for tid =1:N
%         dir_phn_test = sprintf('%s/%d', dir_phn, tid);
%         for imageID=1:y
%             fname = sprintf('%s/osa_phn%1.0e_test%d_img%d.mat', dir_phn_test, pho_cnt(k), tid, imageID);
% 			disp(fname)
%             result = load(fname);
%             currentImg = result.currentImage;
%             %size(currentImg)
%             
%             % plot image
%             figure,imagesc(log10(currentImg),caxis);
%             
%             % 90 degree
%             currentImg_90 = rot90(currentImg, 1)
%             figure,imagesc(log10(currentImg_90),caxis);
%             
%             % 180 degree
%             currentImg_180 = rot90(currentImg, 2)
%             figure,imagesc(log10(currentImg_180),caxis);
%             
%             % 270 degree
%             currentImg_270 = rot90(currentImg, 3)
%             figure,imagesc(log10(currentImg_270),caxis);
%             
%             %fprintf('Generating %s\n',fname);
% 			%currentImage = squeeze(image3D(:,imageID,:));
% 			%feval('save', fname, 'currentImage');
% 			%%currentImage = squeeze(log10(image3D(:,imageID,:)));
% 			%%export_mcx_fig(fname, currentImage);
% 
% 			break
%         end
%        
%         break
%     end
%     
%     break
% end


%% clean image


