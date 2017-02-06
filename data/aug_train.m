



dataDir = '291';%fullfile('data', '291');
mkdir('train');
count = 0;
f_lst = [];
f_lst = [f_lst; dir(fullfile(dataDir, '*.jpg'))];
f_lst = [f_lst; dir(fullfile(dataDir, '*.bmp'))];
for f_iter = 1:numel(f_lst)
%     disp(f_iter);
    f_info = f_lst(f_iter);
    if f_info.name == '.'
        continue;
    end
    f_path = fullfile(dataDir,f_info.name);
    img_raw = imread(f_path);
    img_raw = rgb2ycbcr(img_raw);
    img_raw = im2double(img_raw(:,:,1));
    
    img_size = size(img_raw);
    width = img_size(2);
    height = img_size(1);
    
    img_raw = img_raw(1:height-mod(height,12),1:width-mod(width,12),:);
    
    img_size = size(img_raw);
    patch_size = 41;
    stride = 41;
    x_size = (img_size(2)-patch_size)/stride+1;
    y_size = (img_size(1)-patch_size)/stride+1;
    
    img_2 = imresize(imresize(img_raw,1/2,'bicubic'),[img_size(1),img_size(2)],'bicubic');
    img_3 = imresize(imresize(img_raw,1/3,'bicubic'),[img_size(1),img_size(2)],'bicubic');
    img_4 = imresize(imresize(img_raw,1/4,'bicubic'),[img_size(1),img_size(2)],'bicubic');
    
    for x = 0:x_size-1
        for y = 0:y_size-1
            x_coord = x*stride; y_coord = y*stride; 
            patch_name = sprintf('train/%d',count);
            
            patch = imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0);
            save(patch_name, 'patch');
            patch = imrotate(img_2(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0);
            save(sprintf('%s_2', patch_name), 'patch');
            patch = imrotate(img_3(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0);
            save(sprintf('%s_3', patch_name), 'patch');
            patch = imrotate(img_4(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0);
            save(sprintf('%s_4', patch_name), 'patch');
            
            count = count+1;
            
            patch_name = sprintf('train/%d',count);
            
            patch = imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90);
            save(patch_name, 'patch');
            patch = imrotate(img_2(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90);
            save(sprintf('%s_2', patch_name), 'patch');
            patch = imrotate(img_3(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90);
            save(sprintf('%s_3', patch_name), 'patch');
            patch = imrotate(img_4(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90);
            save(sprintf('%s_4', patch_name), 'patch');
            
            count = count+1;
            
            patch_name = sprintf('train/%d',count);
            
            patch = fliplr(imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0));
            save(patch_name, 'patch');
            patch = fliplr(imrotate(img_2(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0));
            save(sprintf('%s_2', patch_name), 'patch');
            patch = fliplr(imrotate(img_3(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0));
            save(sprintf('%s_3', patch_name), 'patch');
            patch = fliplr(imrotate(img_4(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0));
            save(sprintf('%s_4', patch_name), 'patch');
            
            count = count+1;
            
            patch_name = sprintf('train/%d',count);
            
            patch = fliplr(imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90));
            save(patch_name, 'patch');
            patch = fliplr(imrotate(img_2(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90));
            save(sprintf('%s_2', patch_name), 'patch');
            patch = fliplr(imrotate(img_3(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90));
            save(sprintf('%s_3', patch_name), 'patch');
            patch = fliplr(imrotate(img_4(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90));
            save(sprintf('%s_4', patch_name), 'patch');
            
            count = count+1;
            
            
            %{
            patch_name = sprintf('aug/%d',count);
            
            patch = imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 180);
            save(patch_name, 'patch');
            patch = imrotate(img_2(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 180);
            save(sprintf('%s_2', patch_name), 'patch');
            patch = imrotate(img_3(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 180);
            save(sprintf('%s_3', patch_name), 'patch');
            patch = imrotate(img_4(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 180);
            save(sprintf('%s_4', patch_name), 'patch');
            
            count = count+1;
            
            patch_name = sprintf('aug/%d',count);
            
            patch = fliplr(imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 180));
            save(patch_name, 'patch');
            patch = fliplr(imrotate(img_2(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 180));
            save(sprintf('%s_2', patch_name), 'patch');
            patch = fliplr(imrotate(img_3(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 180));
            save(sprintf('%s_3', patch_name), 'patch');
            patch = fliplr(imrotate(img_4(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 180));
            save(sprintf('%s_4', patch_name), 'patch');
            
            count = count+1;
            
            patch_name = sprintf('aug/%d',count);
            
            patch = imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 270);
            save(patch_name, 'patch');
            patch = imrotate(img_2(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 270);
            save(sprintf('%s_2', patch_name), 'patch');
            patch = imrotate(img_3(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 270);
            save(sprintf('%s_3', patch_name), 'patch');
            patch = imrotate(img_4(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 270);
            save(sprintf('%s_4', patch_name), 'patch');
            
            count = count+1;
            
            patch_name = sprintf('aug/%d',count);
            
            patch = fliplr(imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 180));
            save(patch_name, 'patch');
            patch = fliplr(imrotate(img_2(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 180));
            save(sprintf('%s_2', patch_name), 'patch');
            patch = fliplr(imrotate(img_3(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 180));
            save(sprintf('%s_3', patch_name), 'patch');
            patch = fliplr(imrotate(img_4(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 180));
            save(sprintf('%s_4', patch_name), 'patch');
            
            count = count+1;
            %}
        end
    end
    
    display(count);
    
    
end
