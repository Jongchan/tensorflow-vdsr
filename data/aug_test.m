
target = 'Set14';
dataDir = fullfile('./', target);
count = 0;
f_lst = dir(fullfile(dataDir, '*.bmp'));
folder = fullfile('test', target);
mkdir(folder);
for f_iter = 1:numel(f_lst)
%     disp(f_iter);
    f_info = f_lst(f_iter);
    if f_info.name == '.'
        continue;
    end
    f_path = fullfile(dataDir,f_info.name);
    disp(f_path);
    img_raw = imread(f_path);
    if size(img_raw,3)==3
        img_raw = rgb2ycbcr(img_raw);
        img_raw = img_raw(:,:,1);
%     else
%         img_raw = rgb2ycbcr(repmat(img_raw, [1 1 3]));
    end
    
    img_raw = im2double(img_raw);
    
    img_size = size(img_raw);
    width = img_size(2);
    height = img_size(1);
    
    img_raw = img_raw(1:height-mod(height,12),1:width-mod(width,12),:);
    
    img_size = size(img_raw);
    
    img_2 = imresize(imresize(img_raw,1/2,'bicubic'),[img_size(1),img_size(2)],'bicubic');
    img_3 = imresize(imresize(img_raw,1/3,'bicubic'),[img_size(1),img_size(2)],'bicubic');
    img_4 = imresize(imresize(img_raw,1/4,'bicubic'),[img_size(1),img_size(2)],'bicubic');
    
    patch_name = sprintf('%s/%d',folder,count);
    
    save(patch_name, 'img_raw');
    save(sprintf('%s_2', patch_name), 'img_2');
    save(sprintf('%s_3', patch_name), 'img_3');
    save(sprintf('%s_4', patch_name), 'img_4');
    
    count = count + 1;
    display(count);
    
    
end
