%% ------------------------- SUMMARY --------------------------------------

% This code will extract zorbs and any corresponding co-zorb cores from
% timelapse movies to allow tracking zorb/co-zorb number and morphologies 
% changes over time.

% INPUTS:
% 1) moviePath; directory to where .tif movie is located, currently set up
%to be 3-channel with BF, zorb channel, core channel

% 2) numChans; currently 3, could potentially be adjusted given input
% timelapse


%% ------------------------ LOAD MOVIE -----------------------------------

clear
close all

moviePath = '/Volumes/Jons SSD/2023.11.21_zorbs_diff_medias/spot18_TSB50.tif';
%savePath = '/Users/jonschrope/Library/CloudStorage/Box-Box/co_zorbing_paper/analysis/abx_temporal_results/Fj_Ec_001/';

% if ~exist(savePath)
%        mkdir(savePath)
% end

numChans = 3;

info = imfinfo(moviePath);
numFrames = length(info);
frames = cell(numFrames,1);
for k = 1 : numFrames
    frames{k,1} = imread(moviePath, k);
end

totalFrames = numFrames/numChans;
frames2analyze = 1:(totalFrames-1);

%% ------------------------ MANUALLY CROP ---------------------------------
% Performed on first frame, will apply to all frames, thus assuming no xy drift

close all
    % Draw circular mask around first frame
    t0 = 1;
    chan1_image = frames{t0,1};
    chan2_image = frames{t0+1,1};

%     figure;
    inrange1 = stretchlim(chan1_image,.035); % saturate 1% of pixels (default is 2%)
    image1_lut = imadjust(chan1_image,inrange1);
%     imshow(image1_final)

%     figure;
    inrange2 = stretchlim(chan2_image,.003); % saturate 1% of pixels (default is 2%)
    image2_lut = imadjust(chan2_image,inrange2);
%     imshow(image2_final)

    two_chan_im = imfuse(image1_lut,image2_lut);

    [rows, columns, ~] = size(two_chan_im);
    %subplot(2, 2, 1);

    imshow(two_chan_im)
    axis('on', 'image');
    fontSize = 14;
    title('Original Image', 'FontSize', fontSize);
    % Maximize the window to make it easier to draw.
    g = gcf;
    g.WindowState = 'maximized';

%     im_bin = imbinarize(originalImage);
%     imagesc(im_bin)
    %roi = drawcircle

    %uiwait(helpdlg('Please click and drag out a circle.'));
	h = drawcircle('Color','k','FaceAlpha',0.4);
    
    % Get coordinates of the circle.
    angles = linspace(0, 2*pi, 10000);
    x = cos(angles) * h.Radius + h.Center(1);

    y = sin(angles) * h.Radius + h.Center(2);
    % Show circle over image.
        %subplot(2, 2, 1);
        %imagesc(chan1_image);
        %axis('on', 'image');
        %subplot(2, 2, 2);
        %imagesc(chan1_image);
        %axis('on', 'image');
        %hold on;
        %plot(x, y, 'r-', 'LineWidth', 2);
        %title('Original image with circle mask overlaid', 'FontSize', fontSize);
    % Get a mask of the circle
    mask = poly2mask(x, y, rows, columns);
        %subplot(2, 2, 3);
        %imagesc(mask);
        %axis('on', 'image');
        %title('Circle Mask', 'FontSize', fontSize);
    % Mask the image with the circle.
    maskedImage1 = chan1_image; % Initialize with the entire image.
    maskedImage1(~mask) = 0; % Zero image outside the circle mask.

    maskedImage2 = chan2_image; % Initialize with the entire image.
    maskedImage2(~mask) = 0; % Zero image outside the circle mask.

    % Mask the image.
    maskedImage1 = bsxfun(@times, chan1_image, cast(mask, class(chan1_image)));
    maskedImage2 = bsxfun(@times, chan2_image, cast(mask, class(chan2_image)));

    % Crop the image to the bounding box.
    props = regionprops(mask, 'BoundingBox');
    maskedImage1 = imcrop(maskedImage1, props.BoundingBox);
    maskedImage2 = imcrop(maskedImage2, props.BoundingBox);
    
    % Display it in the lower right plot.
        %subplot(2, 2, 4);

     % auto adjust LUTs
    inrange1 = stretchlim(maskedImage1,.02); % saturate 1% of pixels (default is 2%)
    image1_final = imadjust(maskedImage1,inrange1);
        %imshow(image1_final)

    inrange2 = stretchlim(maskedImage2,.01); % saturate 1% of pixels (default is 2%)
    image2_final = imadjust(maskedImage2,inrange2);

    %imshow(image1_final)
    two_chan_im = imfuse(image1_final,image2_final);
%     figure;
%     imshow(two_chan_im);

%% ANALYSIS FOR ALL FRAMES

counter = 0; % count number of loops
core_fracs_out = cell(numFrames/numChans,1);

for t = frames2analyze %2:numChans:numFrames
    disp(t);
    
    % define channels at thhat time point (assuming 3 channels here I think)
    chan1_image = frames{(t*3)+2,1};
    chan2_image = frames{(t*3)+3,1};

%     figure;
%     subplot (1,2,1)
%     imagesc(chan1_image)
%     subplot (1,2,2)
%     imagesc(chan2_image)

    % Below, Repeat similar process above, applying that first user-defined mask to crop every frame 

    % Mask the image with the circle.
    maskedImage1 = chan1_image; % Initialize with the entire image.
    maskedImage1(~mask) = 0; % Zero image outside the circle mask.

    maskedImage2 = chan2_image; % Initialize with the entire image.
    maskedImage2(~mask) = 0; % Zero image outside the circle mask.

    % Mask the image.
    maskedImage1 = bsxfun(@times, chan1_image, cast(mask, class(chan1_image)));
    maskedImage2 = bsxfun(@times, chan2_image, cast(mask, class(chan2_image)));

    % Crop the image to the bounding box.
    props = regionprops(mask, 'BoundingBox');
    maskedImage1 = imcrop(maskedImage1, props.BoundingBox);
    maskedImage2 = imcrop(maskedImage2, props.BoundingBox);
    
    % Display it in the lower right plot.
        %subplot(2, 2, 4);

     % auto adjust LUTs
    inrange1 = stretchlim(maskedImage1,.02); % saturate 1% of pixels (default is 2%)
    image1_final = imadjust(maskedImage1,inrange1);
        %imshow(image1_final)

    inrange2 = stretchlim(maskedImage2,.01); % saturate 1% of pixels (default is 2%)
    image2_final = imadjust(maskedImage2,inrange2);
   

    %imshow(image1_final)
    %close all
    two_chan_im = imfuse(image1_final,image2_final);

% figure;
%     imshow(two_chan_im);
%     title('Cropped Image')
%     hold on

% FIND OBJECTS IN IMAGE

    windowSize = 10;
    kernel = ones(windowSize) / windowSize ^ 2;
    bw1 = imbinarize(image1_final,.9);
    blurryImage = conv2(single(bw1), kernel, 'same');
    bw1_smooth = blurryImage > 0.5; % Rethreshold
    bw1_proc = bwareaopen(bw1_smooth,80);
    CC1 = bwconncomp(bw1_proc);
    props1 = regionprops(bw1_proc,'Centroid', 'Area', 'Circularity', 'Perimeter', 'PixelIdxList','PixelList');
    perims1 = bwperim(bw1_proc,8);

% figure; %temp for troublshooting
%  subplot(1,2,1)
%  imshow(bw1_smooth)
%  subplot(1,2,2)
%  imshow(bw1_proc)

    bw2 = imbinarize(image2_final,.9);
    blurryImage = conv2(single(bw2), kernel, 'same');
    bw2_smooth = blurryImage > 0.5; % Rethreshold
    bw2_proc = bwareaopen(bw2_smooth,80);
    CC2 = bwconncomp(bw2_proc);
    props2 = regionprops(bw2_proc,'Centroid', 'Area', 'Circularity', 'Perimeter', 'PixelIdxList','PixelList');
    perims2 = bwperim(bw2_proc,8);

% figure;
%    subplot(1,3,1)
%    imshow(bw1_proc)
%    subplot(1,3,2)
%    imshow(bw2_proc)
% 
%    perim_combined = perims1+perims2;
%    se = strel('disk',2);
%    perim_combined_dil = imdilate(perim_combined,se);
%    B2 = labeloverlay(two_chan_im,perim_combined_dil,'Colormap',[1 0 0; 0 0.9 0]);
% 
%   subplot(1,3,3)
%     imshow(B2)

%print([savePath 'frame_' num2str(t)],'-dpng')
%pause(1)
close all

% FIND CIRCLES IN IMAGE

% figure;
%     imshow(two_chan_im);
%     title('Cropped Image')
%     hold on
% 
    [centers1, radii1] = imfindcircles(bw1_proc,[20 100],'ObjectPolarity','bright');
    [centers2, radii2] = imfindcircles(bw2_smooth,[20 100],'ObjectPolarity','bright');
% 
%     viscircles(centers2, radii2,'Color','b');
%     hold on
%     viscircles(centers1, radii1,'Color','r');
%     axis('on', 'image');

    %print([savePath 'frame_' num2str(t)],'-dpng'
    %close all

    areas1 = radii1*pi*2; % Fj
    areas2 = radii2*pi*2; % 2nd Bug (Ec or Bc)

% CO-LOCALIZE CIRCLES TO QUANTIFY CORE AREA FRACTION
    key = [];
    kk = 0; % counter
        for i = 1:length(radii1)
            for j = 1:length(radii2)
                kk = kk + 1;
                dist = norm(centers1(i,:) - centers2(j,:));
                if dist < 50
                    key = [key;[i,j]];
                end
            end
        end

        % KEY: First row is Fj, second row is 2nd bug. Now we know what circle
        % in centers1 corresponds to the 2nd bug circle in circles2
    
    % Quantify core area and area fractions
    core_frac = zeros(size(key,1),1);
    areas = zeros(size(key,1),2);
    for m = 1:size(key,1)
        core_frac(m) = areas2(key(m,2))/areas1(key(m,1)); %divide first column by second column
        areas(m,:) = [areas2(key(m,2)), areas1(key(m,1))];
    end

% this is storage all values, independent of whether the core overlaps with
% the Fj... 

    OUTPUT(t).Fj_props = props1;
    OUTPUT(t).Core_props = props2;
    OUTPUT(t).Fj_area_means = mean([OUTPUT(t).Fj_props.Area]);
    OUTPUT(t).Core_area_means = mean([OUTPUT(t).Core_props.Area]);

    OUTPUT(t).Fj_circ_means = mean([OUTPUT(t).Fj_props.Circularity]);
    OUTPUT(t).Core_circ_means = mean([OUTPUT(t).Core_props.Circularity]);

    OUTPUT(t).Fj_number = length(props1); % total number of objects
    OUTPUT(t).Core_number = length(props2);

    means_out{t,1} = areas;
    means_out{t,2} = core_frac;
    means_out{t,3} = size(key,1);
end % time loop

%% PLOT SIZE

tvec = frames2analyze*.5;

figure;
subplot (1,3,1)
plot(tvec, smoothdata([OUTPUT.Fj_number]),'r','LineWidth', 2);
hold on
plot(tvec, smoothdata([OUTPUT.Core_number]),'c','LineWidth', 2);
xlabel('time (hrs)')
ylabel('number')

subplot (1,3,2)
plot(tvec, smoothdata([OUTPUT.Fj_area_means]),'r','LineWidth', 2);
hold on
plot(tvec, smoothdata([OUTPUT.Core_area_means]),'c','LineWidth', 2);
xlabel('time (hrs)')
ylabel('size (pix^2)')

subplot (1,3,3)
plot(tvec, smoothdata([OUTPUT.Fj_circ_means]),'r','LineWidth', 2);
hold on
plot(tvec, smoothdata([OUTPUT.Core_circ_means]),'c','LineWidth', 2);
xlabel('time (hrs)')
ylabel('circularity')

x0=10;
y0=500;
width=1200;
height=400;
set(gcf,'position',[x0,y0,width,height])

%print(savePath,'-dpng')

%% PLOT CORE FRACTIONS

MEAN_CORE_FRAC_OUT = zeros(size(core_fracs_out,1),1);
for n = 1:size(core_fracs_out,1)
    if cellfun(@isempty,core_fracs_out(n,1)) == 1 % check if cell is empty at that timepoint
        MEAN_CORE_FRAC_OUT(n) = 0;
    else
        MEAN_CORE_FRAC_OUT(n) = mean(core_fracs_out{n,1}(:,1));
    end
end


savePath = '/Users/jonschrope/Library/CloudStorage/Box-Box/co_zorbing_paper/analysis/media_temporal_results';
save([savePath filesep 'spot18_TSB50'],'OUTPUT');

