%% ----------------------------- SUMMARY ----------------------------------

% This code generates a "spider plot" (plots tracks, all starting from the
% same point for visualization purposes); from an input table of track data
% generated from the ImageJ TrackMate PlugIn.

%% ----------------------- GENERATE PLOT ----------------------------------
data_raw = zorb_003; % user assign 'data_raw' as imported TrackMate table


data_concat = [table2array(data_raw(:,10)), table2array(data_raw(:,12)), table2array(data_raw(:,13))];
track_IDs = table2array(unique([data_raw(:,10)]));

xpos_stored = [];
ypos_stored = [];

    for i = 1:length(track_IDs(~isnan(track_IDs)))
        x_pos = data_concat(data_concat(:,1) == track_IDs(i),2); % grab x-positions
        y_pos = data_concat(data_concat(:,1) == track_IDs(i),3); % grab y-positions
    
        xpos_norm = x_pos-(x_pos(1,1));
        ypos_norm = y_pos-(y_pos(1,1));

        xpos_stored = [xpos_stored; xpos_norm] % store each replicate
        ypos_stored = [ypos_stored; ypos_norm]
    
        plot(xpos_norm,ypos_norm,'r','LineWidth',1)
        xlim([-500 500])
        ylim([-500 500])

        xlabel('µm')
        ylabel('µm')
        hold on
    end


%% ------------------- PLOT MULTIPLE REPLICATES ---------------------------

plot(xpos_stored,ypos_stored,'Color','r','LineWidth',1)
axis off
hold on
xlabel('µm')
ylabel('µm')
fontname(gca,'Arial')
set(gca,'FontSize',20)

% exportgraphics(gcf,'transparent.pdf',...   % since R2020a
%     'ContentType','vector',...
%     'BackgroundColor','none')