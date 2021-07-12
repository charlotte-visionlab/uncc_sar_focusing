classdef VideoResultCreator < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        filename
        framerate
        resolution
        frameIndex
        imSize=[];
        matrixRows
        matrixCols
        imageRows
        imageCols
        imageComps
        imageMatrix
        fontsize
        writerObj        
    end
    
    methods
        function obj = VideoResultCreator(filename, framerate, rows, cols)
            obj.framerate = framerate;
            obj.filename = filename;
            % create the video writer with 1 fps
            obj.writerObj = VideoWriter(filename);
            obj.writerObj.FrameRate = framerate;
            % open the video writer
            obj.writerObj.open();            
            obj.matrixRows = rows;
            obj.matrixCols = cols;
            for r=1:obj.matrixRows
                for c=1:obj.matrixCols
                    obj.imageMatrix(r,c).occupied = false;
                    obj.imageMatrix(r,c).titleString = "";
                    obj.imageMatrix(r,c).imageData = [];
                end
            end
            obj.fontsize = 25;
            %obj.imageArr(1,1).titleString = 'Input Image';
            %obj.imageArr(1,2).titleString = 'Output Image';
        end
        
        function writtenImage = addImage(obj, image, r, c)
            writtenImage = [];
            obj.imageMatrix(r,c).occupied = true;
            if (isempty(obj.imSize))
                obj.imSize = size(image);
                obj.imageRows = obj.imSize(1);
                obj.imageCols = obj.imSize(2);
                if (length(obj.imSize)==3)
                    obj.imageComps = obj.imSize(3);
                else
                    obj.imageComps = 1;
                end
                boolVal = true;
            end
            if (any(size(image) ~= obj.imSize))
                obj.imageMatrix(r,c).imageData = imresize(image, obj.imSize);
            else
                obj.imageMatrix(r,c).imageData = image;
            end
            if (all([obj.imageMatrix(:,:).occupied]) == true)
                writtenImage = obj.writeFrame();
            end
        end
        
        function setTitleString(obj, str, r, c)
             obj.imageMatrix(r,c).titleString = str;
        end
        
        function setFontSize(obj, fontsize)
            obj.fontsize = fontsize;
        end
        
        function composite_image = writeFrame(obj, image)
            COLUMN_PADDING = 10;
            ROW_PADDING = 10;
            composite_image = [];
            for r=1:obj.matrixRows
                composite_col_image = [];
                for c=1:obj.matrixCols
                    obj.imageMatrix(r,c).occupied = false;
                    text_str = obj.imageMatrix(r,c).titleString;
                    box_color = 'black';
                    text_color = 'yellow';
                    imageWithText = insertText(obj.imageMatrix(r,c).imageData, ...
                        [obj.imageCols/2, obj.imageRows], text_str, ...
                        'AnchorPoint','CenterBottom', 'FontSize', obj.fontsize, ...
                        'BoxColor', box_color,'BoxOpacity', 0.2, ...
                        'TextColor', text_color);
                    comps = size(imageWithText,3);
                    if (c < obj.matrixCols)
                        padded_values = zeros(obj.imageRows, COLUMN_PADDING , comps);
                    else
                        padded_values = [];
                    end
                    composite_col_image = [composite_col_image, ...
                        imageWithText, padded_values];
                end
                if (r > 1)
                    padded_values = zeros(ROW_PADDING, size(composite_col_image,2), comps);
                else
                    padded_values = [];
                end                
                composite_image = [composite_image; ...
                    padded_values; composite_col_image];
                %size(composite_image)
            end
            % convert the image to a frame
            frame = im2frame(composite_image);
            obj.writerObj.writeVideo(frame);
        end
        function close(obj)
            % close the writer object
            obj.writerObj.close();
        end
        
    end
    
end

