classdef SAR_LargeMotionError < handle
    properties
        BATCHSIZE
        fig
    end
    methods(Static)
    end
    methods
        function self = SAR_LargeMotionError()
            self.BATCHSIZE = 25;
            self.fig = figure();
        end
        
        function [pulse_contrib, imRecon, AntXYZ] = focusalg_BP(self, data, pulseIndex)
            pulse_contrib = zeros(size(data.x_mat));
            imRecon = pulse_contrib;
            if (pulseIndex < self.BATCHSIZE)
                return;
            end
            
            abs_pulseWindowIndices = (pulseIndex - (self.BATCHSIZE - 1)):pulseIndex;
            
            % Calculate the range to every bin in the range profile (m)
            dataWin.r_vec = linspace(-data.Nfft/2,data.Nfft/2-1,data.Nfft)*data.maxWr/data.Nfft;
            for accumIndex=1:pulseIndex
                % Form the range profile with zero padding added
                dataWin.rc(:,abs_pulseWindowIndices) = fftshift(ifft(data.phdata(:,abs_pulseWindowIndices),data.Nfft));
            end
            
            dataWin.x_mat = data.x_mat;
            dataWin.y_mat = data.y_mat;
            dataWin.z_mat = data.z_mat;
            dataWin.c = data.c;
            dataWin.R0 = data.R0(abs_pulseWindowIndices);
            dataWin.minF = data.minF(abs_pulseWindowIndices);
            dataWin.Np = self.BATCHSIZE;
            
            x = self.packUnknowns(data, pulseIndex);

            data.theta = asin(data.AntZ/data.R0);
            minimizationErrorFunction = @(x) -self.calculateEntropy(x, dataWin);
            %options = optimset('Display','iter','MaxIter',50,'PlotFcns',@optimplotfval);
            options = optimset('Display','iter', 'MaxIter', 200, 'TolX', 1e-7, 'MaxFunEvals', 200);
            %x_best = fminsearch(minimizationErrornction, x, options);
            x_best = fminsearch(minimizationErrorFunction, x, options);
            %x_best = x;
            
            optimData = self.unpackUnknowns(x_best);
            errorX = optimData.AntX' - data.AntX(abs_pulseWindowIndices);
            errorY = optimData.AntY' - data.AntY(abs_pulseWindowIndices);
            errorZ = optimData.AntZ' - data.AntZ(abs_pulseWindowIndices);
            for rel_pulseWindowIndex=1:self.BATCHSIZE
                % Calculate differential range for each pixel in the image (m)
                dR = sqrt((optimData.AntX(rel_pulseWindowIndex)-data.x_mat).^2 + ...
                    (optimData.AntY(rel_pulseWindowIndex)-data.y_mat).^2 + ...
                    (optimData.AntZ(rel_pulseWindowIndex)-data.z_mat).^2) - dataWin.R0(rel_pulseWindowIndex);
                
                % Calculate phase correction for image
                phCorr = exp(1i*4*pi*dataWin.minF(rel_pulseWindowIndex)*dR/data.c);
                
                % Determine which pixels fall within the range swath
                I = find(and(dR > min(dataWin.r_vec), dR < max(dataWin.r_vec)));
                
                % Update the image using linear interpolation
                pulse_contrib(I) = pulse_contrib(I) + interp1(dataWin.r_vec, dataWin.rc(:,rel_pulseWindowIndex), dR(I), 'linear') .* phCorr(I);
            end
        end
        
        function alphaXYZ = packUnknowns(self, data, numPulses)
            %x = zeros(3,1);
            %theta_az_start = atan2(data.AntY(1), data.AntX(1));
            %slant_range_to_target_phase_center = data.R0(numPulses);
            %x = [mean(diff(data.AntX(1:numPulses)));
            %mean(diff(data.AntY(1:numPulses)));
            %mean(diff(data.AntZ(1:numPulses)))];
            %[x, alphaX_CI, resX, rintX, statsX] = regress(data.AntX(pulseIndices)', [ones(numel(pulseIndices),1) pulseIndices pulseIndices.^2])
            pulseWindowIndices = (0:(self.BATCHSIZE-1))';
            posIndices = numPulses - ((self.BATCHSIZE-1):-1:0);
            monomialMatrix = [ones(numel(pulseWindowIndices),1) pulseWindowIndices pulseWindowIndices.^2];
            %[alphaX, alphaX_CI, resX] = regress(data.AntX(posIndices)', monomialMatrix);
            %[alphaY, alphaY_CI, resY] = regress(data.AntY(posIndices)', monomialMatrix);
            %[alphaZ, alphaZ_CI, resZ] = regress(data.AntZ(posIndices)', monomialMatrix);
            %mean(resX, resY, resZ)
            alphaXYZ = [ regress(data.AntX(posIndices)', monomialMatrix);
                regress(data.AntY(posIndices)', monomialMatrix);
                regress(data.AntZ(posIndices)', monomialMatrix)];
            
            %theta_az_start;
            %slant_range_to_target_phase_center];
            %x = [diff(data.AntX(1:2));
            %diff(data.AntY(1:2));
            %diff(data.AntZ(1:2))];% ones(numPulses-1,1)];
            % add theta / look angle
            % add R0 / distance to target phase center
        end
        
        function data = unpackUnknowns(self, x)
            %     data.AntX_vel = x(1);
            %     data.AntY_vel = x(2);
            %     data.AntZ_vel = x(3);
            %     data.AntX = data.AntX_vel*(0:(constData.Np - 1));
            %     data.AntY = data.AntY_vel*(0:(constData.Np - 1));
            %     data.AntZ = data.AntZ_vel*(0:(constData.Np - 1));
            pulseWindowIndices = (0:(self.BATCHSIZE-1))';
            monomialMatrix = [ones(numel(pulseWindowIndices),1) pulseWindowIndices pulseWindowIndices.^2];
            data.AntX = monomialMatrix*x(1:3);
            data.AntY = monomialMatrix*x(4:6);
            data.AntZ = monomialMatrix*x(7:9);
            %data.theta_az_start = x(4);
            %data.R0 = x(5);
        end
        
        function H = calculateEntropy(self, x, constData)
            
            optimData = self.unpackUnknowns(x);
            
            % optimData.globalAntX = constData.AntX(1);
            % optimData.globalAntY = 0;
            % optimData.globalAntZ = constData.AntZ(1);
            optimData.globalAntX = 0;
            optimData.globalAntY = 0;
            optimData.globalAntZ = 0;
            optimData.AntX = optimData.AntX + optimData.globalAntX;
            optimData.AntY = optimData.AntY + optimData.globalAntY;
            optimData.AntZ = optimData.AntZ + optimData.globalAntZ;
            
            imRecon = zeros(size(constData.x_mat));
            %imRecon = constData.im_final;
            
            for pulseIndex=1:self.BATCHSIZE
                
                % Calculate differential range for each pixel in the image (m)
                dR = sqrt((optimData.AntX(pulseIndex)-constData.x_mat).^2 + ...
                    (optimData.AntY(pulseIndex)-constData.y_mat).^2 + ...
                    (optimData.AntZ(pulseIndex)-constData.z_mat).^2) - constData.R0(pulseIndex);
                
                % Calculate phase correction for image
                phCorr = exp(1i*4*pi*constData.minF(pulseIndex)*dR/constData.c);
                
                % Determine which pixels fall within the range swath
                I = find(and(dR > min(constData.r_vec), dR < max(constData.r_vec)));
                
                % Update the image using linear interpolation
                imRecon(I) = imRecon(I) + interp1(constData.r_vec, constData.rc(:,pulseIndex), dR(I), 'linear') .* phCorr(I);
            end
            dyn_range = 70;
            Iout = uint8((255/dyn_range)*((20*log10(abs(imRecon)./...
                max(max(abs(imRecon))))) + dyn_range));

            set(0, 'CurrentFigure', self.fig);
            imshow(uint8(Iout));
            drawnow;
            
            H = 0;
            for col = 1:1:size(Iout,2)
                H = H + sum(entropy(Iout(:,col)));
            end
            Iout(Iout < 1e-6) = 0;
            H = H + 1e6*sum(Iout(Iout==0));
        end
        
    end
end