clc;
clear;
addpath('./npy-matlab/npy-matlab') 
nx=300;
ny=300;
pml=10;
nr=32;
nt=300;
dt=0.002;
dx = 5;
dy = 5;

t = 0:dt:(nt-1)*dt;
t = t';

% read wavefields from numpy data format
wavefields = readNPY('../marmousi_2ms/wavefields_00000000_00000000.npy');
% subsample wavefield by 4
wavefields_subsample = wavefields(201:4:end,:,:);
% read velocity
vp = readNPY('velocity_00000000.npy');
ip=0;
for iy=1:ny-2*pml
    for ix=1:nx-2*pml
        ip=ip+1;
        X_star(ip,1) = (ix-1)*dx;
        X_star(ip,2) = (iy-1)*dy;
        vp_star(ip,1) = double(vp(ix+pml,iy+pml));
        for it=1:nt
            u_star(ip,it) = double(wavefields_subsample(it,ix+pml,iy+pml));
        end
    end
end

save('weq_homo_6.mat', 't', 'X_star', 'u_star','vp_star');

snap = u_star(:,20);
snap = reshape(snap,[nx-2*pml ny-2*pml]);
vpn = reshape(vp_star,[nx-2*pml ny-2*pml]);


figure();
imagesc(snap');
caxis([-1,1]);

figure();
imagesc(vpn');
set(gca,'YDir','normal');


