%========================================================================
% A very simple Navier-Stokes solver for a drop falling in a rectangular
% box. A forward in time, centered in space discretization is used.
% The density is advected by a front tracking scheme.
% This version uses a stretched grid and ENO for the advection
%========================================================================
%domain size and physical variables
Lx=1.0;Ly=1.0;gx=0.0;gy=100.0; rho1=1.0; rho2=2.0;
m1=0.01; m2=0.05; sigma=10; rro=rho1;
un=0;us=0;ve=0;vw=0;time=0.0;
rad=0.15;xc=0.5;yc=0.7; % Initial drop size and location
% Numerical variables
nx=32;ny=32;dt=0.00125;nstep=300;maxit=200;maxError=0.001;beta=1.2;
% Zero various arrys
u=zeros(nx+1,ny+2); v=zeros(nx+2,ny+1); p=zeros(nx+2,ny+2);
ut=zeros(nx+1,ny+2); vt=zeros(nx+2,ny+1); tmp1=zeros(nx+2,ny+2);
uu=zeros(nx+1,ny+1); vv=zeros(nx+1,ny+1); tmp2=zeros(nx+2,ny+2);
fx=zeros(nx+2,ny+2); fy=zeros(nx+2,ny+2);
dk=zeros(1,nx+2); dl=zeros(1,ny+2); dkh=zeros(1,nx+1); dlh=zeros(1,ny+1);
ux=zeros(nx+2,ny+2); uy=zeros(nx+2,ny+2);
vx=zeros(nx+2,ny+2); vy=zeros(nx+2,ny+2);
% Set the stretched grid
for i=1:nx+2, s=Lx*(i-1.5)/(nx); x(i)=1.5*s*(0.5-s)*(1-s)+s; end;
for j=1:ny+2, s=Ly*(j-1.5)/(ny); y(j)=1.5*s*(0.5-s)*(1-s)+s; end;
for i=1:nx+1, xh(i)=0.5*(x(i+1)+x(i)); end;
for j=1:ny+1, yh(j)=0.5*(y(j+1)+y(j)); end
for i=1:nx+1, dkh(i)=x(i+1)-x(i); end;
for j=1:ny+1, dlh(j)=y(j+1)-y(j); end;
for i=2:nx+1, dk(i)=xh(i)-xh(i-1); end;
for j=2:ny+1, dl(j)=yh(j)-yh(j-1); end;
dk(1)=dk(2);dk(nx+2)=dk(nx+1); dl(1)=dl(2);dl(ny+2)=dl(ny+1);
% Set density and viscosity in the domain and the drop
r=zeros(nx+2,ny+2)+rho1;m=zeros(nx+2,ny+2)+m1;
for i=2:nx+1,for j=2:ny+1;
        if ( (x(i)-xc)^2+(y(j)-yc)^2 < rad^2), r(i,j)=rho2;m(i,j)=m2; end,
    end,end
%================== SETUP THE FRONT ===================
Nf=100; xf=zeros(1,Nf+2);yf=zeros(1,Nf+2);
uf=zeros(1,Nf+2);vf=zeros(1,Nf+2);
tx=zeros(1,Nf+2);ty=zeros(1,Nf+2);
for l=1:Nf+2, xf(l)=xc-rad*sin(2.0*pi*(l-1)/(Nf));
    yf(l)=yc+rad*cos(2.0*pi*(l-1)/(Nf));end
%------------find the mapped front coordinates--------
sxf=zeros(1,Nf+2);syf=zeros(1,Nf+2);
for l=1:Nf+2,
    for i=1:nx+1
        if xf(l) > x(i+1), % DO NOTHING
        else
            sxf(l)=i*(x(i+1)-xf(l))/dkh(i)+(i+1)*(xf(l)-x(i))/dkh(i);break
        end
    end
    for j=1:ny+1
        if yf(l) >= y(j+1), % DO NOTHING
        else
            syf(l)=j*(y(j+1)-yf(l))/dlh(j)+(j+1)*(yf(l)-y(j))/dlh(j);break
        end
    end
end
%================== START TIME LOOP======================================
for is=1:nstep,is
    %------------------ FIND SURFACE TENSION --------------
    fx=zeros(nx+2,ny+2);fy=zeros(nx+2,ny+2); % Set fx & fy to zero
    for l=1:Nf+1,
        ds=sqrt((xf(l+1)-xf(l))^2+(yf(l+1)-yf(l))^2);
        tx(l)=(xf(l+1)-xf(l))/ds;
        ty(l)=(yf(l+1)-yf(l))/ds; % Tangent vectors
    end
    tx(Nf+2)=tx(2);ty(Nf+2)=ty(2);
    for l=2:Nf+1
        nfx=sigma*(tx(l)-tx(l-1));nfy=sigma*(ty(l)-ty(l-1));
        ip=floor(sxf(l)-0.5);jp=floor(syf(l));
        ax=sxf(l)-0.5-ip;ay=syf(l)-jp;
        fx(ip,jp) =fx(ip,jp)+(1.0-ax)*(1.0-ay)*nfx/dkh(ip)/dl(jp);
        fx(ip+1,jp) =fx(ip+1,jp)+ax*(1.0-ay)*nfx/dkh(ip+1)/dl(jp);
        fx(ip,jp+1) =fx(ip,jp+1)+(1.0-ax)*ay*nfx/dkh(ip)/dl(jp+1);
        fx(ip+1,jp+1)=fx(ip+1,jp+1)+ax*ay*nfx/dkh(ip+1)/dl(jp+1);
        ip=floor(sxf(l));jp=floor(syf(l)-0.5);
        ax=sxf(l)-ip;ay=syf(l)-0.5-jp;
        fy(ip,jp) =fy(ip,jp)+(1.0-ax)*(1.0-ay)*nfy/dk(ip)/dlh(jp);
        fy(ip+1,jp) =fy(ip+1,jp)+ax*(1.0-ay)*nfy/dk(ip+1)/dlh(jp);
        fy(ip,jp+1) =fy(ip,jp+1)+(1.0-ax)*ay*nfy/dk(ip)/dlh(jp+1);
        fy(ip+1,jp+1)=fy(ip+1,jp+1)+ax*ay*nfy/dk(ip+1)/dlh(jp+1);
    end
    %---------------------------------------------------------
    % tangential velocity at boundaries
    u(1:nx+1,1)=2*us-u(1:nx+1,2);u(1:nx+1,ny+2)=2*un-u(1:nx+1,ny+1);
    v(1,1:ny+1)=2*vw-v(2,1:ny+1);v(nx+2,1:ny+1)=2*ve-v(nx+1,1:ny+1);
    for i=2:nx-1,for j=2:ny+1; % finding the ENO values for the face velocities
            ss=0.5*sign(u(i+1,j)+u(i,j));
            ux(i+1,j)=(0.5+ss)*(u(i,j)+0.5*minabs((u(i+1,j)-u(i,j)),(u(i,j)-u(i-1,j))) )...
                +(0.5-ss)*(u(i+1,j)-0.5*minabs((u(i+2,j)-u(i+1,j)),(u(i+1,j)-u(i,j))) );
        end,end
    ux(2,2:ny+1)=0.5*u(2,2:ny+1)+u(1,2:ny+1); ux(nx+1,2:ny+1)=0.5*u(nx,2:ny+1)+u(nx+1,2:ny+1); %Bdry
    for i=2:nx,for j=3:ny;
            ss=0.5*sign(v(i+1,j)+v(i,j));
            uy(i,j+1)=(0.5+ss)*(u(i,j)+0.5*minabs((u(i,j+1)-u(i,j)),(u(i,j)-u(i,j-1))) )...
                +(0.5-ss)*(u(i,j+1)-0.5*minabs((u(i,j+2)-u(i,j+1)),(u(i,j+1)-u(i,j))) );
        end,end
    uy(1:nx+1,2)=0.0;uy(1:nx+1,ny+1)=0.0; % Bottom and Top
    for i=2:nx,for j=2:ny+1 % TEMPORARY u-velocity-ADVECTION
            ut(i,j)=u(i,j)+dt*(-(u(i,j)*(ux(i+1,j)-ux(i,j))/dkh(i)+...
                0.25*(v(i,j-1)+v(i+1,j-1)+v(i+1,j)+v(i,j))*(uy(i,j+1)-uy(i,j))/dl(j) )+...
                fx(i,j)/(0.5*(r(i+1,j)+r(i,j))) ...
                - (1.0 -rro/(0.5*(r(i+1,j)+r(i,j))) )*gx );
        end,end
    for i=3:nx,for j=2:ny; % finding the ENO values for the face velocities
            ss=0.5*sign(u(i,j+1)+u(i,j));
            vx(i+1,j)=(0.5+ss)*(v(i,j)+0.5*minabs((v(i+1,j)-v(i,j)),(v(i,j)-v(i-1,j))) )...
                +(0.5-ss)*(v(i+1,j)-0.5*minabs((v(i+2,j)-v(i+1,j)),(v(i+1,j)-v(i,j))) );
        end,end
    vx(2,1:ny+1)=0.0;vx(nx+1,1:ny+1)=0.0; % Left and Right
    for i=2:nx+1,for j=2:ny-1;
            ss=0.5*sign(v(i,j+1)+v(i,j));
            vy(i,j+1)=(0.5+ss)*(v(i,j)+0.5*minabs((v(i,j+1)-v(i,j)),(v(i,j)-v(i,j-1))) )...
                +(0.5-ss)*(v(i,j+1)-0.5*minabs((v(i,j+2)-v(i,j+1)),(v(i,j+1)-v(i,j))) );
        end,end
    vy(2:nx+1,2)=0.5*v(2:nx+1,2)+v(2:nx+1,1); vy(2:nx+1,ny+1)=0.5*v(2:nx+1,ny)+v(2:nx+1,ny+1); %Bdry
    for i=2:nx+1,for j=2:ny % TEMPORARY v-velocity-ADVECTION
            vt(i,j)=v(i,j)+dt*(-(0.25*(u(i-1,j)+u(i,j)+u(i,j+1)+...
                u(i-1,j+1))*(vx(i+1,j)-vx(i,j))/dk(i)+ ...
                v(i,j)*(vy(i,j+1)-vy(i,j))/dlh(j))+ ...
                fy(i,j)/(0.5*(r(i,j+1)+r(i,j))) ...
                - (1.0 -rro/(0.5*(r(i,j+1)+r(i,j))) )*gy );
        end,end
    for i=2:nx,for j=2:ny+1 % TEMPORARY u-velocity-DIFFUSION
            ut(i,j)=ut(i,j)+dt*(...
                (1./dkh(i))*2.*(m(i+1,j)*(1./dk(i+1))*(u(i+1,j)-u(i,j)) - ...
                m(i,j) *(1./dk(i))*(u(i,j)-u(i-1,j)) ) ...
                +(1./dl(j))*( 0.25*(m(i,j)+m(i+1,j)+m(i+1,j+1)+m(i,j+1))* ...
                ((1./dlh(j))*(u(i,j+1)-u(i,j)) + (1./dkh(i))*(v(i+1,j)-v(i,j)) ) - ...
                0.25*(m(i,j)+m(i+1,j)+m(i+1,j-1)+m(i,j-1))* ...
                ((1./dlh(j-1))*(u(i,j)-u(i,j-1))+ (1./dkh(i-1))*(v(i+1,j-1)- v(i,j-1))) )...
                )/(0.5*(r(i+1,j)+r(i,j)) );
        end,end
    for i=2:nx+1,for j=2:ny % TEMPORARY v-velocity-DIFFUSION
            vt(i,j)=vt(i,j)+dt*(...
                (1./dk(i))*( 0.25*(m(i,j)+m(i+1,j)+m(i+1,j+1)+m(i,j+1))* ...
                ((1./dlh(j))*(u(i,j+1)-u(i,j)) + (1./dkh(i))*(v(i+1,j)-v(i,j)) ) - ...
                0.25*(m(i,j)+m(i,j+1)+m(i-1,j+1)+m(i-1,j))* ...
                ((1./dlh(j))*(u(i-1,j+1)-u(i-1,j))+ (1./dkh(i-1))*(v(i,j)- v(i-1,j))) )...
                +(1./dlh(j))*2.*(m(i,j+1)*(1./dl(j+1))*(v(i,j+1)-v(i,j)) - ...
                m(i,j) *(1./dl(j))*(v(i,j)-v(i,j-1)) ) ...
                )/(0.5*(r(i,j+1)+r(i,j)) );
        end,end
    %========================================================================
    % Compute source term and the coefficient for p(i,j)
    rt=r; lrg=1000;
    rt(1:nx+2,1)=lrg;rt(1:nx+2,ny+2)=lrg;
    rt(1,1:ny+2)=lrg;rt(nx+2,1:ny+2)=lrg;
    for i=2:nx+1,for j=2:ny+1
            tmp1(i,j)= (0.5/dt)*( (ut(i,j)-ut(i-1,j))/dk(i)+(vt(i,j)-vt(i,j-1))/dl(j) );
            tmp2(i,j)=1.0/( (1./dk(i))*( 1./(dkh(i)*(rt(i+1,j)+rt(i,j)))+...
                1./(dkh(i-1)*(rt(i-1,j)+rt(i,j))) )+...
                (1./dl(j))*(1./(dlh(j)*(rt(i,j+1)+rt(i,j)))+...
                1./(dlh(j-1)*(rt(i,j-1)+rt(i,j))) ) );
        end,end
    for it=1:maxit % SOLVE FOR PRESSURE
        oldArray=p;
        for i=2:nx+1,for j=2:ny+1
                p(i,j)=(1.0-beta)*p(i,j)+beta* tmp2(i,j)*(...
                    (1./dk(i))*( p(i+1,j)/(dkh(i)*(rt(i+1,j)+rt(i,j)))+...
                    p(i-1,j)/(dkh(i-1)*(rt(i-1,j)+rt(i,j))) )+...
                    (1./dl(j))*( p(i,j+1)/(dlh(j)*(rt(i,j+1)+rt(i,j)))+...
                    p(i,j-1)/(dlh(j-1)*(rt(i,j-1)+rt(i,j))) ) - tmp1(i,j));
            end,end
        if max(max(abs(oldArray-p))) <maxError, break,end
    end
    for i=2:nx,for j=2:ny+1 % CORRECT THE u-velocity
            u(i,j)=ut(i,j)-dt*(2.0/dkh(i))*(p(i+1,j)-p(i,j))/(r(i+1,j)+r(i,j));
        end,end
    for i=2:nx+1,for j=2:ny % CORRECT THE v-velocity
            v(i,j)=vt(i,j)-dt*(2.0/dlh(j))*(p(i,j+1)-p(i,j))/(r(i,j+1)+r(i,j));
        end,end
    %================== ADVECT FRONT =====================
    for l=2:Nf+1
        ip=floor(sxf(l)-0.5);jp=floor(syf(l)); ax=sxf(l)-0.5-ip;ay=syf(l)-jp;
        uf(l)=(1.0-ax)*(1.0-ay)*u(ip,jp)/dkh(ip)+ax*(1.0-ay)*u(ip+1,jp)/dkh(ip+1)+...
            (1.0-ax)*ay*u(ip,jp+1)/dkh(ip)+ax*ay*u(ip+1,jp+1)/dkh(ip+1);
        ip=floor(sxf(l));jp=floor(syf(l)-0.5); ax=sxf(l)-ip;ay=syf(l)-0.5-jp;
        vf(l)=(1.0-ax)*(1.0-ay)*v(ip,jp)/dlh(jp)+ax*(1.0-ay)*v(ip+1,jp)/dlh(jp)+...
            (1.0-ax)*ay*v(ip,jp+1)/dlh(jp+1)+ax*ay*v(ip+1,jp+1)/dlh(jp+1);
    end
    for i=2:Nf+1, sxf(i)=sxf(i)+dt*uf(i); syf(i)=syf(i)+dt*vf(i);end %MOVE THE FRONT
    sxf(1)=sxf(Nf+1);syf(1)=syf(Nf+1);sxf(Nf+2)=sxf(2);syf(Nf+2)=syf(2);
    %------------ Add points to the front ------------
    sxfold=sxf;syfold=syf; j=1;
    for l=2:Nf+1
        ds=sqrt( (sxfold(l)-sxf(j))^2 + (syfold(l)-syf(j))^2);
        if (ds > 0.5)
            j=j+1;sxf(j)=0.5*(sxfold(l)+sxf(j-1));syf(j)=0.5*(syfold(l)+syf(j-1));
            j=j+1;sxf(j)=sxfold(l);syf(j)=syfold(l);
        elseif (ds < 0.25)
            % DO NOTHING!
        else
            j=j+1;sxf(j)=sxfold(l);syf(j)=syfold(l);
        end
    end
    Nf=j-1;
    sxf(1)=sxf(Nf+1);syf(1)=syf(Nf+1);sxf(Nf+2)=sxf(2);syf(Nf+2)=syf(2);
    %-----------find the coordinates in real space-------------
    for l=2:Nf+1
        ip=floor(sxf(l));jp=floor(syf(l));
        xf(l)=(ip+1-sxf(l))*x(ip)+(sxf(l)-ip)*x(ip+1);
        yf(l)=(jp+1-syf(l))*y(jp)+(syf(l)-jp)*y(jp+1);
    end
    xf(1)=xf(Nf+1);yf(1)=yf(Nf+1);xf(Nf+2)=xf(2);yf(Nf+2)=yf(2);
    %------------ distribute gradient --------------
    fx=zeros(nx+2,ny+2);fy=zeros(nx+2,ny+2); % Set fx & fy to zero
    for l=2:Nf+1
        nfx=-0.5*(yf(l+1)-yf(l-1))*(rho2-rho1);
        nfy=0.5*(xf(l+1)-xf(l-1))*(rho2-rho1); % Normal vector
        ip=floor(sxf(l)-0.5);jp=floor(syf(l));
        ax=sxf(l)-0.5-ip;ay=syf(l)-jp;
        fx(ip,jp) =fx(ip,jp)+(1.0-ax)*(1.0-ay)*nfx/dkh(ip)/dl(jp);
        fx(ip+1,jp) =fx(ip+1,jp)+ax*(1.0-ay)*nfx/dkh(ip+1)/dl(jp);
        fx(ip,jp+1) =fx(ip,jp+1)+(1.0-ax)*ay*nfx/dkh(ip)/dl(jp+1);
        fx(ip+1,jp+1)=fx(ip+1,jp+1)+ax*ay*nfx/dkh(ip+1)/dl(jp+1);
        ip=floor(sxf(l));jp=floor(syf(l)-0.5);
        ax=sxf(l)-ip;ay=syf(l)-0.5-jp;
        fy(ip,jp) =fy(ip,jp)+(1.0-ax)*(1.0-ay)*nfy/dk(ip)/dlh(jp);
        fy(ip+1,jp) =fy(ip+1,jp)+ax*(1.0-ay)*nfy/dk(ip+1)/dlh(jp);
        fy(ip,jp+1) =fy(ip,jp+1)+(1.0-ax)*ay*nfy/dk(ip)/dlh(jp+1);
        fy(ip+1,jp+1)=fy(ip+1,jp+1)+ax*ay*nfy/dk(ip+1)/dlh(jp+1);
    end
    %------------ construct the density --------------
    r=zeros(nx+2,ny+2)+rho1;
    for iter=1:maxit
        oldArray=r;
        for i=2:nx+1,for j=2:ny+1
                r(i,j)=0.25*(r(i+1,j)+r(i-1,j)+r(i,j+1)+r(i,j-1)+...
                    dkh(i-1)*fx(i-1,j)-dkh(i)*fx(i,j)+...
                    dlh(j-1)*fy(i,j-1)-dlh(j)*fy(i,j));
            end,end
        if max(max(abs(oldArray-r))) <maxError, break,end
    end
    %------------ update the viscosity --------------
    m=zeros(nx+2,ny+2)+m1;
    for i=2:nx+1,for j=2:ny+1
            m(i,j)=m1+(m2-m1)*(r(i,j)-rho1)/(rho2-rho1);
        end,end
    %========================================================================
    time=time+dt % plot the results
    uu(1:nx+1,1:ny+1)=0.5*(u(1:nx+1,2:ny+2)+u(1:nx+1,1:ny+1));
    vv(1:nx+1,1:ny+1)=0.5*(v(2:nx+2,1:ny+1)+v(1:nx+1,1:ny+1));
    hold off,contour(x,y,flipud(rot90(r))),axis equal,axis([0 Lx 0 Ly]);
    hold on;quiver(xh,yh,flipud(rot90(uu)),flipud(rot90(vv)),'r');
    plot(xf(1:Nf),yf(1:Nf),'k','linewidth',5);pause(0.01)
end
function x=minabs(a,b)
x=0.5*(sign(abs(b)-abs(a))*(a-b)+a+b);
end