%===============================================================
% CodeC2-frt.m
% A very simple Navier-Stokes solver for a drop falling in a
% rectangular box, using a conservative form of the equations. 
% A first-order explicit projection method and centered in space 
% discretizationa are used. The marker function is advected by 
% Tront Tracking. Last edited 7/6/2016
% https://www.multiphaseflowdns.com/
%===============================================================

Lx=1.0;Ly=1.0;gx=0.0;gy=-100.0;  % Domain size and
rho1=1.0; rho2=2.0; m0=0.01;     % physical variables
unorth=0;usouth=0;veast=0;vwest=0;time=0.0; 
rad=0.15;xc=0.5;yc=0.7; % Initial drop size and location

%-------------------- Numerical variables ----------------------
nx=32;ny=32;dt=0.00125;nstep=400; maxit=200;maxError=0.001;beta=1.5; Nf=100;

%-------------------- Zero various arrys -----------------------
u=zeros(nx+1,ny+2);  v=zeros(nx+2,ny+1);  p=zeros(nx+2,ny+2);
ut=zeros(nx+1,ny+2); vt=zeros(nx+2,ny+1); tmp1=zeros(nx+2,ny+2); 
uu=zeros(nx+1,ny+1); vv=zeros(nx+1,ny+1); tmp2=zeros(nx+2,ny+2);
r=zeros(nx+2,ny+2); chi=zeros(nx+2,ny+2);
xf=zeros(1,Nf+2); yf=zeros(1,Nf+2); 
uf=zeros(1,Nf+2); vf=zeros(1,Nf+2);
dx=Lx/nx;dy=Ly/ny;                          % Set the grid 
for i=1:nx+2; x(i)=dx*(i-1.5);end; for j=1:ny+2; y(j)=dy*(j-1.5);end;

%-------------------- Initial Conditions -----------------------
r=zeros(nx+2,ny+2)+rho1;                       % Set density
for i=2:nx+1,for j=2:ny+1;                     % for the domain and the drop
  if((x(i)-xc)^2+(y(j)-yc)^2 <rad^2),r(i,j)=rho2;chi(i,j)=1.0;end, 
end,end
                                          
for l=1:Nf+2, xf(l)=xc-rad*sin(2.0*pi*(l-1)/(Nf));      % Initialize 
              yf(l)=yc+rad*cos(2.0*pi*(l-1)/(Nf));end   % the Front
                                          
hold off,contour(x,y,chi'),axis equal,axis([0 Lx 0 Ly]);
hold on;plot(xf(1:Nf),yf(1:Nf),'k','linewidth',3);pause(0.01)               

%---------------------- START TIME LOOP ------------------------
for is=1:nstep,is

%---------------------- Advect the Front -----------------------
	for l=2:Nf+1                       % Interpolate the Front Velocities
      ip=floor(xf(l)/dx)+1; jp=floor((yf(l)+0.5*dy)/dy)+1;
      ax=xf(l)/dx-ip+1;ay=(yf(l)+0.5*dy)/dy-jp+1;	   
      uf(l)=(1.0-ax)*(1.0-ay)*u(ip,jp)+ax*(1.0-ay)*u(ip+1,jp)+...
		             (1.0-ax)*ay*u(ip,jp+1)+ax*ay*u(ip+1,jp+1);
						
      ip=floor((xf(l)+0.5*dx)/dx)+1; jp=floor(yf(l)/dy)+1;
      ax=(xf(l)+0.5*dx)/dx-ip+1;ay=yf(l)/dy-jp+1;
	  vf(l)=(1.0-ax)*(1.0-ay)*v(ip,jp)+ax*(1.0-ay)*v(ip+1,jp)+...
		             (1.0-ax)*ay*v(ip,jp+1)+ax*ay*v(ip+1,jp+1);
    end     

    for i=2:Nf+1, xf(i)=xf(i)+dt*uf(i); yf(i)=yf(i)+dt*vf(i);end % Move the
	xf(1)=xf(Nf+1);yf(1)=yf(Nf+1);xf(Nf+2)=xf(2);yf(Nf+2)=yf(2); % Front
 	
%-------------- Update the marker function ---------------------
    d(2:nx+1,2:ny+1)=2;
    
    for l=2:Nf+1
      nfx=-(yf(l+1)-yf(l))/dx;   
      nfy=(xf(l+1)-xf(l))/dy;  % Normal vector
      ds=sqrt(nfx*nfx+nfy*nfy); nfx=nfx/ds; nfy=nfy/ds;
      xfront=0.5*(xf(l)+xf(l+1)); yfront=0.5*(yf(l)+yf(l+1));
      ip=floor((xfront+0.5*dx)/dx)+1; jp=floor((yfront+0.5*dy)/dy)+1;

      d1=sqrt(((xfront-  x(ip))/dx)^2+((yfront-  y(jp))/dy)^2);
      d2=sqrt(((xfront-x(ip+1))/dx)^2+((yfront-  y(jp))/dy)^2);
      d3=sqrt(((xfront-x(ip+1))/dx)^2+((yfront-y(jp+1))/dy)^2);
      d4=sqrt(((xfront-  x(ip))/dx)^2+((yfront-y(jp+1))/dy)^2);

      if d1<d(ip,jp), d(ip,jp)=d1;...
        dn1=(x(ip)-  xfront)*nfx/dx+(y(jp)-  yfront)*nfy/dy;
        chi(ip,jp)=    0.5*(1.0+sign(dn1)); 
        if abs(dn1)<0.5, chi(ip,jp)=   0.5+dn1; end;
      end
      if d2<d(ip+1,jp), d(ip+1,jp)=d2;...
        dn2=(x(ip+1)-xfront)*nfx/dx+(y(jp)-  yfront)*nfy/dy;
        chi(ip+1,jp)=  0.5*(1.0+sign(dn2));
        if abs(dn2)<0.5, chi(ip+1,jp)=  0.5+dn2; end;
      end
      if d3<d(ip+1,jp+1), d(ip+1,jp+1)=d3;...
        dn3=(x(ip+1)-xfront)*nfx/dx+(y(jp+1)-yfront)*nfy/dy;
        chi(ip+1,jp+1)=0.5*(1.0+sign(dn3)); 
        if abs(dn3)<0.5, chi(ip+1,jp+1)=0.5+dn3; end;
      end
      if d4<d(ip,jp+1), d(ip,jp+1)=d4;...
        dn4=(x(ip)-  xfront)*nfx/dx+(y(jp+1)-yfront)*nfy/dy;
        chi(ip,jp+1)=  0.5*(1.0+sign(dn4)); 
        if abs(dn4)<0.5, chi(ip,jp+1)=  0.5+dn4; end;
      end
    end
              
%-------------------- Update the density -----------------------
    ro=r;
    for i=1:nx+2,for j=1:ny+2
      r(i,j)=rho1+(rho2-rho1)*chi(i,j);
    end,end 

%------------- Set tangential velocity at boundaries -----------	     
    u(1:nx+1,1)=2*usouth-u(1:nx+1,2);u(1:nx+1,ny+2)=2*unorth-u(1:nx+1,ny+1);
    v(1,1:ny+1)=2*vwest-v(2,1:ny+1);v(nx+2,1:ny+1)=2*veast-v(nx+1,1:ny+1);

%-------------- Find the predicted velocities ------------------	     
    for i=2:nx,for j=2:ny+1      % Temporary u-velocity
      ut(i,j)=(2.0/(r(i+1,j)+r(i,j)))*(0.5*(ro(i+1,j)+ro(i,j))*u(i,j)+ dt* (...
      -(0.25/dx)*(ro(i+1,j)*(u(i+1,j)+u(i,j))^2-ro(i,j)*(u(i,j)+u(i-1,j))^2)...
      -(0.0625/dy)*( (ro(i,j)+ro(i+1,j)+ro(i,j+1)+ro(i+1,j+1))*...
                                       (u(i,j+1)+u(i,j))*(v(i+1,j)+v(i,j)) ...
      -(ro(i,j)+ro(i+1,j)+ro(i+1,j-1)+ro(i,j-1))*(u(i,j)...
                                       +u(i,j-1))*(v(i+1,j-1)+v(i,j-1)))...
      +m0*((u(i+1,j)-2*u(i,j)+u(i-1,j))/dx^2+ (u(i,j+1)-2*u(i,j)+u(i,j-1))/dy^2)...
                               + 0.5*(ro(i+1,j)+ro(i,j))*gx ) );
    end,end

    for i=2:nx+1,for j=2:ny       % Temporary v-velocity 
      vt(i,j)=(2.0/(r(i,j+1)+r(i,j)))*(0.5*(ro(i,j+1)+ro(i,j))*v(i,j)+ dt* (...     
      -(0.0625/dx)*( (ro(i,j)+ro(i+1,j)+ro(i+1,j+1)+ro(i,j+1))*...
                                        (u(i,j)+u(i,j+1))*(v(i,j)+v(i+1,j)) ...
                  - (ro(i,j)+ro(i,j+1)+ro(i-1,j+1)+ro(i-1,j))*...
                                    (u(i-1,j+1)+u(i-1,j))*(v(i,j)+v(i-1,j)) )...                                 
      -(0.25/dy)*(ro(i,j+1)*(v(i,j+1)+v(i,j))^2-ro(i,j)*(v(i,j)+v(i,j-1))^2 )...
      +m0*((v(i+1,j)-2*v(i,j)+v(i-1,j))/dx^2+(v(i,j+1)-2*v(i,j)+v(i,j-1))/dy^2)...
                               + 0.5*(ro(i,j+1)+ro(i,j))*gy  ) );    
    end,end   

%------------------ Solve the Pressure Equation ----------------    
    rt=r; lrg=1000;   % Compute source term and the coefficient for p(i,j)
    rt(1:nx+2,1)=lrg;rt(1:nx+2,ny+2)=lrg;
    rt(1,1:ny+2)=lrg;rt(nx+2,1:ny+2)=lrg;

    for i=2:nx+1,for j=2:ny+1
      tmp1(i,j)= (0.5/dt)*( (ut(i,j)-ut(i-1,j))/dx+(vt(i,j)-vt(i,j-1))/dy );
      tmp2(i,j)=1.0/( (1./dx)*(1./(dx*(rt(i+1,j)+rt(i,j)))+   ...
                               1./(dx*(rt(i-1,j)+rt(i,j))) )+ ...
                      (1./dy)*(1./(dy*(rt(i,j+1)+rt(i,j)))+   ...
                               1./(dy*(rt(i,j-1)+rt(i,j))) )   );
    end,end

    for it=1:maxit	               % Solve for pressure by SOR
      oldArray=p;
      for i=2:nx+1,for j=2:ny+1
        p(i,j)=(1.0-beta)*p(i,j)+beta* tmp2(i,j)*(        ...
        (1./dx)*( p(i+1,j)/(dx*(rt(i+1,j)+rt(i,j)))+      ...
                  p(i-1,j)/(dx*(rt(i-1,j)+rt(i,j))) )+ ...
        (1./dy)*( p(i,j+1)/(dy*(rt(i,j+1)+rt(i,j)))+      ...
                  p(i,j-1)/(dy*(rt(i,j-1)+rt(i,j))) ) - tmp1(i,j));
      end,end
      if max(max(abs(oldArray-p))) <maxError, break, end
    end
                                      
    for i=2:nx,for j=2:ny+1   % Correct the u-velocity 
      u(i,j)=ut(i,j)-dt*(2.0/dx)*(p(i+1,j)-p(i,j))/(r(i+1,j)+r(i,j));
    end,end
      
    for i=2:nx+1,for j=2:ny   % Correct the v-velocity
      v(i,j)=vt(i,j)-dt*(2.0/dy)*(p(i,j+1)-p(i,j))/(r(i,j+1)+r(i,j));
    end,end

%--------------- Add and deleate points in the Front -----------
  xfold=xf;yfold=yf; j=1;
  for l=2:Nf+1
    ds=sqrt( ((xfold(l)-xf(j))/dx)^2 + ((yfold(l)-yf(j))/dy)^2);
    if (ds > 0.5)
      j=j+1;xf(j)=0.5*(xfold(l)+xf(j-1));yf(j)=0.5*(yfold(l)+yf(j-1));
      j=j+1;xf(j)=xfold(l);yf(j)=yfold(l);
    elseif (ds < 0.25)
       % DO NOTHING!
    else
      j=j+1;xf(j)=xfold(l);yf(j)=yfold(l);
    end    
  end
  Nf=j-1;
  xf(1)=xf(Nf+1);yf(1)=yf(Nf+1);xf(Nf+2)=xf(2);yf(Nf+2)=yf(2);

%------------------ Plot the results ---------------------------
  time=time+dt                   % plot the results
  uu(1:nx+1,1:ny+1)=0.5*(u(1:nx+1,2:ny+2)+u(1:nx+1,1:ny+1));
  vv(1:nx+1,1:ny+1)=0.5*(v(2:nx+2,1:ny+1)+v(1:nx+1,1:ny+1));
  for i=1:nx+1,xh(i)=dx*(i-1);end;     for j=1:ny+1,yh(j)=dy*(j-1);end
  hold off,contour(x,y,r'),axis equal,axis([0 Lx 0 Ly]);
  hold on;quiver(xh,yh,uu',vv','r');
  plot(xf(1:Nf),yf(1:Nf),'k','linewidth',5);pause(0.01)

end                  % End of time step
