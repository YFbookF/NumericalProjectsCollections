// A beginner Course in boundary element
//https://www.mathworks.com/matlabcentral/fileexchange/32531-laplace-2d-boundary-element-method
import numpy as np
import math
NO = 5 # boundary elements per side
N = 4 * NO # boundary element num of four side
dl = 1 / NO # grid size
xb = np.zeros((N + 1)) # coordinates
yb = np.zeros((N + 1)) # coordinates
for i in range(NO):
    xb[i] = i * dl
    yb[i] = 0
    xb[i + NO] = 1
    yb[i + NO] = xb[i]
    xb[i + 2*NO] = 1 - xb[i]
    yb[i + 2*NO] = 1
    xb[i + 3*NO] = 0
    yb[i + 3*NO] = 1 - xb[i]

xb[N] = xb[0]
yb[N] = yb[0]
x_midpoint = np.zeros((N)) 
y_midpoint = np.zeros((N)) 
edge_len = np.zeros((N))
x_normal = np.zeros((N))
y_normal = np.zeros((N))

boundary = np.zeros((N))
boundary_c = np.zeros((N))
boundary_v = np.zeros((N))

for i in range(N):
    x_midpoint[i] = 0.5*(xb[i] + xb[i+1])
    y_midpoint[i] = 0.5*(yb[i] + yb[i+1])
    edge_len[i] = np.sqrt((xb[i+1] - xb[i])**2 + (yb[i+1]-yb[i])**2)
    x_normal = (yb[i+1] - yb[i]) / edge_len[i]
    y_normal = (-xb[i+1] + xb[i]) / edge_len[i]

for i in range(N):
    if i < NO:
        boundary_c[i] = 1
        boundary_v[i] = 0
    elif i < NO*2:
        boundary_c[i] = 0
        boundary_v[i] = np.cos(np.pi * y_midpoint[i])
    elif i < NO*3:
        boundary_c[i] = 1
        boundary_v[i] = 0
    else:
        boundary_c[i] = 0
        boundary_v[i] = 0
        
for i in range(N):
    boundary[i] = 0
    for j in range(N):
        A = edge_len[j]**2
        B = 2*edge_len[j]*(-y_normal[j]*(xb[j] - x_midpoint[j]) + x_normal[j]*(yb[j] - y_midpoint[i]))
        E = (xb[j] - x_midpoint[i])**2 + (yb[j] - y_midpoint[i])**2
        D = np.sqrt(abs(4 * A * E - B * B))
        B_A = B / A
        E_A = E / A
        PF1 = 0
        PF2 = 0
        if 4*A(k)*E(k)-B(k)^2 == 0
            PF1(k)=0.5*lg(k)*(log(lg(k))+(1+0.5*BA(k))*log(abs(1+0.5*BA(k)))-0.5*BA(k)*log(abs(0.5*BA(k)))-1);
            PF2(k)=0;
        else
            PF1(k)=0.25*lg(k)*(2*(log(lg(k))-1)-0.5*BA(k)*log(abs(EA(k)))+
                               (1+0.5*BA(k))*log(abs(1+BA(k)+EA(k)))+(D(k)/A(k))*(atan((2*A(k)+B(k))/D(k))-atan(B(k)/D(k))));
            PF2(k)=lg(k)*(nx(k)*(xb(k)-xm(m))+ny(k)*(yb(k)-ym(m)))/D(k)*(atan((2*A(k)+B(k))/D(k))-atan(B(k)/D(k)));
        end
        if 4 * A * E - B * B == 0:
            term1 = np.log(edge_len[j])
            term2 = (1 + 0.5 * B_A)*np.log(abs(1 + 0.5 * B_A))
            term3 = - 0.5 * B_A * np.log(abs(0.5 * B_A))
            PF1 = 0.5 * edge_len[j]*(term1 + term2 + term3 - 1)
            PF2 = 0
        else:
            term1 = 2 * (np.log(edge_len[j]) - 1)
            term2 = -0.5 * B_A * np.log(abs(E_A))
            term3 = (1 + 0.5 * B_A)*np.log(abs(1 + B_A + E_A))
            term4 = D / A * (math.atan((2 * A + B)/D) - math.atan(B / D))
            PF1 = 0.25 * edge_len[j] * (term1 + term2 + term3 + term4)
            term1 = x_normal[j] * (xb[j] - x_midpoint[i]) + y_normal[j] * (yb[j] - y_normal[i])
            term2 = math.atan((2 * A + B)/D) - math.atan(B / D)
            PF2 = edge_len[j] * term1 / D / term2
           
    