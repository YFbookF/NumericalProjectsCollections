// A beginner Course in boundary element
//https://www.mathworks.com/matlabcentral/fileexchange/32531-laplace-2d-boundary-element-method
// finished
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
boundary_c = np.zeros((N)) # type 0 is phi 1 is dphidn
boundary_v = np.zeros((N)) # value phi or dphidn

for i in range(N):
    x_midpoint[i] = 0.5*(xb[i] + xb[i+1])
    y_midpoint[i] = 0.5*(yb[i] + yb[i+1])
    edge_len[i] = np.sqrt((xb[i+1] - xb[i])**2 + (yb[i+1]-yb[i])**2)
    x_normal[i] = (yb[i+1] - yb[i]) / edge_len[i]
    y_normal[i] = (-xb[i+1] + xb[i]) / edge_len[i]

ABmat = np.zeros((N,N))
BCvec = np.zeros((N))

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
        B = 2.0*edge_len[j]*(-y_normal[j]*(xb[j] - x_midpoint[i]) + x_normal[j]*(yb[j] - y_midpoint[i]))
        E = (xb[j] - x_midpoint[i])**2 + (yb[j] - y_midpoint[i])**2
        D = np.sqrt(abs(4 * A * E - B * B))
        B_A = B / A
        E_A = E / A
        PF1 = 0
        PF2 = 0
        if 4 * A * E - B * B == 0:
            term1 = np.log(edge_len[j])
            term2 = (1 + 0.5 * B_A)*np.log(abs(1 + 0.5 * B_A))
            term3 = - 0.5 * B_A * np.log(abs(0.5 * B_A))
            PF1 = 0.5 * edge_len[j]*(term1 + term2 + term3 - 1) / np.pi
            PF2 = 0
        else:
            term1 = 2 * (np.log(edge_len[j]) - 1)
            term2 = -0.5 * B_A * np.log(abs(E_A))
            term3 = (1 + 0.5 * B_A)*np.log(abs(1 + B_A + E_A))
            term4 = D / A * (math.atan((2 * A + B)/D) - math.atan(B / D))
            PF1 = 0.25 * edge_len[j] * (term1 + term2 + term3 + term4) / np.pi
            term1 = x_normal[j] * (xb[j] - x_midpoint[i]) + y_normal[j] * (yb[j] - y_midpoint[i])
            term2 = math.atan((2 * A + B)/D) - math.atan(B / D)
            PF2 = edge_len[j] * term1 / D * term2 / np.pi
        
        pf1 = PF1
        pf2 = PF2
        delta = 0
        if i == j:
            delta = 1
        
        if boundary_c[j] == 0:
            ABmat[i,j] = - PF1
            BCvec[i] = BCvec[i] + boundary_v[j]*(-PF2 + 0.5 * delta)
        else:
            ABmat[i,j] = PF2 - 0.5 * delta
            BCvec[i] = BCvec[i] + boundary_v[j] * PF1
            
numerical = np.dot(np.linalg.inv(ABmat),BCvec)

exact = np.zeros((N))
phi = np.zeros((N))
dphi = np.zeros((N))
for i in range(N):
    if boundary_c[i] == 0:
        phi[i] = boundary_v[i]
        dphi[i] = numerical[i]
    else:
        phi[i] = numerical[i]
        dphi[i] = boundary_v[i]
        
xi = 0.5
eta = 0.5
summ = 0
for i in range(N):
    summ = 0
    for j in range(N):
        A = edge_len[j]**2
        B = 2.0*edge_len[j]*(-y_normal[j]*(xb[j] - xi) + x_normal[j]*(yb[j] - eta))
        E = (xb[j] - xi)**2 + (yb[j] - eta)**2
        D = np.sqrt(abs(4 * A * E - B * B))
        B_A = B / A
        E_A = E / A
        PF1 = 0
        PF2 = 0
        if 4 * A * E - B * B == 0:
            term1 = np.log(edge_len[j])
            term2 = (1 + 0.5 * B_A)*np.log(abs(1 + 0.5 * B_A))
            term3 = - 0.5 * B_A * np.log(abs(0.5 * B_A))
            PF1 = 0.5 * edge_len[j]*(term1 + term2 + term3 - 1) / np.pi
            PF2 = 0
        else:
            term1 = 2 * (np.log(edge_len[j]) - 1)
            term2 = -0.5 * B_A * np.log(abs(E_A))
            term3 = (1 + 0.5 * B_A)*np.log(abs(1 + B_A + E_A))
            term4 = D / A * (math.atan((2 * A + B)/D) - math.atan(B / D))
            PF1 = 0.25 * edge_len[j] * (term1 + term2 + term3 + term4) / np.pi
            term1 = x_normal[j] * (xb[j] - xi) + y_normal[j] * (yb[j] - eta)
            term2 = math.atan((2 * A + B)/D) - math.atan(B / D)
            PF2 = edge_len[j] * term1 / D * term2 / np.pi
        summ = summ + phi[i]  * PF2 - dphi[i] * PF1
    exact[i] = summ
           
    