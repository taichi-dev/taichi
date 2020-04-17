import taichi as ti
import matplotlib.cm as cm
# A simple compressible euler equation solver using a 2nd order muscl method

real = ti.f32
ti.init(arch=ti.cuda, default_fp=real)

N = 512 # grid resolution
CFL = .8
IC_type = 0  # 0:sod
BC_type = 0 # 0:walls
img_field = 0 # 0:density, 1: schlieren, 2:vorticity, 3: velocity mag, 4: temperature
res = 1024  # gui resolution
cmap_name = 'magma_r' # python colormap


Q = ti.Vector(4, dt=real, shape=(N,N)) # [rho, rho*u, rho*v, rho*e] consv vars
Q_old = ti.Vector(4, dt=real, shape=(N,N))
W = ti.Vector(4, dt=real, shape=(N,N)) # [rho, u, v, p] primtive vars
F_x = ti.Vector(4, dt=real, shape=(N,N)) # x-face flux vector
F_y = ti.Vector(4, dt=real, shape=(N,N)) # y-face flux vector
dt = ti.var(dt=real, shape=())
img = ti.var(dt=ti.f32, shape=(res,res))

gamma = 1.4 # ratio of specific heats
h = 1.0/(N-2) # cell size
vol = h*h  # cell volume

@ti.func
def is_interior_cell(i,j):
    return 0 < i < N-1 and 0 < j < N-1

@ti.func
def is_interior_x_face(i,j):
    return 1 < i < N-1 and 0 < j < N-1

@ti.func
def is_boundary_x_face(i,j):
    return (i == 1 or i == N-1) and 0 < j < N-1

@ti.func
def is_interior_y_face(i,j):
    return 0 < i < N-1 and 1 < j < N-1

@ti.func
def is_boundary_y_face(i,j):
    return 0 < i < N-1 and (j == 1 or j == N-1)

@ti.func
def get_cell_pos(i,j):
    return ti.Vector([i*h - h/2.0, j*h - h/2.0])

@ti.kernel
def compute_W():
    # conversion from conservative variables to primitive variables
    for i,j in Q:
        W[i,j] = q_to_w(Q[i,j])


@ti.kernel
def copy_to_old():
    for i,j in Q:
        Q_old[i,j] = Q[i,j]


@ti.kernel
def set_ic():
    for i,j in Q:
        if IC_type == 0:
            # primitive variable initial conditions
            w_in = ti.Vector([10.0,0.0,0.0,10.0])
            w_out = ti.Vector([.125,0.0,0.0,.1])

            pos = get_cell_pos(i,j)
            center = ti.Vector([.5,.5])

            if (pos-center).norm() < .25:
                Q[i,j] = w_to_q(w_in)
            else:
                Q[i,j] = w_to_q(w_out)

        # implement more ic's later


@ti.kernel
def set_bc():
    # enforce boundary conditions by setting ghost cells
    for i,j in Q:
        if not is_interior_cell(i,j):
            if BC_type == 0: # walls
                # enforce neumann=0 and zero normal velocity on face
                if i == 0:
                    Q[i,j] = Q[i+1,j]
                    Q[i,j][1] = -Q[i+1,j][1]
                if i == N-1:
                    Q[i,j] = Q[i-1,j] # neumann 0 bc
                    Q[i,j][1] = -Q[i-1,j][1] # enforce 0 normal velocty at face
                if j == 0:
                    Q[i,j] = Q[i,j+1]
                    Q[i,j][2] = -Q[i,j+1][2]
                if j == N-1:
                    Q[i,j] = Q[i,j-1]
                    Q[i,j][2] = -Q[i,j-1][2]

            # implement more bc's later

@ti.func
def mc_lim(r):
    # MC flux limiter
    return max(0.0, min(2.0*r, min(.5*(r+1.0), 2.0)))

@ti.func
def w_to_q(w):
    # convert primitive variables to conserved variables
    q = ti.Vector([0.0,0.0,0.0,0.0])
    q[0] = w[0] # rho
    q[1] = w[0]*w[1] # rho*u
    q[2] = w[0]*w[2] # rho*v
    q[3] = w[0]*(w[3]/((gamma-1)*w[0]) + 0.5*(w[1]**2 + w[2]**2)); # rho*e
    return q

@ti.func
def q_to_w(q):
    # convert conserved variables to primitive variables
    w = ti.Vector([0.0,0.0,0.0,0.0])
    w[0] = q[0]   # rho
    w[1] = q[1]/q[0] # u
    w[2] = q[2]/q[0] # v
    w[3] = (gamma-1)*(q[3] - 0.5*(q[1]**2 + q[2]**2)/q[0]); # p
    return w


@ti.func
def HLLC_flux(qL, qR, n):

    # normal vector
    nx = n[0];
    ny = n[1];

    # Left state
    rL = qL[0] # rho
    uL = qL[1]/qL[0] # u
    vL = qL[2]/qL[0] # v
    pL = (gamma-1.0)*(qL[3] - 0.5*(qL[1]**2 + qL[2]**2)/qL[0]); #p
    vnL = uL*nx + vL*ny
    vtL = -uL*ny + vL*nx
    aL = ti.sqrt(gamma*pL/rL)
    HL = ( qL[3] + pL) / rL

    # Right state
    rR = qR[0] # rho
    uR = qR[1]/qR[0] # u
    vR = qR[2]/qR[0] # v
    pR = (gamma-1.0)*(qR[3] - 0.5*(qR[1]**2 + qR[2]**2)/qR[0]); #p
    vnR = uR*nx + vR*ny
    vtR = -uR*ny + vR*nx
    aR = ti.sqrt(gamma*pR/rR)
    HR = ( qR[3] + pR ) / rR

    # Left and Right fluxes
    fL = ti.Vector([rL*vnL, rL*vnL*uL + pL*nx, rL*vnL*vL + pL*ny, rL*vnL*HL])
    fR = ti.Vector([rR*vnR, rR*vnR*uR + pR*nx, rR*vnR*vR + pR*ny, rR*vnR*HR])

    # Roe Averages
    rt = ti.sqrt(rR/rL)
    u = (uL+rt*uR)/(1.0+rt)
    v = (vL+rt*vR)/(1.0+rt)
    H = (HL+rt*HR)/(1.0+rt)
    a = ti.sqrt( (gamma-1.0)*(H-(u**2+v**2)/2.0))
    vn = u*nx+v*ny

    # wavespeeds
    sL = min(vnL - aL, vn - a)
    sR = max(vnR + aR, vn + a)
    sM = (pL-pR + rR*vnR*(sR-vnR) - rL*vnL*(sL-vnL))/(rR*(sR-vnR) - rL*(sL-vnL))

    # HLLC flux.
    HLLC = ti.Vector([0.0,0.0,0.0,0.0])
    if (0 <= sL):
        HLLC = fL;
    elif (sL <= 0) and (0 <= sM):
        qsL = rL * (sL-vnL)/(sL-sM) \
                  * ti.Vector([1.0, sM*nx-vtL*ny,sM*ny+vtL*nx, \
                               qL[3]/rL + (sM-vnL)*(sM+pL/(rL*(sL-vnL)))])
        HLLC = fL + sL*(qsL - qL)
    elif (sM <= 0) and (0 <=sR):
        qsR = rR * (sR-vnR)/(sR-sM) \
                   * ti.Vector([1.0, sM*nx-vtR*ny,sM*ny+vtR*nx, \
                                qR[3]/rR + (sM-vnR)*(sM+pR/(rR*(sR-vnR)))])
        HLLC = fR + sR*(qsR - qR)
    elif (0 >= sR):
        HLLC = fR

    return HLLC

@ti.kernel
def compute_F():
    for i,j in Q:
        if is_interior_x_face(i,j):
            # muscl reconstrucion of left and right states with HLLC flux
            wL = ti.Vector([0.0,0.0,0.0,0.0])
            wR = ti.Vector([0.0,0.0,0.0,0.0])
            for f in ti.static(range(4)):
                ratio_l = (W[i,j][f] - W[i-1,j][f])/(W[i-1,j][f] - W[i-2,j][f])
                ratio_r = (W[i,j][f] - W[i-1,j][f])/(W[i+1,j][f] - W[i,j][f])
                wL[f] = W[i-1,j][f] + 0.5*mc_lim(ratio_l) * (W[i-1,j][f] - W[i-2,j][f]);
                wR[f] = W[i,j][f] - 0.5*mc_lim(ratio_r) * (W[i+1,j][f] - W[i,j][f]);
            F_x[i,j] = HLLC_flux(w_to_q(wL), w_to_q(wR), ti.Vector([1.0,0.0]))

        elif is_boundary_x_face(i,j):
            F_x[i,j] = HLLC_flux(Q[i-1,j], Q[i,j], ti.Vector([1.0,0.0]))

        if is_interior_y_face(i,j):
            # muscl reconstrucion of left and right states with HLLC flux
            wL = ti.Vector([0.0,0.0,0.0,0.0])
            wR = ti.Vector([0.0,0.0,0.0,0.0])
            for f in ti.static(range(4)):
                ratio_l = (W[i,j][f] - W[i,j-1][f])/(W[i,j-1][f] - W[i,j-2][f])
                ratio_r = (W[i,j][f] - W[i,j-1][f])/(W[i,j+1][f] - W[i,j][f])
                wL[f] = W[i,j-1][f] + 0.5*mc_lim(ratio_l) * (W[i,j-1][f] - W[i,j-2][f]);
                wR[f] = W[i,j][f] - 0.5*mc_lim(ratio_r) * (W[i,j+1][f] - W[i,j][f]);
            F_y[i,j] = HLLC_flux(w_to_q(wL), w_to_q(wR), ti.Vector([0.0,1.0]))

        elif is_boundary_y_face(i,j):
            F_y[i,j] = HLLC_flux(Q[i,j-1], Q[i,j], ti.Vector([0.0,1.0]))

@ti.kernel
def calc_dt():
    dt[None] = 1.0e5
    for i,j in Q:
        w = q_to_w(Q[i,j])
        a = ti.sqrt(gamma*w[3]/w[0])
        vel = ti.sqrt(w[1]**2 + w[2]**2)
        ws = a + vel
        ti.atomic_min(dt[None], CFL*h/ws/2.0)

@ti.kernel
def update_Q(rk_step: ti.template()):
    for i,j in Q:
        if is_interior_cell(i,j):
            if ti.static(rk_step == 0):
                Q[i,j] = Q[i,j] + dt[None]*(F_x[i,j] - F_x[i+1,j] + F_y[i,j] - F_y[i,j+1])/h
            if ti.static(rk_step == 1):
                Q[i,j] = (Q[i,j] + Q_old[i,j])/2.0 + dt[None]*(F_x[i,j] - F_x[i+1,j] + F_y[i,j] - F_y[i,j+1])/h

@ti.kernel
def paint():
    for i,j in img:
        ii = min(max(1,i*N//res),N-2)
        jj = min(max(1,j*N//res),N-2)
        if img_field == 0: # density
            img[i,j] = Q[ii,jj][0]/5
        elif img_field == 1: # numerical schlieren
            img[i,j] = ti.log(ti.sqrt(((Q[ii+1,jj][0]-Q[ii-1,jj][0])/h)**2 + ((Q[ii,jj+1][0]-Q[ii,jj-1][0])/h)**2))
        elif img_field == 2: # vorticity
            img[i,j] = (Q[ii+1,jj][2]-Q[ii-1,jj][2])/h - (Q[ii,jj+1][1]-Q[ii,jj-1][1])/h
        elif img_field == 3: # velocity magnitude
            img[i,j] = ti.sqrt(Q[ii,jj][1]**2 + Q[ii,jj][2]**2)
        elif img_field == 3: # temperature
            w = q_to_w(Q[ii,jj])
            img[i,j] = w[0]/(Q[ii,jj][0]*287.0)


gui = ti.GUI('Euler Equations', (res, res))
cmap = cm.get_cmap(cmap_name)
set_ic()
set_bc()

n = 0
while(1):
    calc_dt()
    copy_to_old()
    for rk_step in range(2):
        compute_W()
        compute_F()
        update_Q(rk_step)
        set_bc()

    if n%10 == 0:
        paint()
        gui.set_image(cmap(img.to_numpy()))
        gui.show()
    n += 1
