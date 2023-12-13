#!/usr/bin/env python3
"""
2-D fluid simulator
===================

Based on the excellent code and video from Ten Minute Physics:
    - https://www.youtube.com/watch?v=iKAVRgIrUOU
    - https://github.com/matthias-research/pages/blob/master/tenMinutePhysics/17-fluidSim.html
"""

import numpy as np

class Fluid:
    """2-D representation of fluid field, using matrix-style coordinate frame: origin is in the top left, with x down
    and y across"""
    def __init__(self, density, grid_size, grid_spacing, overrelaxation):
        self.density = density
        self.num_r = grid_size[0] + 2
        self.num_c = grid_size[1] + 2
        self.num_cells = self.num_r * self.num_c
        self.grid_spacing = grid_spacing # h
        self.h_vel = np.zeros((self.num_r, self.num_c)) # u
        self.v_vel = np.zeros((self.num_r, self.num_c)) # v
        self.pressure = np.zeros((self.num_r, self.num_c)) # p
        self.solid = np.zeros((self.num_r, self.num_c), dtype=bool) # s, True = solid, False = fluid; in ref. implementation 1=fluid, 0=solid
        self.smoke = np.ones((self.num_r, self.num_c))
        self.overrelaxation = overrelaxation

    def integrate(self, dt, gravity):
        for i in range(self.num_r):
            for j in range(self.num_c):
                if not self.solid[i, j] and (i > 0 and not self.solid[i-1, j]):
                    self.v_vel[i, j] += gravity * dt

    def solve_incompressibility(self, num_iters, dt):
        cp = self.density * self.grid_spacing / dt # TODO(jstech): better name

        for it in range(num_iters):
            for i in range(1, self.num_r - 1):
                for j in range(1, self.num_c - 1):
                    if self.solid[i, j]:
                        continue
                    s_up = self.solid[i-1, j];
                    s_dn = self.solid[i+1, j];
                    s_lf = self.solid[i, j-1];
                    s_rt = self.solid[i, j+1];
                    num_solid_edges = s_up + s_dn + s_lf + s_rt
                    if 4 == num_solid_edges: continue

                    divergence = (self.v_vel[i+1, j] - self.v_vel[i, j] +
                                  self.h_vel[i, j+1] - self.h_vel[i, j])

                    pressure = self.overrelaxation * (-divergence / (4 - num_solid_edges))
                    self.pressure[i, j] += cp * pressure
                    if not s_up: self.v_vel[i, j] -= pressure
                    if not s_dn: self.v_vel[(i+1), j] += pressure
                    if not s_lf: self.h_vel[i, j] -= pressure
                    if not s_rt: self.h_vel[i, j+1] += pressure

    def extrapolate(self):
        #TODO: vectorize
        for i in range(self.num_r):
            self.h_vel[i, 0] = self.h_vel[i, 1]
            self.h_vel[i, self.num_c - 1] = self.h_vel[i, self.num_c - 2]
        for j in range(self.num_c):
            self.v_vel[0, j] = self.v_vel[1, j]
            self.v_vel[self.num_r - 1, j] = self.v_vel[self.num_r - 2, j]

    def sample_field(self, x, y, field):
        """x down, y right"""
        h = self.grid_spacing
        inv_h = 1./h
        half_h = 0.5 * h
        x = max(min(x, self.num_r * h), h)
        y = max(min(y, self.num_c * h), h)
        dx = half_h
        dy = half_h

        if "U_FIELD" == field:
            f = self.h_vel
            dx = 0.
        elif "V_FIELD" == field:
            f = self.v_vel
            dy = 0.
        elif "S_FIELD" == field:
            f = self.smoke

        x0 = min(int((x - dx) * inv_h), self.num_r - 1)
        tx = ((x - dx) - x0*h) * inv_h
        x1 = min(x0 + 1, self.num_r - 1)

        y0 = min(int((y - dy) * inv_h), self.num_c - 1)
        ty = ((y - dy) - y0*h) * inv_h
        y1 = min(y0 + 1, self.num_c - 1)

        sx = 1. - tx
        sy = 1. - ty

        return (sx*sy * f[x0, y0] +
                tx*sy * f[x1, y0] +
                tx*ty * f[x1, y1] +
                sx*ty * f[x0, y1])

    def avg_u(self, i, j):
        return (
                self.h_vel[i, j-1] +
                self.h_vel[i, j] +
                self.h_vel[i+1, j-1] +
                self.h_vel[i+1, j]) * 0.25

    def avg_v(self, i, j):
        return (self.v_vel[i-1, j] +
                self.v_vel[i, j] +
                self.v_vel[i-1, j+1] +
                self.v_vel[i, j+1]) * 0.25

    def advect_vel(self, dt):
        new_h_vel = np.zeros(self.h_vel.shape)
        new_v_vel = np.zeros(self.v_vel.shape)
        h = self.grid_spacing
        half_h = 0.5 * h

        for i in range(1, self.num_r):
            for j in range(1, self.num_c):
                if self.solid[i, j]: continue
                if not self.solid[i-1, j] and i < self.num_r - 1:
                    x = i*h
                    y = j*h + half_h
                    u = self.avg_u(i, j)
                    v = self.v_vel[i, j]
                    x = x - dt*u
                    y = y - dt*v
                    u = self.sample_field(x, y, "U_FIELD")
                    new_v_vel[i, j] = u
                if not self.solid[i, j-1] and j < self.num_c - 1:
                    x = i*h + half_h
                    y = j*h
                    u = self.h_vel[i, j]
                    v = self.avg_v(i, j)
                    x = x - dt*u
                    y = y - dt*v
                    v = self.sample_field(x, y, "V_FIELD")
                    new_h_vel[i, j] = v
        self.h_vel = new_h_vel
        self.v_vel = new_v_vel

    def advect_smoke(self, dt):
        new_smoke = np.zeros(self.smoke.shape)
        h = self.grid_spacing
        h2 = 0.5 * h

        for i in range(1, self.num_r-1):
            for j in range(1, self.num_c-1):
                if self.solid[i, j]: continue
                u = (self.h_vel[i, j] + self.h_vel[i, j+1]) * 0.5
                v = (self.v_vel[i, j] + self.v_vel[i+1, j]) * 0.5
                x = i*h + h2 - dt*u
                y = j*h + h2 - dt*v
                new_smoke[i, j] = self.sample_field(x, y, "S_FIELD")
        self.smoke = new_smoke

    def simulate(self, dt, gravity, num_iters):
        self.integrate(dt, gravity)
        self.pressure.fill(0.)
        self.solve_incompressibility(num_iters, dt)
        self.extrapolate()
        self.advect_vel(dt)
        self.advect_smoke(dt)

if __name__=="__main__":
    import matplotlib.pyplot as plt
    _num_r = 20
    _num_c = 30
    _grid_spacing = 0.1
    _density = 1000
    _overrelaxation = 1.9
    _gravity = 0.
    _dt = 1/120
    _num_iters = 60
    fluid = Fluid(_density, (_num_r, _num_c), _grid_spacing, _overrelaxation)

    circle = ((1., 1.), 0.3)

    for i in range(fluid.num_r):
        for j in range(fluid.num_c):
            fluid.solid[i, j] = (i == 0 or j in (0, fluid.num_c - 1))
            if j==1:
                fluid.h_vel[i, j] = 2.
            dx = (i + 0.5) * fluid.grid_spacing - circle[0][0]
            dy = (j + 0.5) * fluid.grid_spacing - circle[0][1]

            if dx**2 + dy**2 < circle[1]**2:
                fluid.solid[i, j] = True
                fluid.h_vel[i, j] = 0
                fluid.h_vel[(i+1), j] = 0
                fluid.v_vel[i, j] = 0
                fluid.v_vel[i, j+1] = 0

    for i in range(5):
        plt.subplot(2, 2, 1)
        plt.imshow(fluid.h_vel.reshape((fluid.num_r, fluid.num_c)))
        plt.colorbar()
        plt.title("Horizontal velocity")
        plt.subplot(2, 2, 2)
        plt.imshow(fluid.v_vel.reshape((fluid.num_r, fluid.num_c)))
        plt.colorbar()
        plt.title("Vertical velocity")
        plt.subplot(2, 2, 3)
        plt.imshow(fluid.solid.reshape((fluid.num_r, fluid.num_c)))
        plt.colorbar()
        plt.title("Solid")
        plt.subplot(2, 2, 4)
        plt.imshow(fluid.pressure.reshape((fluid.num_r, fluid.num_c)))
        plt.colorbar()
        plt.title("Pressure")
        plt.show()
        fluid.simulate(_dt, _gravity, _num_iters)

