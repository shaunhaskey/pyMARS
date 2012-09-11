import data
import matplotlib.pyplot as pt

shot = 146392

channel = 'reflect_3dr'
n = data.Data([channel,'d3d'],shot,save_xext=1)
n_time = n.xext[0]
n_radius = n.xext[1]
n_data = n.y

fig_prof, ax_prof = pt.subplots()

for i in range(0,n_data.shape[0]):
    tmp_rad = n_radius[:,i].flatten()
    tmp_dens = n_data[:,i].flatten()
    ax_prof.plot(tmp_rad, tmp_dens)

ax_prof.set_xlabel('R (m)')
ax_prof.set_ylabel('density')
ax_prof.set_title('%s shot:%d'%(channel, shot))
fig_prof.canvas.draw(); fig_prof.show()

fig_prof.savefig('/u/haskeysr/%s_%d.png'%(channel,shot))
