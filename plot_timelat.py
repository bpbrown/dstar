"""
Plot time-latitude maps of azimuthally averaged quantities.

Usage:
     plot_timelat.py <datacube> [options]

Options:
     <datacube>  Pre-existing time-latitude datacube

     --radius=<radius>  Radius to extract from [default: 1]
     --window=<window>  Time window to extract from; format is window=min_t,max_t

     --integrate        Integrate quantity in radius (with r^2 weighting)
     --field=<field>    Data to timeaverage and plot [default: <Bφ>]
     --filename=<file>  Filename for output
     --title=<title>    Title for plot
     --cmap=<cmap>      Colormap for plot [default: plasma]
     --symmetric_cmap   Symmeterize image limits
     --use_std_dev_avg  Set image limits using the standard deviation of the average profile, rather than full time-sequence

     --output=<output>  Location to save figures; if blank, a sensible guess based on <files> will be made.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docopt import docopt
import time
import h5py
import pathlib
import glob
import os
from dedalus.tools import post
import dedalus.public as de

import logging
logger = logging.getLogger(__name__)
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

from plot_azavg import cylinder_plot

args = docopt(__doc__)
if args['--output'] is not None:
    output_path = pathlib.Path(args['--output']).absolute()
else:
    data_dir = args['<datacube>'].split('/')[0]
    data_dir += '/'
    output_path = pathlib.Path(data_dir).absolute()

field = args['--field']
target_radius = float(args['--radius'])

if field=='<Ωz>':
    title = r'$\langle \Omega \rangle$'
    filename = 'omega'
    cmap = 'plasma'
    center_zero = False
elif field=='<Bφ>':
    title = r'$\langle B_\phi \rangle$'
    filename = 'Bphi'
    cmap = 'RdYlBu_r'
    center_zero = True
elif field=='<s>':
    title = r'$\langle s \rangle$'
    filename = 's'
    cmap = 'RdYlBu_r'
    center_zero = False
elif field=='<Aφ>':
    title = r'$\langle A_\phi \rangle$'
    filename = 'Aphi'
    cmap = 'PRGn'
    center_zero = True
else:
    print('taking default choices')
    title = args['--title']
    filename = args['--filename']
    cmap = args['--cmap']
    center_zero = False

#if args['--datacube']:
datacube_filename = args['<datacube>']
# else:
#     datacube_filename = data_dir+'time_lat_datacube.h5'

start_time = time.time()
f_cube = h5py.File(datacube_filename,'r')
r = f_cube['scales/r'][:]
theta = f_cube['scales/theta/'][:]
times = f_cube['scales/sim_time'][:]
data = f_cube['tasks/{:s}'.format(field)][:]
f_cube.close()
end_time = time.time()
print("time to read datacube {:g}sec".format(end_time-start_time))

logger.info("data shape on read in {}".format(data.shape))

if args['--window']:
    t_min, t_max = [float(t) for t in args['--window'].split(',')]
    i_min = np.argmin(np.abs(times-t_min))
    i_max = np.argmin(np.abs(times-t_max))
    data = data[i_min:i_max,...]
    times = times[i_min:i_max]
    logger.info("windowing data to {}--{}".format(times[0], times[-1]))

data_avg = np.mean(data, axis=0)

logger.info("averaged from t={:.3g}--{:.3g} ({:3g} rotations)".format(min(times),max(times), (max(times)-min(times))/(4*np.pi)))
# import dedalus_sphere.ball_wrapper as ball
# L_max = N_theta = theta.shape[0]
# N_max = N_r     = r.shape[0]
# R_max = 0
# B = ball.Ball(N_max,L_max,N_theta=N_theta,N_r=N_r,R_max=R_max,ell_min=0,ell_max=N_theta-1,m_min=0,m_max=0,a=0.)
# weight_theta = B.weight(1,dimensions=3)
# weight_r = B.weight(2,dimensions=3)
# avg = np.sum(data_avg*weight_theta*weight_r)/np.sum(weight_theta*weight_r)
# std_dev = np.sum(np.abs(data_avg-avg)*(weight_theta*weight_r))/np.sum(weight_theta*weight_r)
avg = np.mean(data_avg)
std_dev = np.std(data_avg)
image_min = max(np.min(data_avg), avg-3*std_dev)
image_max = min(np.max(data_avg), avg+3*std_dev)

if center_zero and image_min > 0:
    image_min = -1*image_max
if center_zero and image_max < 0:
    image_max = -1*image_min
print(image_min, image_max)
fig_data, pcm = cylinder_plot(r, theta, data_avg, cmap=cmap, title='{:s}'.format(title), min=image_min, max=image_max, center_zero=center_zero)
fig_data.savefig('{:s}/{:s}_{:.0f}_{:.0f}.png'.format(str(output_path), filename, times[0], times[-1]), dpi=600)

# import dedalus_sphere.ball_wrapper as ball
# L_max = N_theta = theta.shape[0]
# N_max = N_r     = r.shape[0]
# R_max = 0
# B = ball.Ball(N_max,L_max,N_theta=N_theta,N_r=N_r,R_max=R_max,ell_min=0,ell_max=N_theta-1,m_min=0,m_max=0,a=0.)
# weight_theta = B.weight(1,dimensions=3)
# weight_r = B.weight(2,dimensions=3)

print('data cube shape: {}'.format(data.shape))
data = np.squeeze(data)
if args['--integrate']:
    data = np.sum(data*r**2, axis=2)
    print('data integrated across all radii')
else:
    print(data.shape)
    i_radius = np.argmin(np.abs(r-target_radius))
    data = data[:,:,i_radius]
    print('data extracted at radius={} with index {:d}'.format(target_radius, i_radius))

data_avg = np.expand_dims(np.mean(data, axis=0), axis=0)
logger.info(data_avg.shape)
print("averaged from t={:.3g}--{:.3g} ({:3g} rotations)".format(min(times),max(times), (max(times)-min(times))/(4*np.pi)))
n_times = times.shape[0]
# weight_theta = weight_theta[:,:,0]
# avg = np.sum(data_avg*weight_theta)/np.sum(weight_theta)
# if args['--use_std_dev_avg']:
#     std_dev = np.sum(np.abs(data_avg-avg)*(weight_theta))/np.sum(weight_theta)
# else:
#     std_dev = np.sum(np.abs(data-avg)*(weight_theta))/np.sum(weight_theta)/n_times

avg = np.mean(data)
std_dev = np.std(data)
image_min = avg-3*std_dev
image_max = avg+3*std_dev

# if args['--symmetric_cmap']:
#     if image_max > 0 and image_min < 0:
#         image_max = max(np.abs(image_max), np.abs(image_min))
#         image_min = -1*image_max

print("number of times: {:d}".format(n_times))
#print("weight_theta.shape: {}".format(weight_theta.shape))
print("data_avg.shape: {}".format(data_avg.shape))
print("data.shape: {}".format(data.shape))
#print("avg: {:g}, std dev: {:g}".format(avg,std_dev))
print("data     min/max: {:g}/{:g}".format(np.min(data), np.max(data)))
print("data_avg min/max: {:g}/{:g}".format(np.min(data_avg), np.max(data_avg)))
print("image    min/max: {:g}/{:g}".format(image_min, image_max))
print(times)
fig, ax = plt.subplots()
lat = 180*(np.pi/2-theta)/np.pi

print(times.shape, lat.shape, data.T.shape, theta.shape)

ax.pcolormesh(times, lat, data.T, vmin=image_min, vmax=image_max, cmap=cmap)
ax.set_ylabel('latitude')
ax.set_xlabel('time')
ax.set_title(title)
fig.savefig('{:s}/time_lat_{:s}_{:.0f}_{:.0f}.png'.format(str(output_path), filename, times[0], times[-1]), dpi=600)
