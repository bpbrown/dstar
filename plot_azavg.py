"""
Plot time-averaged profile of azimuthally averaged quantities.

Usage:
     plot_azavg.py <files>... [options]
     plot_azavg.py [options]

Options:
     <files>...         List of files to compute azavg over.
     --field=<field>    Data to timeaverage and plot [default: <Bφ>]
     --filename=<file>  Filename for output [default: omega.png]
     --title=<title>    Title for plot [default: $<\Omega>$]
     --cmap=<cmap>      Colormap for plot [default: plasma]

     --output=<output>  Location to save figures; if blank, a sensible guess based on <files> will be made.

     --datacube=<datacube>  Pre-existing time-latitude datacube

"""
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dedalus.tools import post


def cylinder_plot(r, theta, data, min=None, max=None, cmap=None, title=None, center_zero=False):
    import matplotlib
    import logging
    logger = logging.getLogger(__name__)

    padded_data = np.pad(data, ((0,1),(1,0)), mode='edge')
    r_pad = np.expand_dims(np.pad(r, ((1,1)), mode='constant', constant_values=(0,1)), axis=0)
    theta_pad = np.expand_dims(np.pad(theta, ((1,1)), mode='constant', constant_values=(np.pi,0)), axis=1)
    theta_pad = np.pi/2 - theta_pad # convert colatitude to latitude for plot labelling

    r_plot, theta_plot = np.meshgrid(r_pad,theta_pad)
    fig, ax = plt.subplots(subplot_kw=dict(polar=True))

    pcm = ax.pcolormesh(theta_plot,r_plot,padded_data, cmap=cmap)
    ax.set_aspect(1)
    ax.set_thetalim(np.pi/2,-np.pi/2)
    ax.set_theta_offset(0)
    ax.set_xticks(np.linspace(-np.pi/2,np.pi/2, num=7))
    pmin,pmax = pcm.get_clim()
    logger.info("data  min {:.4g}, max {:.4g}".format(pmin, pmax))
    if min is not None: pmin=min
    if max is not None: pmax=max
    if center_zero:
        cNorm = matplotlib.colors.TwoSlopeNorm(vmin=pmin, vcenter=0, vmax=pmax)
        logger.info("centering zero: {} -- 0 -- {}".format(pmin, pmax))
    else:
        cNorm = matplotlib.colors.Normalize(vmin=pmin, vmax=pmax)
    pcm = ax.pcolormesh(theta_plot,r_plot,padded_data, cmap=cmap, vmin=pmin, vmax=pmax, norm=cNorm)
    logger.info("image min {:.4g}, max {:.4g}".format(pmin, pmax))

    ax_cb = fig.add_axes([0.8, 0.3, 0.03, 1-0.3*2])
    cb = fig.colorbar(pcm, cax=ax_cb, norm=cNorm, cmap=cmap)
    fig.subplots_adjust(left=0.05,right=0.85)

    r_cont, theta_cont = np.meshgrid(r_pad[:,:-1],theta_pad[:-1,:])
    levels = np.linspace(pmin,pmax,num=11)
    ax.contour(theta_cont, r_cont, padded_data, levels, colors='darkgrey')

    if title is not None:
        ax_cb.set_title(title)

    return fig, pcm

def read_data(files,field):
    start_time = time.time()

    def accumulate_files(filename,start,count,file_list):
        if start==0:
            file_list.append(filename)

    file_list = []
    post.visit_writes(files,  accumulate_files, file_list=file_list)
    logger.info(file_list)

    data = None
    times = None
    r = None
    theta = None

    color = int(len(file_list) > 0)
    comm = MPI.COMM_WORLD.Split(color=color)
    rank = comm.rank
    size = comm.size
    if color:
        for file in file_list:
            logger.debug("opening file: {}".format(file))
            f = h5py.File(file, 'r')
            data_slices = (slice(None), 0, slice(None), slice(None))
            if data is None:
                data = np.array(f['tasks'][field][data_slices])
            else:
                data = np.append(data, f['tasks'][field][data_slices], axis=0)
            theta = f['tasks'][field].dims[2][0][:]
            r = f['tasks'][field].dims[3][0][:]
            if times is None:
                times = f['scales/sim_time'][:]
            else:
                times = np.append(times, f['scales/sim_time'][:])
            f.close()
    else:
        data = np.zeros(1,1,1)
        times = np.zeros(1)

    comm.Barrier()

    n_global_time = np.array(0)
    n_global_data = np.array(0)

    comm.Reduce([np.array(times.size), MPI.INT], [n_global_time, MPI.INT], op=MPI.SUM, root=0)
    comm.Reduce([np.array(data.size), MPI.INT], [n_global_data, MPI.INT], op=MPI.SUM, root=0)

    if rank == 0:
        logger.info("n_time = {}, n_data = {}".format(n_global_time, n_global_data))
        n_times_each = np.empty([size], dtype=np.int)
        n_data_each = np.empty([size], dtype=np.int)
        global_time = np.empty([n_global_time], dtype=np.float64)
        global_data = np.empty([n_global_time,]+list(data[0,:,:].shape),dtype=np.float64)
        logger.debug(n_global_time)
        logger.debug("{}, {}, {}".format(data.shape, global_time.shape, global_data.shape))
    else:
        n_times_each = None
        n_data_each = None
        global_time = None
        global_data = None
        send_counts = None
        displacements = None

    comm.Gather(np.array(times.size), n_times_each, root=0)
    comm.Gather(np.array(data.size), n_data_each, root=0)
    if rank == 0:
        displacements = tuple(np.append(np.zeros(1, dtype=np.int), np.cumsum(n_times_each))[0:-1])
        send_counts = tuple(n_times_each)

    comm.Gatherv(times.astype(np.float64), [global_time, send_counts, displacements, MPI.DOUBLE], root=0)

    if rank == 0:
        displacements = tuple(np.append(np.zeros(1, dtype=np.int), np.cumsum(n_data_each))[0:-1])
        send_counts = tuple(n_data_each)
    comm.Gatherv(data.astype(np.float64), [global_data, send_counts, displacements, MPI.DOUBLE], root=0)

    end_time = time.time()
    logger.info("time to build dataset {:g}sec".format(end_time-start_time))
    return global_data, global_time, theta, r

if __name__ == "__main__":
    import h5py
    import pathlib
    import time
    import dedalus.public as de # parse dedalus.cfg file

    from docopt import docopt
    import matplotlib
    import logging
    logger = logging.getLogger(__name__.split('.')[-1])
    matplotlib_logger = logging.getLogger('matplotlib')
    matplotlib_logger.setLevel(logging.WARNING)

    args = docopt(__doc__)
    if args['--output'] is not None:
        output_path = pathlib.Path(args['--output']).absolute()
    else:
        if '<files>' in args:
            data_dir = args['<files>'][0].split('/')[0]
        elif args['--datacube']:
            data_dir = args['--datacube'].split('/')[0]
        else:
            data_dir = '.'
        data_dir += '/'
        output_path = pathlib.Path(data_dir).absolute()
    logger.info("outputting analysis to {:}".format(output_path))

    field = args['--field']

    if field=='<Bφ>':
        title = r'$\langle B_\phi \rangle$'
        filename = 'Bphi'
        cmap = 'RdYlBu_r'
        center_zero = True
    elif field=='<Aφ>':
        title = r'$\langle A_\phi \rangle$'
        filename = 'Aphi'
        cmap = 'viridis'
        center_zero = True
    else:
        print('taking default choices')
        title = args['--title']
        filename = args['--filename']
        cmap = args['--cmap']
        center_zero = False

    data, times, theta, r = read_data(args['<files>'], field)

    if MPI.COMM_WORLD.rank == 0:
        data_avg = np.mean(data, axis=0)
        if args['--datacube']:
            data_avg = np.expand_dims(data_avg, axis=0)
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
        # TODO: add properly weight average and std-dev (for ball geometry)
        avg = np.mean(data_avg)
        std_dev = np.std(data_avg)
        image_min = max(np.min(data_avg), avg-3*std_dev)
        image_max = min(np.max(data_avg), avg+3*std_dev)
        print(r.shape, theta.shape, data_avg.shape)
        fig_data, pcm = cylinder_plot(r, theta, data_avg, cmap=cmap, title='{:s}'.format(title), min=image_min, max=image_max, center_zero=center_zero)
        fig_data.savefig('{:s}/{:s}'.format(str(output_path), filename))
