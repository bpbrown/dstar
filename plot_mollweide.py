"""
Plot constant radius slices from joint analysis files.

Usage:
    plot_mollweide.py <files>... [options]

Options:
    --output=<output>    Output directory; if blank a guess based on likely case name will be made
    --fields=<fields>    Comma separated list of fields to plot [default: ur r0.95,s r0.95]
    --dpi=<dpi>          dpi for image files (if png) [default: 300]
    --fps=<fps>          Frames per second for auto-generated mp4 movie [default: 30]
    --remove_m0          remove m=0 component
"""
import logging
logger = logging.getLogger(__name__.split('.')[-1])
for system in ['matplotlib', 'h5py']:
    dlog = logging.getLogger(system)
    dlog.setLevel(logging.WARNING)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import h5py

def mollweide_plot(phi, theta, data, time=None, index=None, pcm=None, cmap=None, title=None, center_zero=False, norm=1):

    lat, lon = np.meshgrid(theta, phi)
    fig = plt.figure(figsize=(6.5, 3)) #, constrained_layout=True)
    ax = fig.add_subplot(111, projection='mollweide')
    fig.subplots_adjust(left=0.0, bottom=0.025, top=0.975)
    pcm = ax.pcolormesh(lon,lat,data[:,:], cmap=cmap)
    pmin,pmax = pcm.get_clim()
    if center_zero:
        cNorm = matplotlib.colors.TwoSlopeNorm(vmin=pmin, vcenter=0, vmax=pmax)
        logger.info("centering zero: {:.1g} -- 0 -- {:.1g}".format(pmin, pmax))
    else:
        cNorm = matplotlib.colors.Normalize(vmin=pmin, vmax=pmax)
    pcm = ax.pcolormesh(lon,lat,data[:,:], cmap=cmap, norm=cNorm)
    cb_y_off = 0.25
    ax_cb = fig.add_axes([0.9, cb_y_off, 0.02, 1-cb_y_off*2])
    cb = fig.colorbar(pcm, cax=ax_cb, norm=cNorm, cmap=cmap)
    cb.formatter.set_scientific(True)
    cb.formatter.set_powerlimits((0,4))
    cb.ax.yaxis.set_offset_position('left')
    cb.update_ticks()
    #cb = plt.colorbar(mappable=pcm, aspect=10, shrink=0.5, norm=cNorm, cmap=cmap)
    #cb.set_label(label=r'$u_r(r = 0.95)$', fontsize=14)
    #cb.set_label(label='{}'.format(field), fontsize=10)
    #ax_cb.set_title(label=title, fontsize=10)
    ax_cb.text(0.5, 1.25, title, horizontalalignment='center', verticalalignment='center', transform=ax_cb.transAxes)
    ax_cb.text(0.5, -0.25, "t = {:.0f}".format(time/2)+r'$\,\Omega^{-1}$', horizontalalignment='center', verticalalignment='center', transform=ax_cb.transAxes)
    #ax.set_title("t = {:.0f}".format(time),fontsize=12)
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_formatter(plt.NullFormatter())

    return fig, pcm

if __name__ == "__main__":
    import pathlib
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    from docopt import docopt
    args = docopt(__doc__)

    dpi = float(args['--dpi'])

    fields = args['--fields'].split(',')

    if args['--output'] is not None:
        output_path = pathlib.Path(args['--output']).absolute()
    else:
        data_dir = args['<files>'][0].split('/')[0]
        data_dir += '/frames/'

        output_path = pathlib.Path(data_dir).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    logger.info("output to {}".format(output_path))

    def accumulate_files(filename,start,count,file_list):
        print(filename, start, count)
        if start==0:
            file_list.append(filename)

    file_list = []
    post.visit_writes(args['<files>'],  accumulate_files, file_list=file_list)
    logger.info(file_list)

    if len(file_list) > 0:
        for file in file_list:
            print('reading in {:s}'.format(file))
            f = h5py.File(file, 'r')
            t = np.array(f['scales/sim_time'])
            print(f['scales/write_number'][:])
            for k in range(len(t)):
                for i, field in enumerate(fields):
                    time = t
                    center_zero=False
                    title = field
                    task = f['tasks'][field]
                    phi = task.dims[1][0][:] - np.pi
                    theta = np.pi/2 - task.dims[2][0][:]
                    r = task.dims[3][0][:]

                    data_slices = (k, slice(None), slice(None), 0)
                    shell_slice = task[data_slices]
                    center_zero = False
                    title = '{}'.format(field)
                    if field.startswith('u'):
                        cmap = 'plasma'
                        title = field[0]+r'$_'+field[1]+'$\n$' + field.split()[-1][0] + '=' + field.split()[-1][1:] + r'$'
                    elif field.startswith('s'):
                        cmap = 'RdYlBu_r'
                        center_zero = True
                        title = field[0]
                    elif field.startswith('B'):
                        cmap = 'RdYlBu_r'
                        center_zero = True
                        title = field[0]+r'$_'+field[1]+'$\n$' + field.split()[-1][0] + '=' + field.split()[-1][1:] + r'$'
                    else:
                        cmap = 'viridis'
                    if args['--remove_m0']:
                        center_zero = True
                        shell_slice -= np.mean(shell_slice, axis=0, keepdims=True)
                        title += "'"

                    fig, pcm = mollweide_plot(phi,theta, shell_slice,time=t[k], cmap=cmap,center_zero=center_zero,title=title)
                    fig.savefig('{:s}/{:s}_mollweide_{:06d}.png'.format(str(output_path),field.replace(' ','_'),f['scales/write_number'][k]), dpi=dpi)
