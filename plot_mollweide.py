"""
Plot constant radius slices from joint analysis files.

Usage:
    plot_mollweide.py <files>... [options]

Options:
    --output=<output>    Output directory; if blank a guess based on likely case name will be made
    --fields=<fields>    Comma separated list of fields to plot [default: ur r0.95,Br r1.0,Br r0.95]
    --dpi=<dpi>          dpi for image files (if png) [default: 300]
    --fps=<fps>          Frames per second for auto-generated mp4 movie [default: 30]
    --no_movie           Skip movie making stage
"""
import logging
logger = logging.getLogger(__name__.split('.')[-1])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import h5py

from docopt import docopt
args = docopt(__doc__)

fields = args['--fields'].split(',')

def draw_mollweide_frame(data, i, output_dir='.', dpi=300):

    time = data['scales/sim_time'][i]
    phi = data['scales/phi/1.5'][:] - np.pi
    theta = np.pi/2 - data['scales/theta/1.5/'][:]

    lat, lon = np.meshgrid(theta, phi)
    for field in fields:
        center_zero = False
        title = '{}'.format(field)
        if field.startswith('u'):
            cmap = 'plasma'
            title = field[0]+r'$_'+field[1]+'$\n$' + field.split()[-1][0] + '=' + field.split()[-1][1:] + r'$'
        elif field.startswith('B'):
            cmap = 'RdYlBu_r'
            center_zero = True
            title = field[0]+r'$_'+field[1]+'$\n$' + field.split()[-1][0] + '=' + field.split()[-1][1:] + r'$'
        else:
            cmap = 'viridis'

        fig = plt.figure(figsize=(6.5, 3)) #, constrained_layout=True)
        ax = fig.add_subplot(111, projection='mollweide')
        fig.subplots_adjust(left=0.0, bottom=0.025, top=0.975)
        pcm = ax.pcolormesh(lon,lat,data['tasks/{}'.format(field)][i,:,:,0], cmap=cmap)
        pmin,pmax = pcm.get_clim()
        if center_zero:
            cNorm = matplotlib.colors.TwoSlopeNorm(vmin=pmin, vcenter=0, vmax=pmax)
            logger.info("centering zero: {} -- 0 -- {}".format(pmin, pmax))
        else:
            cNorm = matplotlib.colors.Normalize(vmin=pmin, vmax=pmax)
        pcm = ax.pcolormesh(lon,lat,data['tasks/{}'.format(field)][i,:,:,0], cmap=cmap, vmin=pmin, vmax=pmax, norm=cNorm)
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

        file_label = field.replace(' ','_')
        fig.savefig("{}/{}_mollweide_{:06d}.png".format(output_dir,file_label,data['scales/write_number'][i]), dpi=dpi)

if __name__ == "__main__":
    import pathlib
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

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
            data = h5py.File(file, 'r')
            for i in range(len(data['scales/sim_time'])):
                draw_mollweide_frame(data,i, output_dir=str(output_path), dpi=int(args['--dpi']))
            data.close()

    import subprocess
    with Sync() as sync:
        if sync.comm.rank == 0 and not args['--no_movie']:
            fps = int(args['--fps'])
            for field in fields:
                field_name = field.replace(' ','_')
                print("saving movie {:s}.mp4".format(field_name))
                movie_process = subprocess.run("png2mp4 {:s}/{:s} {:s}/../{:s}.mp4 {:d}".format(str(output_path), field_name, str(output_path), field_name, fps),
                shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                logger.info(movie_process.stderr)
