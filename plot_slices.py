"""
Plot slices from joint analysis files.

Usage:
    plot_slices.py <files>... [options]

Options:
    --output=<output>    Output directory; if blank a guess based on likely case name will be made
    --fields=<fields>    Comma separated list of fields to plot [default: s_fluc,ur,enstrophy]
    --dpi=<dpi>          dpi for image files (if png) [default: 300]
    --fps=<fps>          Frames per second for auto-generated mp4 movie [default: 30]
    --no_movie           Skip movie making stage
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

def cylinder_plot(r, theta, data, pcm=None, cmap=None):
    #padded_data = np.pad(data, ((0,0),(0,1),(1,0)), mode='edge')
    padded_data = data
    if pcm is None:
        #r_pad = np.pad(r, ((0,0),(1,1)), mode='constant', constant_values=(0,1))
        #theta_pad = np.pad(theta, ((1,1), (0,0)), mode='constant', constant_values=(np.pi,0))
        r_pad = r[None,:]
        theta_pad = theta[:,None]
        z, R = r_pad*np.cos(theta_pad), r_pad*np.sin(theta_pad)

        fig, ax = plt.subplots()
        pcm = ax.pcolormesh(R,z,padded_data[:,:], cmap=cmap)
        ax.set_aspect(1)
        return fig, pcm
    else:
        pcm.set_array(np.ravel(padded_data[:,:-1]))
        pcm.set_clim([np.min(data),np.max(data)])


def equator_plot(r, phi, data, time=None, index=None, pcm=None, cmap=None, title=None, center_zero=False, norm=1):
    padded_data = data*norm #np.pad(data, ((0,0),(1,0)), mode='edge')

    # if pcm is None:
    r_pad = np.pad(r, ((0,1)), mode='constant', constant_values=(0,1))
    phi_pad = np.append(phi, 2*np.pi)

    fig, ax = plt.subplots(subplot_kw=dict(polar=True),figsize=(4, 3))
    r_plot, phi_plot = np.meshgrid(r_pad,phi_pad)
    pcm = ax.pcolormesh(phi_plot,r_plot,padded_data[:,:], cmap=cmap)
    pmin,pmax = pcm.get_clim()

    if center_zero:
        cNorm = matplotlib.colors.TwoSlopeNorm(vmin=pmin, vcenter=0, vmax=pmax)
        logger.info("centering zero: {} -- 0 -- {}".format(pmin, pmax))
    else:
        cNorm = matplotlib.colors.Normalize(vmin=pmin, vmax=pmax)

    pcm = ax.pcolormesh(phi_plot,r_plot,padded_data[:,:], cmap=cmap, vmin=pmin, vmax=pmax, norm=cNorm)
    ax.set_rticks([])
    ax.set_aspect(1)

    cNorm = matplotlib.colors.Normalize(vmin=pmin, vmax=pmax)
    ax_cb = fig.add_axes([0.825, 0.3, 0.03, 1-0.3*2])
    cb = fig.colorbar(pcm, cax=ax_cb, norm=cNorm, cmap=cmap)
    cb.formatter.set_scientific(True)
    cb.formatter.set_powerlimits((0,4))
    cb.ax.yaxis.set_offset_position('left')
    cb.update_ticks()
    fig.subplots_adjust(left=0.025,right=0.8)
    if title is not None:
        ax_cb.text(0.5, 1.25, title, horizontalalignment='center', verticalalignment='center', transform=ax_cb.transAxes)
    if time is not None:
        ax_cb.text(0.5, -0.25, "t = {:.0f}".format(time/2)+r'$\,\Omega^{-1}$', horizontalalignment='center', verticalalignment='center', transform=ax_cb.transAxes)
    pcm.ax_cb = ax_cb
    pcm.cb_cmap = cmap
    pcm.cb = cb
    return fig, pcm
    # else:
    #     pcm.set_array(np.ravel(padded_data[:,:]))
    #     pcm.set_clim([np.min(data),np.max(data)])
    #     cNorm = matplotlib.colors.Normalize(vmin=np.min(data), vmax=np.max(data))
    #     pcm.cb.mappable.set_norm(cNorm)

if __name__ == "__main__":
    import pathlib
    from docopt import docopt
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    import logging
    logger = logging.getLogger(__name__.split('.')[-1])

    args = docopt(__doc__)
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

    dpi = float(args['--dpi'])

    fields = args['--fields'].split(',')

    def accumulate_files(filename,start,count,file_list):
        if start==0:
            file_list.append(filename)

    file_list = []
    post.visit_writes(args['<files>'],  accumulate_files, file_list=file_list)
    logger.info(file_list)
    if len(file_list) > 0:
        for file in file_list:
            print('reading in {:s}'.format(file))
            f = h5py.File(file, 'r')
            r = np.array(f['scales/r/1.5'])
            theta = np.array(f['scales/theta/1.5'])
            phi = np.array(f['scales/phi/1.5'])
            t = np.array(f['scales/sim_time'])
            # pcm_list = []
            # fig_list = []
            # for field in fields:
            #     if field == 's_fluc':
            #         cmap = 'RdYlBu_r'
            #         title="s'"
            #         center_zero=True
            #         time = t
            #     else:
            #         cmap = None
            #         center_zero=False
            #         title=None
            #         time=None
            #     fig, pcm = equator_plot(r,phi,np.array(f['tasks/{:s} eq'.format(field)][0,:,0]),cmap=cmap,center_zero=center_zero,title=title)
            #     pcm_list.append(pcm)
            #     fig_list.append(fig)

            for k in range(len(t)):
                for i, field in enumerate(fields):
                    if field == 's_fluc':
                        cmap = 'RdYlBu_r'
                        title="s'"
                        center_zero=True
                        time = t
                        σ = 0.11510794072958948
                        norm = 1/σ**2
                    else:
                        cmap = None
                        center_zero=False
                        title=None
                        time=None
                        norm=1
                    fig, pcm = equator_plot(r,phi,np.array(f['tasks/{:s} eq'.format(field)][k,:,0]),time=t[k], cmap=cmap,center_zero=center_zero,title=title, norm=norm)#,pcm=pcm_list[i])
                    # fig_list[i].savefig('{:s}/{:s}_eq_{:06d}.png'.format(str(output_path),field,f['scales/write_number'][k]), dpi=dpi)
                    fig.savefig('{:s}/{:s}_eq_{:06d}.png'.format(str(output_path),field,f['scales/write_number'][k]), dpi=dpi)
    import subprocess
    with Sync() as sync:
        if sync.comm.rank == 0 and not args['--no_movie']:
            fps = int(args['--fps'])
            for field in fields:
                print("saving movie {:s}.mp4".format(field))
                movie_process = subprocess.run("png2mp4 {:s}/{:s} {:s}/../{:s}.mp4 {:d}".format(str(output_path), field, str(output_path), field, fps),
                shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                logger.info(movie_process.stderr)
